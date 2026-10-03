"""
FastAPI Backend for AI Data Analyst.
Serves analytical APIs, file ingestion, deterministic orchestration, and report generation
to the React.js client interface.

Primary UI: React SPA (frontend/) served from /.
Legacy Streamlit UI (backend/app.py) is deprecated and no longer part of the served stack.
Sessions: isolated per X-Session-Id header via SessionManager (defaults to "default").
"""

import json as jsonlib
import os
from typing import List, Optional
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.agent.analyst import DataAnalystAgent
from src.agent.sessions import session_manager
from src.agent.state import SessionState
from src.models.schemas import AgentResponse
from src.services.llm import LLMService, LLMSettings
from src.tools.dashboard import build_dashboard_data
from src.tools.export import generate_executive_html_report
from src.utils.csv import read_csv_bytes
from src.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("api.server")

app = FastAPI(
    title="AI Data Analyst API",
    description="Deterministic Data Analysis backend powered by NVIDIA NIM & DuckDB",
    version="1.0.0",
)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_session(
    session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    sid: Optional[str] = None,
) -> tuple[str, SessionState]:
    """Resolve per-client SessionState from X-Session-Id (isolated, thread-safe)."""
    return session_manager.get_or_create(session_id or sid)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ChatRequest(BaseModel):
    query: str
    provider: str = "nvidia"
    api_key: Optional[str] = None
    model: str = "meta/muse-glimmer-30b"
    base_url: str = "https://integrate.api.nvidia.com/v1"


class DatasetSelectRequest(BaseModel):
    dataset_name: str


class LlmTestRequest(BaseModel):
    provider: str = "nvidia"
    api_key: Optional[str] = None
    model: str = "meta/muse-glimmer-30b"
    base_url: str = "https://integrate.api.nvidia.com/v1"


@app.post("/api/llm-test")
def test_llm_connection(req: LlmTestRequest):
    """Fast key/endpoint check (GET /v1/models, ~15s max). Key is never stored."""
    settings = LLMSettings(
        LLM_PROVIDER=req.provider,
        LLM_API_KEY=req.api_key or "",
        LLM_MODEL=req.model,
        LLM_BASE_URL=req.base_url,
    )
    llm = LLMService(settings=settings)
    try:
        return llm.check_connection()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/")
def root_status():
    return {
        "status": "ok",
        "service": "prism-backend",
        "version": "1.0.0",
        "health": "/api/health",
        "docs": "/docs"
    }


@app.get("/health")
def health_alias(session: tuple[str, SessionState] = Depends(resolve_session)):
    return health_check(session)


@app.get("/api/health")
def health_check(session: tuple[str, SessionState] = Depends(resolve_session)):
    _, state = session
    # Server-side default LLM status (env key, if any). A per-request key sent
    # from the UI Settings modal additionally enables live mode for that call.
    default_llm = LLMService()
    return {
        "status": "ok",
        "provider": "nvidia-nim",
        "tables_loaded": len(state.datasets),
        "llm_live": default_llm.is_configured(),
        "llm_provider": default_llm.settings.provider,
        "llm_model": default_llm.settings.model,
    }


@app.get("/api/catalog")
def get_catalog(session: tuple[str, SessionState] = Depends(resolve_session)):
    """Returns metadata for all registered tables."""
    _, session_state = session
    tables = []
    for name, meta in session_state.metadata_cache.items():
        tables.append({
            "name": name,
            "row_count": meta.row_count,
            "column_count": meta.column_count,
            "memory_bytes": meta.memory_bytes,
            "columns": [c.model_dump() for c in meta.columns],
            "is_active": name == session_state.active_dataset_name,
        })
    return {
        "active_dataset": session_state.active_dataset_name,
        "tables": tables,
    }


@app.post("/api/session")
def create_session():
    """Creates a fresh isolated session and returns its id."""
    sid, _ = session_manager.new_session()
    return {"session_id": sid}


@app.delete("/api/session")
def delete_session(session: tuple[str, SessionState] = Depends(resolve_session)):
    """Deletes this session entirely (datasets + conversation history in SQLite)."""
    sid, _ = session
    session_manager.clear(sid)
    return {"deleted": sid}


@app.post("/api/load-samples")
def load_sample_datasets(session: tuple[str, SessionState] = Depends(resolve_session)):
    """Explicit opt-in demo helper: loads bundled sample CSVs into THIS session only.

    Samples are never auto-loaded (assignment flow starts empty; the user uploads
    CSVs). The frontend offers this behind a "Load sample data" button.
    """
    _, session_state = session
    sample_sales = os.path.join(BASE_DIR, "data", "samples", "sales_data.csv")
    sample_cust = os.path.join(BASE_DIR, "data", "samples", "customers.csv")
    for path, name in ((sample_sales, "sales_data"), (sample_cust, "customers")):
        if os.path.exists(path) and name not in session_state.datasets:
            with open(path, "rb") as fh:
                raw = fh.read()
            # source_bytes persisted so samples survive restarts like uploads.
            session_state.register_dataset(
                name, read_csv_bytes(raw, os.path.basename(path)),
                source_bytes=raw, filename=os.path.basename(path),
            )
    if "sales_data" in session_state.datasets:
        session_state.set_active_dataset("sales_data")
    return {"message": "Sample datasets loaded successfully", "active": session_state.active_dataset_name}


@app.post("/api/select-dataset")
def select_dataset(
    req: DatasetSelectRequest,
    session: tuple[str, SessionState] = Depends(resolve_session),
):
    """Sets active dataset in session state."""
    _, session_state = session
    try:
        session_state.set_active_dataset(req.dataset_name)
        return {"status": "success", "active_dataset": session_state.active_dataset_name}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session: tuple[str, SessionState] = Depends(resolve_session),
):
    """Uploads one or more CSV files into DuckDB in-memory tables (per-session)."""
    _, session_state = session
    uploaded_tables = []
    for file in files:
        table_name = (file.filename or "dataset").rsplit(".", 1)[0]
        # sanitize
        clean_name = "".join(c if c.isalnum() else "_" for c in table_name).strip("_").lower() or "dataset"
        contents = await file.read()
        try:
            df = read_csv_bytes(contents, file.filename or "upload.csv")
            if df.empty or len(df.columns) == 0:
                raise ValueError("CSV contains no data columns.")
            # source_bytes persisted to SQLite: the file stays in this
            # conversation's context across server restarts.
            meta = session_state.register_dataset(
                clean_name, df, source_bytes=contents, filename=file.filename or "upload.csv"
            )
            uploaded_tables.append({
                "table_name": clean_name,
                "rows": meta.row_count,
                "columns": meta.column_count,
            })
        except HTTPException:
            raise
        except Exception as err:
            logger.error(f"Failed to read CSV {file.filename}: {err}")
            raise HTTPException(status_code=400, detail=f"Invalid CSV format for {file.filename}: {str(err)}")

    return {"uploaded": uploaded_tables, "active_dataset": session_state.active_dataset_name}


@app.post("/api/chat", response_model=AgentResponse)
def chat_with_agent(
    req: ChatRequest,
    session: tuple[str, SessionState] = Depends(resolve_session),
):
    """Processes a natural language query through the DataAnalystAgent lifecycle."""
    _, session_state = session
    settings = LLMSettings(
        LLM_PROVIDER=req.provider,
        LLM_API_KEY=req.api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY", ""),
        LLM_MODEL=req.model,
        LLM_BASE_URL=req.base_url,
    )
    llm = LLMService(settings=settings)
    agent = DataAnalystAgent(session_state=session_state, llm_service=llm)

    response = agent.run(req.query)
    return response


@app.post("/api/chat-stream")
def chat_stream(
    req: ChatRequest,
    session: tuple[str, SessionState] = Depends(resolve_session),
):
    """ChatGPT-style streaming: SSE status/token events + a final result event."""
    _, session_state = session
    settings = LLMSettings(
        LLM_PROVIDER=req.provider,
        LLM_API_KEY=req.api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY", ""),
        LLM_MODEL=req.model,
        LLM_BASE_URL=req.base_url,
    )
    llm = LLMService(settings=settings)
    agent = DataAnalystAgent(session_state=session_state, llm_service=llm)

    def event_gen():
        try:
            for event in agent.run_stream(req.query):
                etype = event.get("type")
                if etype == "result":
                    payload = event["response"].model_dump(mode="json")
                    yield f"data: {jsonlib.dumps({'type': 'result', 'response': payload})}\n\n"
                elif etype == "token":
                    yield f"data: {jsonlib.dumps({'type': 'token', 'text': event.get('text', '')})}\n\n"
                elif etype in ("status", "step"):
                    yield f"data: {jsonlib.dumps({'type': etype, 'text': event.get('text', '')})}\n\n"
        except Exception as exc:
            logger.exception(f"Stream failed: {exc}")
            yield f"data: {jsonlib.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/api/dashboard")
def get_dashboard(
    table_name: Optional[str] = None,
    session: tuple[str, SessionState] = Depends(resolve_session),
):
    """Returns schema-aware KPI cards, charts, and data quality report for the active table."""
    _, session_state = session
    if table_name and table_name in session_state.datasets:
        session_state.set_active_dataset(table_name)

    active_df = session_state.get_active_df()
    if active_df is None:
        raise HTTPException(status_code=400, detail="No active dataset.")

    resolved_table = session_state.active_dataset_name or "active_dataset"
    return build_dashboard_data(active_df, resolved_table)


@app.get("/api/export-report", response_class=HTMLResponse)
def export_report(session: tuple[str, SessionState] = Depends(resolve_session)):
    """Generates the executive HTML summary report."""
    _, session_state = session
    return generate_executive_html_report(session_state)


@app.post("/api/clear-chat")
def clear_chat(session: tuple[str, SessionState] = Depends(resolve_session)):
    _, session_state = session
    session_state.clear_history()
    return {"message": "Chat history cleared"}


# Mirror all /api/... routes to /... so both /api/endpoint and /endpoint work seamlessly
for _route in list(app.routes):
    if hasattr(_route, "path") and _route.path.startswith("/api/"):
        _unprefixed = _route.path[4:]
        if _unprefixed and not any(r.path == _unprefixed for r in app.routes):
            app.add_api_route(
                _unprefixed,
                _route.endpoint,
                methods=_route.methods,
                response_model=getattr(_route, "response_model", None),
                dependencies=getattr(_route, "dependencies", None),
                name=f"{_route.name}_alias",
            )


# Mount React build assets if present
frontend_dist = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist"))
if not os.path.exists(frontend_dist):
    frontend_dist = os.path.abspath(os.path.join(BASE_DIR, "frontend", "dist"))

if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def serve_spa(full_path: str):
        target_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(target_path):
            return FileResponse(target_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
