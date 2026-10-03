# AI Data Analyst - Backend

High-performance analytical backend powered by FastAPI, DuckDB, Pandas, and NVIDIA NIM (`meta/muse-glimmer-30b`).

## Directory Structure
- `src/agent/`: 7-step DataAnalystAgent orchestrator & session state.
- `src/tools/`: Deterministic DuckDB SQL execution, IQR/Z-score anomaly detection, Plotly charts, profiling, and report export.
- `src/services/`: NVIDIA NIM LLM client with OpenAI API compatibility and heuristic fallback.
- `src/models/`: Pydantic schemas and typed data contracts.
- `src/utils/`: Structured logging and benchmark evaluation suites.
- `data/samples/`: Benchmark CSV datasets (`sales_data.csv`, `customers.csv`).
- `tests/`: 23 comprehensive unit and integration tests.
- `server.py`: FastAPI application serving REST endpoints and serving compiled React SPA.
- `run.py`: Production/unified launcher script.

## Setup & Running with uv

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/ -v

# Run backend development server
uv run uvicorn server:app --port 8000 --reload

# Or run unified app (builds & serves frontend)
uv run python run.py
```
