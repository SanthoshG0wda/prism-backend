# Prism Backend (FastAPI + DuckDB + Pandas)

[![Monorepo](https://img.shields.io/badge/GitHub-Monorepo-181717?logo=github)](https://github.com/SanthoshG0wda/prism)
[![Frontend Repo](https://img.shields.io/badge/GitHub-Frontend_Repo-181717?logo=github)](https://github.com/SanthoshG0wda/prism-frontend)
[![Tests Passing](https://img.shields.io/badge/Tests-155%20Passing-success)](https://github.com/SanthoshG0wda/prism/tree/main/backend/tests)
[![Live API](https://img.shields.io/badge/Vercel-Live%20API-black?logo=vercel)](https://prism-backend-tau.vercel.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory-FFF000?logo=duckdb)](https://duckdb.org)

> 🌐 **Live Production API**: [https://prism-backend-tau.vercel.app](https://prism-backend-tau.vercel.app)  
> 📖 **Interactive API Documentation (Swagger)**: [https://prism-backend-tau.vercel.app/docs](https://prism-backend-tau.vercel.app/docs)  
> 🩺 **Health Check**: [https://prism-backend-tau.vercel.app/api/health](https://prism-backend-tau.vercel.app/api/health)  
> 📦 **Primary Submission Monorepo**: [https://github.com/SanthoshG0wda/prism](https://github.com/SanthoshG0wda/prism)  
> 🎯 **Assignment**: Digital Back Office Software Engineer Intern Assignment

---

## 📌 Overview

This is the standalone **FastAPI backend** for **Prism**, a conversational AI Data Analyst. It combines an agentic orchestrator with deterministic computing engines (`DuckDB`, `Pandas`, `SciPy`/numpy) to ensure **100% mathematical accuracy with zero numerical hallucinations**.

The service is fully decoupled and deployed independently to Vercel Serverless Functions.

---

## 🌟 Key Architecture & Capabilities

1. **7-Step Deterministic Agent Lifecycle**:
   - Parses natural-language questions and extracts analytical intent.
   - Inspects the active DuckDB table schema and column statistical profiles.
   - Formulates a structured `QueryPlan` (Pydantic schema).
   - Executes deterministic tools (`DuckDB`, `Pandas`, `Plotly`).
   - Receives verified computational outputs (enforces structured whole-record serialization).
   - Performs result sanity and containment validation.
   - Synthesizes transparent explanations with actionable business takeaways.

2. **Deterministic Analytical Tools**:
   - **DuckDB SQL Engine** (`src/tools/sql.py`): In-memory analytical queries with AST-level safety guards blocking destructive operations (`DROP`, `DELETE`, `ALTER`).
   - **Statistical Anomaly Detection** (`src/tools/anomalies.py`): Tukey's IQR fences ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$) and Z-scores ($|Z| > 3.0$) with bounds and explanations.
   - **Data Quality & Profiling** (`src/tools/profiling.py`): Completeness score, null rates, duplicates, and column distribution audits.
   - **Interactive Chart Spec Generator** (`src/tools/charts.py`): Generates dark-themed Plotly specifications.
   - **Executive Dashboard Builder** (`src/tools/dashboard.py`): Claude-style KPI cards and quality distributions.
   - **Predictive Forecasting** (`src/tools/forecasting.py`): Metric projections with 95% confidence intervals.

3. **Per-Conversation Session Isolation & SQLite Persistence**:
   - Manages state via `SessionManager` (`src/agent/sessions.py`) keyed by `X-Session-Id`.
   - Datasets and conversation histories are stored in a write-ahead-logged SQLite store (`data/sessions.db`), surviving restarts without leaking state between concurrent conversations.

4. **Slash Command Normalization**:
   - Natively normalizes commands like `/dashboard`, `/anomalies`, `/quality`, `/profile`, `/sql`, `/forecast`, `/chart`, and `/help` to their deterministic tool workflows.

5. **ChatGPT-Style SSE Streaming**:
   - `POST /api/chat-stream` streams status steps, reasoning tokens, and structured results in real time over Server-Sent Events (SSE).

6. **Offline Zero-Key Heuristic Mode**:
   - Operates reliably even without an external API key using deterministic heuristic planning.
   - Fully compatible with NVIDIA NIM (`meta/muse-glimmer-30b`), OpenAI, Groq, and Ollama.

---

## 📁 Repository Structure

```
├── api/
│   └── index.py            # Vercel Serverless Function entrypoint
├── data/
│   └── samples/            # Benchmark CSV datasets (sales_data.csv, customers.csv)
├── src/
│   ├── agent/              # Orchestrator agent, prompts, session manager, and state
│   ├── models/             # Typed Pydantic schemas and contracts
│   ├── services/           # LLM service client with OpenAI compatibility and fallback
│   ├── tools/              # Deterministic DuckDB, Pandas, and Plotly tools
│   └── utils/              # Structured logging, CSV parsers, and SQLite store
├── tests/                  # 155 automated unit and integration tests
├── pyproject.toml          # uv/pip dependencies and project metadata
├── requirements.txt        # Production deployment dependencies
├── server.py               # FastAPI application with routes and CORS
└── vercel.json             # Vercel serverless deployment configuration
```

---

## 🚀 Local Development Setup

### 1. Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 2. Installation
```bash
# Clone the standalone repository
git clone https://github.com/SanthoshG0wda/prism-backend.git
cd prism-backend

# Install dependencies using uv
uv sync

# Or using standard venv + pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` (optional for local development, as offline fallback mode works out of the box):
```bash
cp .env.example .env
```
Key variables:
- `NVIDIA_API_KEY`: API key for NVIDIA NIM (`meta/muse-glimmer-30b`).
- `LLM_MODEL`: Target model (defaults to `meta/muse-glimmer-30b`).
- `LLM_BASE_URL`: API base URL (defaults to `https://integrate.api.nvidia.com/v1`).

### 4. Running the Server
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
API endpoints will be accessible at `http://localhost:8000`, and interactive Swagger documentation at `http://localhost:8000/docs`.

### 5. Running the Test Suite
```bash
uv run pytest tests/ -v
```

---

## 🔗 Related Repositories
- **Primary Monorepo**: [https://github.com/SanthoshG0wda/prism](https://github.com/SanthoshG0wda/prism)
- **Standalone Frontend**: [https://github.com/SanthoshG0wda/prism-frontend](https://github.com/SanthoshG0wda/prism-frontend)
