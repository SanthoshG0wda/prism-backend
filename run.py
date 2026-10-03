"""
Unified launcher for the AI Data Analyst application.
Serves the FastAPI analytics backend and the pre-built React SPA at http://localhost:8000.
"""

import os
import subprocess
import sys
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
    if not os.path.exists(frontend_dir):
        frontend_dir = os.path.abspath(os.path.join(BASE_DIR, "frontend"))

    dist_path = os.path.join(frontend_dir, "dist")
    if not os.path.exists(dist_path) and os.path.exists(frontend_dir):
        print("⚡ Building React frontend...")
        subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)

    print("🚀 Starting AI Data Analyst on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

