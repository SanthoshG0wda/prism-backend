import os
import sys

# Ensure backend root directory is on sys.path
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

# Set writable SQLite path for serverless environments (e.g. Vercel /tmp)
os.environ.setdefault("SESSION_DB_PATH", "/tmp/sessions.db")

from server import app

# Export as ASGI application for Vercel Serverless
app = app
