import os
import sys

# Ensure backend root directory is on sys.path
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

# Set writable SQLite path for serverless environments (e.g. Vercel /tmp)
os.environ.setdefault("SESSION_DB_PATH", "/tmp/sessions.db")

from server import app as fastapi_app

class VercelPathNormalizer:
    """Recovers real request path from Vercel edge headers (x-matched-path, x-forwarded-uri)."""
    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
            forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("utf-8", errors="ignore")
            orig = matched_path or forwarded_uri
            
            raw_path = scope.get("path", "")
            if raw_path in ("/api/index.py", "/api/index", "/api/index.py/", "/index.py"):
                if orig:
                    scope["path"] = orig.split("?")[0]
                else:
                    scope["path"] = "/api/health"
            elif orig and not raw_path.startswith("/api") and orig.startswith("/api"):
                scope["path"] = orig.split("?")[0]

            if "raw_path" in scope:
                scope["raw_path"] = scope["path"].encode("utf-8")

        await self.asgi_app(scope, receive, send)

app = VercelPathNormalizer(fastapi_app)
