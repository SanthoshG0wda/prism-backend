"""
SQLite-backed persistence for analyst sessions.

What is stored per session:
- datasets: original CSV bytes + filename (re-parsed on load, so profiling
  metadata and DuckDB tables are rebuilt deterministically)
- messages: full conversation history (role, content, metadata, timestamp)
- active dataset name

Sessions start empty; rows are only written via explicit user actions
(upload / chat / select-dataset / load-samples). Thread-safe; WAL mode.
"""

import datetime as _dt
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    active_dataset TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    session_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    content BLOB NOT NULL,
    uploaded_at TEXT NOT NULL,
    PRIMARY KEY (session_id, table_name)
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class SQLiteSessionStore:
    """Thread-safe SQLite persistence for sessions, datasets, and messages."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if not db_path and not os.getenv("SESSION_DB_PATH") and os.getenv("VERCEL"):
            resolved = "/tmp/sessions.db"
        else:
            resolved = (
                db_path
                or os.getenv("SESSION_DB_PATH")
                or os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "sessions.db"
                )
            )
        self.db_path = os.path.abspath(resolved)
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError:
            self.db_path = "/tmp/sessions.db"
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info(f"Session store ready at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
        # API keys live here: restrict file access when possible.
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    # -- sessions --------------------------------------------------------

    def touch_session(self, session_id: str, active_dataset: Optional[str] = None) -> None:
        now = _utcnow()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT active_dataset FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO sessions (session_id, active_dataset, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (session_id, active_dataset, now, now),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET active_dataset = COALESCE(?, active_dataset),"
                    " updated_at = ? WHERE session_id = ?",
                    (active_dataset, now, session_id),
                )

    def set_active(self, session_id: str, table_name: str) -> None:
        self.touch_session(session_id, active_dataset=table_name)

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM datasets WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # -- datasets --------------------------------------------------------

    def save_dataset(self, session_id: str, table_name: str, filename: str, content: bytes) -> None:
        self.touch_session(session_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO datasets"
                " (session_id, table_name, filename, content, uploaded_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, table_name, filename, content, _utcnow()),
            )
            conn.execute(
                "UPDATE sessions SET active_dataset = ?, updated_at = ? WHERE session_id = ?",
                (table_name, _utcnow(), session_id),
            )

    # -- messages --------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        self.touch_session(session_id)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            seq = (row[0] if row else -1) + 1
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, metadata_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    seq,
                    role,
                    content,
                    json.dumps(metadata or {}, default=str),
                    created_at or _utcnow(),
                ),
            )

    def clear_messages(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    # -- loading -----------------------------------------------------------

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Returns datasets/messages/active for rebuild, or None if unknown."""
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            sess = conn.execute(
                "SELECT active_dataset FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if sess is None:
                # No session row, but orphan dataset/message rows may exist; treat as unknown.
                return None
            datasets = conn.execute(
                "SELECT table_name, filename, content FROM datasets WHERE session_id = ?"
                " ORDER BY uploaded_at",
                (session_id,),
            ).fetchall()
            messages = conn.execute(
                "SELECT role, content, metadata_json, created_at FROM messages"
                " WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        return {
            "active_dataset": sess["active_dataset"],
            "datasets": [
                {"table_name": r["table_name"], "filename": r["filename"], "content": bytes(r["content"])}
                for r in datasets
            ],
            "messages": [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created_at": r["created_at"],
                }
                for r in messages
            ],
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT session_id, active_dataset, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- app settings (API keys etc.) --------------------------------------

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value, updated_at)"
                " VALUES (?, ?, ?)",
                (key, value, _utcnow()),
            )

    def get_setting(self, key: str) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def delete_setting(self, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM app_settings WHERE key = ?", (key,))


API_KEY_SETTING = "nvidia_api_key"


def key_hint(key: Optional[str]) -> Optional[str]:
    """Last-4 masked hint for display. The full key is never returned."""
    if not key or len(key) < 4:
        return None
    return f"•••{key[-4:]}"


def effective_api_key(explicit: Optional[str], store=None) -> str:
    """Resolve which API key to use. Precedence:
    explicit per-request key > environment > DB-stored key > "".
    """
    if explicit and explicit.strip():
        return explicit.strip()
    for var in ("NVIDIA_API_KEY", "LLM_API_KEY"):
        val = os.getenv(var)
        if val and val.strip():
            return val.strip()
    if store is not None:
        try:
            stored = store.get_setting(API_KEY_SETTING)
        except Exception:
            stored = None
        if stored and stored.strip():
            return stored.strip()
    return ""
