"""
Per-client session management.

Replaces the previous single global SessionState singleton (unsafe for
concurrent users) with an isolated SessionState per session id.
Backwards compatible: callers without a session id share the "default" session.

Sessions always start EMPTY, per the assignment: users upload one or more
CSV files via POST /api/upload. Sample files in data/samples/ are never
auto-loaded; POST /api/load-samples exists only as an explicit opt-in demo helper.

Persistence: every bound session writes through to SQLite (datasets as original
CSV bytes, full conversation history, active table), so uploads and context
survive server restarts. On a memory miss the session is rebuilt from SQLite.
"""

import datetime
import threading
import uuid
from typing import Dict, Optional

from src.agent.state import SessionState
from src.models.schemas import ChatMessage
from src.utils.logging import get_logger
from src.utils.sqlite_store import SQLiteSessionStore

logger = get_logger(__name__)


class SessionManager:
    """Thread-safe registry of SessionState objects keyed by session id."""

    def __init__(self, store: Optional[SQLiteSessionStore] = None) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._store = store or SQLiteSessionStore()

    @property
    def store(self) -> SQLiteSessionStore:
        return self._store

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, SessionState]:
        sid = (session_id or "").strip() or "default"
        with self._lock:
            state = self._sessions.get(sid)
            if state is not None:
                return sid, state
            state = SessionState()
            state.bind(sid, self._store)
            self._restore_from_store(state, sid)
            self._sessions[sid] = state
            logger.info(f"Session '{sid}' ready ({len(state.datasets)} dataset(s) restored).")
            return sid, state

    def _restore_from_store(self, state: SessionState, sid: str) -> None:
        """Rebuild datasets + history + active table from SQLite (no re-persist)."""
        try:
            snapshot = self._store.load_session(sid)
        except Exception as exc:
            logger.warning(f"Session restore failed for '{sid}': {exc}")
            return
        if snapshot is None:
            return
        from src.utils.csv import read_csv_bytes

        for ds in snapshot.get("datasets", []):
            try:
                df = read_csv_bytes(ds["content"], ds.get("filename") or ds["table_name"])
                # source_bytes=None: already persisted, don't rewrite the blob.
                state.register_dataset(ds["table_name"], df)
            except Exception as exc:
                logger.warning(f"Restore of dataset '{ds.get('table_name')}' failed: {exc}")
        active = snapshot.get("active_dataset")
        if active and active in state.datasets:
            try:
                state.set_active_dataset(active)
            except KeyError:
                pass
        for m in snapshot.get("messages", []):
            try:
                ts = m.get("created_at")
                msg = ChatMessage(
                    role=m["role"],
                    content=m["content"],
                    metadata=m.get("metadata") or {},
                    **({"timestamp": datetime.datetime.fromisoformat(ts)} if ts else {}),
                )
                state.conversation_history.append(msg)
            except Exception as exc:
                logger.warning(f"Restore of message failed: {exc}")

    def new_session(self) -> tuple[str, SessionState]:
        return self.get_or_create(f"ses_{uuid.uuid4().hex[:12]}")

    def clear(self, session_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        try:
            self._store.delete_session(session_id)
        except Exception as exc:
            logger.warning(f"Session delete failed for '{session_id}': {exc}")
        return existed


session_manager = SessionManager()
