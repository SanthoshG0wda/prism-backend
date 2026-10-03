"""
Session state and dataset catalog management.
Manages multi-file datasets, DuckDB in-memory registration, and conversation history.
"""

from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd
from src.models.schemas import ChatMessage, DatasetMetadata
from src.tools.profiling import profile_dataframe
from src.tools.sql import harden_duckdb_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SessionState:
    """
    Maintains session data, in-memory DuckDB tables, active tables, and conversation context.
    """

    def __init__(self) -> None:
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata_cache: Dict[str, DatasetMetadata] = {}
        self.active_dataset_name: Optional[str] = None
        self.conversation_history: List[ChatMessage] = []
        self.duckdb_conn: duckdb.DuckDBPyConnection = duckdb.connect(database=":memory:")
        # Engine hardening: no external file access, locked config, bounded
        # memory/threads (see src/tools/sql.py). register()/SELECT keep working.
        harden_duckdb_connection(self.duckdb_conn)
        # Optional SQLite persistence (bound by SessionManager; None in tests/tools).
        self._session_id: Optional[str] = None
        self._store = None
        logger.info("Initialized fresh SessionState with in-memory DuckDB connection.")

    def bind(self, session_id: str, store) -> None:
        """Attach SQLite persistence; subsequent mutations are written through."""
        self._session_id = session_id
        self._store = store

    def register_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        source_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
    ) -> DatasetMetadata:
        """
        Registers a new dataset, registers it in DuckDB, and caches its profiling metadata.
        When bound to a session store and source CSV bytes are provided, the bytes
        are persisted so the dataset survives restarts (re-parsed on load).
        """
        # Sanitize table name to be SQL-safe
        clean_name = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
        if not clean_name:
            clean_name = "dataset_1"

        self.datasets[clean_name] = df
        self.active_dataset_name = clean_name

        # Register in DuckDB
        self.duckdb_conn.register(clean_name, df)
        # Also maintain 'active_dataset' view pointing to active table
        self.duckdb_conn.register("active_dataset", df)

        metadata = profile_dataframe(df, clean_name)
        self.metadata_cache[clean_name] = metadata
        logger.info(f"Registered dataset '{clean_name}' ({len(df)} rows, {len(df.columns)} cols).")
        if self._store is not None and self._session_id and source_bytes:
            try:
                self._store.save_dataset(
                    self._session_id, clean_name, filename or f"{clean_name}.csv", source_bytes
                )
            except Exception as exc:
                logger.warning(f"Dataset persistence failed: {exc}")
        elif self._store is not None and self._session_id:
            # Keep the persisted active pointer in sync even for derived tables.
            try:
                self._store.set_active(self._session_id, clean_name)
            except Exception as exc:
                logger.warning(f"Session persistence failed: {exc}")
        return metadata

    def set_active_dataset(self, name: str) -> None:
        """Switches the active dataset view."""
        if name in self.datasets:
            self.active_dataset_name = name
            self.duckdb_conn.register("active_dataset", self.datasets[name])
            logger.info(f"Set active dataset to '{name}'.")
            if self._store is not None and self._session_id:
                try:
                    self._store.set_active(self._session_id, name)
                except Exception as exc:
                    logger.warning(f"Session persistence failed: {exc}")
        else:
            raise KeyError(f"Dataset '{name}' not found in active session catalog.")

    def get_active_df(self) -> Optional[pd.DataFrame]:
        """Returns the currently active pandas DataFrame, if any."""
        if self.active_dataset_name and self.active_dataset_name in self.datasets:
            return self.datasets[self.active_dataset_name]
        return None

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Appends a new message to conversation history (persisted when bound)."""
        msg = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.conversation_history.append(msg)
        if self._store is not None and self._session_id:
            try:
                self._store.save_message(
                    self._session_id, role, content, msg.metadata,
                    msg.timestamp.isoformat() if hasattr(msg.timestamp, "isoformat") else None,
                )
            except Exception as exc:
                logger.warning(f"Message persistence failed: {exc}")
        return msg

    def clear_history(self) -> None:
        """Clears chat history (in memory and in SQLite when bound)."""
        self.conversation_history.clear()
        if self._store is not None and self._session_id:
            try:
                self._store.clear_messages(self._session_id)
            except Exception as exc:
                logger.warning(f"Message persistence failed: {exc}")

    def get_catalog_schema_summary(self) -> str:
        """
        Produces a concise, structured markdown string summarizing all loaded tables,
        their column names, data types, and sample values for LLM grounding.
        """
        if not self.datasets:
            return "No datasets currently loaded in session."

        summary_parts = []
        for name, meta in self.metadata_cache.items():
            is_active = "(ACTIVE)" if name == self.active_dataset_name else ""
            part = [f"### Table: `{name}` {is_active} ({meta.row_count} rows, {meta.column_count} columns)"]
            part.append("| Column | Type | Null% | Distinct | Sample Values |")
            part.append("|---|---|---|---|---|")
            for col in meta.columns:
                samples_str = ", ".join(str(s) for s in col.sample_values[:3])
                part.append(f"| `{col.name}` | {col.dtype} | {col.null_percentage}% | {col.distinct_count} | {samples_str} |")
            summary_parts.append("\n".join(part))

        return "\n\n".join(summary_parts)
