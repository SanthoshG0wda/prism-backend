"""
JSON sanitation for API responses.

Plotly figure dicts and pandas .to_dict() records routinely contain numpy
scalars/arrays, pandas Timestamps, and NaN/Infinity — all of which crash
FastAPI/Pydantic serialization (500s). Sanitize payloads at the boundary.
"""

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively convert a payload to strict JSON-serializable types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1")

    # numpy / pandas types (optional deps guarded by duck-typing to avoid imports)
    type_name = type(value).__name__
    module = type(value).__module__
    if module.startswith("numpy"):
        if type_name == "ndarray":
            return [json_safe(v) for v in value.tolist()]
        if hasattr(value, "item"):
            try:
                return json_safe(value.item())
            except (ValueError, TypeError):
                pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass

    return str(value)
