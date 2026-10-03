"""
Robust CSV ingestion with encoding fallback.

Real-world CSVs (including this repo's own sales_data_sample.csv) are often
Windows-encoded (cp1252/latin-1), not UTF-8. The default pd.read_csv assumes
UTF-8 and 400s on the first non-UTF-8 byte. This helper tries UTF-8 first
(BOM-aware), then cp1252, then latin-1 (which always decodes).
"""

import io
from typing import Tuple

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def read_csv_bytes(contents: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, falling back across encodings and delimiters.

    1. Falls back across encodings (utf-8-sig, utf-8, cp1252, latin-1).
    2. Automatically sniffs alternative delimiters (; \t |) if comma yields a single compound column.
    3. Strips whitespace from column names.
    """
    if not contents or not contents.strip():
        raise ValueError("CSV file is empty.")

    last_decode_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding=enc)
                # If parsed as a single column containing delimiters, sniff with python engine
                if len(df.columns) == 1 and any(d in str(df.columns[0]) for d in (';', '\t', '|')):
                    df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', encoding=enc)
            except Exception:
                df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', encoding=enc)

            df.columns = [str(c).strip() for c in df.columns]
            if enc not in ("utf-8-sig", "utf-8"):
                logger.info(f"Decoded '{filename}' with fallback encoding '{enc}'.")
            return df
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_decode_err = exc
            continue

    raise ValueError(
        f"Could not decode '{filename}' (tried {', '.join(ENCODINGS)}): {last_decode_err}"
    )
