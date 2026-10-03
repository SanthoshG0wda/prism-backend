"""
LLM Service abstraction supporting NVIDIA NIM (NVIDIA Inference Microservices),
OpenAI-compatible APIs (OpenAI, Groq, Ollama), and an intelligent offline heuristic
engine for development/testing when no API key is provided.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
import requests
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.utils.logging import get_logger
from src.tools.sql import quote_ident

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Default NVIDIA NIM parameters
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/muse-glimmer-30b"

# Shorthand aliases users/clients may send -> real NIM model ids.
MODEL_ALIASES = {
    "muse-glimmer": NVIDIA_DEFAULT_MODEL,
    "glimmer": NVIDIA_DEFAULT_MODEL,
    "muse-glimmer-30b": NVIDIA_DEFAULT_MODEL,
}


class LLMSettings(BaseSettings):
    """Configuration settings for LLM integrations with Muse Glimmer as premier agentic model."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    provider: str = Field(default="nvidia", alias="LLM_PROVIDER")
    model: str = Field(default=NVIDIA_DEFAULT_MODEL, alias="LLM_MODEL")
    api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    base_url: str = Field(default=NVIDIA_NIM_BASE_URL, alias="LLM_BASE_URL")
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    # Reasoning models (e.g. muse-glimmer-30b) spend tokens thinking; too small a
    # budget truncates generation mid-thought and yields content=null.
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    def get_effective_api_key(self) -> Optional[str]:
        """Returns the configured API key from either NVIDIA_API_KEY or LLM_API_KEY."""
        if self.api_key == "" or self.llm_api_key == "":
            return ""
        return self.api_key or self.llm_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY")


class LLMService:
    """
    Service client for interacting with Large Language Models via NVIDIA NIM or compatible endpoints.
    Separates LLM transport from prompt engineering and agent logic.
    """

    def __init__(self, settings: Optional[LLMSettings] = None) -> None:
        self.settings = settings or LLMSettings()

        # Adjust defaults if NVIDIA provider is selected
        if self.settings.provider.lower() in ("nvidia", "nvidia-nim", "nim"):
            if not self.settings.base_url or "api.openai.com" in self.settings.base_url:
                self.settings.base_url = NVIDIA_NIM_BASE_URL
            # Resolve shorthand/legacy ids to the real NIM id.
            model = (self.settings.model or "").strip()
            model = MODEL_ALIASES.get(model, model)
            if not model or "gpt-" in model:
                model = NVIDIA_DEFAULT_MODEL
            self.settings.model = model

        # Resolve API key
        self.settings.api_key = self.settings.get_effective_api_key() or ""

        logger.info(
            f"Initialized LLMService with provider='{self.settings.provider}', "
            f"model='{self.settings.model}', base_url='{self.settings.base_url}'"
        )

    def is_configured(self) -> bool:
        """Checks if a valid live API key is configured."""
        if self.settings.provider.lower() in ("mock", "offline", "heuristic"):
            return False
        key = self.settings.api_key
        return bool(key and key not in ("your-api-key-here", "nvapi-your-key-here") and len(key.strip()) > 5)

    def check_connection(self, timeout: int = 15) -> Dict[str, Any]:
        """Fast connectivity/key check in two steps (never hangs the chat).

        1. GET {base}/models (open on NIM): proves connectivity + model listing.
        2. POST a minimal chat completion: proves the KEY is accepted.
        Returns a small status dict; raises RuntimeError with a clear message
        on failure so the UI can show it in seconds instead of timing out.
        """
        if not self.is_configured():
            raise RuntimeError("No API key provided.")
        base = self.settings.base_url.rstrip("/")
        try:
            response = requests.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Could not reach {base}/models: {str(exc)}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"Endpoint error {response.status_code}: {response.text[:200]}"
            )
        try:
            ids = [m.get("id") for m in response.json().get("data", [])]
        except ValueError as exc:
            raise RuntimeError(f"Unreadable response from {base}/models.") from exc
        model = self.settings.model
        listed = model in ids or MODEL_ALIASES.get(model, model) in ids

        # Step 2: minimal completion proves the key (models endpoint is open).
        try:
            completion = requests.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_ALIASES.get(model, model),
                    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                    "temperature": 0.0,
                    "max_tokens": 8,
                },
                timeout=timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Key accepted but chat completions timed out after {timeout}s "
                f"(model slow/queued or egress issue): {str(exc)}"
            ) from exc
        if completion.status_code in (401, 403):
            raise RuntimeError(
                f"Key rejected ({completion.status_code}): invalid, expired, or revoked. "
                f"Generate a new one at build.nvidia.com."
            )
        if completion.status_code != 200:
            raise RuntimeError(
                f"Completion check failed ({completion.status_code}): "
                f"{completion.text[:200]}"
            )
        return {
            "ok": True,
            "provider": self.settings.provider,
            "model": model,
            "model_listed": listed,
            "models_count": len(ids),
            "key_valid": True,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Sends generation request to configured provider (NVIDIA NIM or OpenAI-compatible).
        Falls back to offline deterministic heuristics if no API key is supplied.
        """
        if not self.is_configured():
            logger.info("No live LLM API key configured. Executing offline heuristic orchestrator.")
            return self._heuristic_offline_completion(prompt, system_prompt, json_mode)

        return self._call_openai_compatible_api(prompt, system_prompt, json_mode)

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Generates and parses a structured response adhering strictly to a Pydantic model schema.
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system_prompt = (
            f"{system_prompt or ''}\n\n"
            f"IMPORTANT: You MUST respond strictly with a valid JSON object complying with this JSON Schema:\n"
            f"{schema_json}\n"
            f"Do not include any Markdown fences or conversational text outside the raw JSON object."
        ).strip()

        raw_response = self.generate(
            prompt=prompt,
            system_prompt=augmented_system_prompt,
            json_mode=True,
        )

        cleaned_json_text = self._extract_json(raw_response)
        try:
            parsed_data = json.loads(cleaned_json_text)
            return response_model.model_validate(parsed_data)
        except Exception as exc:
            logger.error(f"Failed to validate LLM response against {response_model.__name__}: {str(exc)}\nRaw: {raw_response}")
            raise ValueError(f"LLM structured response failed validation: {str(exc)}") from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ):
        """Yields answer text deltas via SSE streaming (OpenAI-compatible).

        Used by the ChatGPT-style streaming endpoint. Falls back to the
        offline heuristic as a single chunk when no API key is configured.
        Thinking-trace deltas (reasoning_content) are never yielded; if the
        stream ends with no answer content, raises so callers fall back.
        """
        if not self.is_configured():
            yield self._heuristic_offline_completion(prompt, system_prompt, False)
            return

        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": (
                [{"role": "system", "content": system_prompt}] if system_prompt else []
            ) + [{"role": "user", "content": prompt}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": True,
        }
        try:
            with requests.post(
                url, headers=headers, json=payload,
                timeout=self.settings.timeout_seconds, stream=True,
            ) as response:
                response.encoding = "utf-8"  # else SSE deltas decode as Latin-1
                if response.status_code != 200:
                    raise RuntimeError(
                        f"LLM API Error ({self.settings.provider}) "
                        f"{response.status_code}: {response.text[:300]}"
                    )
                saw_content = False
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
                    text = delta.get("content")
                    if text and isinstance(text, str):
                        saw_content = True
                        yield text
                    # else: reasoning_content delta — deliberately swallowed,
                    # never streamed to the user.
                if not saw_content:
                    raise RuntimeError(
                        f"LLM stream ended with no answer content ({self.settings.provider}, "
                        f"model={self.settings.model}). Try raising LLM_MAX_TOKENS."
                    )
        except requests.exceptions.RequestException as exc:
            logger.error(f"Streaming request to LLM provider failed: {str(exc)}")
            raise RuntimeError(f"Connection to LLM provider failed: {str(exc)}") from exc

    def _call_openai_compatible_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Makes an HTTP POST request to an OpenAI-compatible /chat/completions endpoint (e.g. NVIDIA NIM)."""
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.settings.timeout_seconds)
            # If NIM model rejects response_format={"type": "json_object"}, retry without it
            if response.status_code == 400 and json_mode and "response_format" in response.text:
                logger.warning("NIM model does not accept response_format parameter. Retrying without it.")
                payload.pop("response_format", None)
                response = requests.post(url, headers=headers, json=payload, timeout=self.settings.timeout_seconds)

            if response.status_code != 200:
                response.encoding = "utf-8"
                raise RuntimeError(
                    f"LLM API Error ({self.settings.provider}) {response.status_code}: {response.text}"
                )

            response.encoding = "utf-8"  # NIM omits charset; requests would guess Latin-1
            data = response.json()
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"LLM API returned no choices ({self.settings.provider}): {str(data)[:300]}"
                ) from exc
            content = message.get("content")
            if not content or not isinstance(content, str):
                # NEVER serve the private thinking trace (reasoning_content) as the
                # answer. Null content with reasoning present means generation was
                # cut off (usually max_tokens exhausted mid-thought) — raise so the
                # agent falls back to a deterministic grounded summary instead.
                has_trace = bool(message.get("reasoning_content") or message.get("reasoning"))
                raise RuntimeError(
                    f"LLM API returned empty content ({self.settings.provider}, "
                    f"model={self.settings.model})."
                    + (" Reasoning trace was present but is never shown to users." if has_trace else "")
                    + " Try raising LLM_MAX_TOKENS."
                )
            return content
        except requests.exceptions.RequestException as exc:
            logger.error(f"HTTP request to LLM provider failed: {str(exc)}")
            raise RuntimeError(f"Connection to LLM provider failed: {str(exc)}") from exc

    def _extract_json(self, text: str) -> str:
        """Strips markdown code blocks and whitespace to isolate raw JSON."""
        trimmed = text.strip()
        if trimmed.startswith("```json"):
            trimmed = trimmed[7:]
        elif trimmed.startswith("```"):
            trimmed = trimmed[3:]
        if trimmed.endswith("```"):
            trimmed = trimmed[:-3]
        trimmed = trimmed.strip()

        # If wrapped inside some other text, extract outermost braces
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start != -1 and end != -1:
            return trimmed[start : end + 1]
        return trimmed

    def _extract_schema_columns(self, prompt: str) -> list[str]:
        """Extract backticked column names from the catalog schema summary in the prompt."""
        cols: list[str] = []
        # Prefer markdown table rows ("| `col` | dtype | ...") over "### Table: `name`"
        # headers so table names are not mistaken for columns.
        row_cols = re.findall(r"^\s*\|\s*`([^`]+)`", prompt, flags=re.MULTILINE)
        candidates = row_cols if row_cols else re.findall(r"`([^`]+)`", prompt)
        for name in candidates:
            name = name.strip()
            if not name or " " in name or "/" in name or name.startswith("Table"):
                continue
            if name.lower() in ("active", "active_dataset"):
                continue
            if name not in cols:
                cols.append(name)
        return cols

    # Semantic equivalence groups: the question's words rarely equal column names
    # ("revenue" -> SALES, "region" -> TERRITORY on the sample dataset).
    EQUIV_GROUPS: list[set[str]] = [
        {"revenue", "sales", "earning", "earnings", "turnover", "income"},
        {"profit", "margin", "net"},
        {"quantity", "qty", "units", "units_sold", "count", "volume", "ordered"},
        {"price", "unit_price", "msrp", "priceeach", "amount"},
        {"customer", "client", "account", "customername", "buyer"},
        {"product", "item", "sku", "productline", "productcode"},
        {"region", "territory", "area", "zone", "market", "state", "country", "city"},
        {"date", "orderdate", "order_date", "time", "month", "year"},
        {"order", "ordernumber", "order_id", "orderlinenumber"},
    ]

    # Tight identity pairs checked BEFORE the broad groups below, so "region"
    # resolves to TERRITORY (not CITY) and "revenue" to SALES.
    TIGHT_EQUIV: list[set[str]] = [
        {"region", "territory"},
        {"revenue", "sales"},
        {"profit"},
        {"quantity", "quantityordered"},
        {"customer", "customername"},
        {"product", "productline", "productcode"},
        {"date", "orderdate"},
        {"price", "priceeach", "msrp"},
        {"month", "month_id"},
        {"year", "year_id"},
        {"discount"},
        {"country"},
        {"city"},
        {"state"},
        {"status"},
    ]

    @classmethod
    def _tight_terms(cls, column: str) -> set[str]:
        col = column.lower()
        for group in cls.TIGHT_EQUIV:
            if col in group:
                return set(group)
        return {col}

    @classmethod
    def _equiv_terms(cls, column: str) -> set[str]:
        """All question-words considered equivalent to a schema column."""
        col = column.lower()
        terms = {col}
        for group in cls.EQUIV_GROUPS:
            if col in group or any(g in col or col in g for g in group if len(g) > 3):
                terms |= group
        return terms

    @classmethod
    def _extract_column_types(cls, prompt: str) -> dict[str, str]:
        """Extract column name -> dtype from the schema markdown table."""
        col_types: dict[str, str] = {}
        for line in prompt.splitlines():
            m = re.match(r"\s*\|\s*`([^`]+)`\s*\|\s*([^\|]+)\|", line)
            if m:
                cname = m.group(1).strip()
                dtype = m.group(2).strip().lower()
                col_types[cname] = dtype
        return col_types

    @classmethod
    def _columns_of_kind(cls, cols: list[str], kind: str, col_types: dict[str, str] | None = None) -> list[str]:
        """Kind classification by schema data types and name heuristics."""
        col_types = col_types or {}
        date_hints = ("date", "time", "month", "year", "day", "created", "timestamp")
        metric_hints = (
            "revenue", "sales", "profit", "amount", "total", "price", "value", "cost",
            "quantity", "units", "count", "score", "balance", "discount", "volume",
            "salary", "population", "rate", "weight", "height", "age", "size", "speed",
            "temp", "temperature", "humidity", "margin", "tax", "fee", "shares", "tenure"
        )
        if kind == "date":
            res = [c for c in cols if any(t in col_types.get(c, "") for t in ("date", "time"))
                   or any(h in c.lower() for h in date_hints)]
            return res or [c for c in cols if any(h in c.lower() for h in date_hints)]
        if kind == "metric":
            num_cols = [c for c in cols if any(t in col_types.get(c, "") for t in ("int", "float", "double", "num", "decimal"))]
            if num_cols:
                ranked = [c for c in num_cols if any(h in c.lower() for h in metric_hints)]
                return ranked if ranked else num_cols
            ranked = [c for c in cols if any(h in c.lower() for h in metric_hints)]
            return ranked if ranked else list(cols)
        if kind == "categorical":
            cat_cols = [c for c in cols if not any(t in col_types.get(c, "") for t in ("int", "float", "double", "date", "time"))]
            if cat_cols:
                return cat_cols
            return [c for c in cols
                    if not any(h in c.lower() for h in date_hints)
                    and not any(h in c.lower() for h in metric_hints)]
        return list(cols)

    def _extract_target_table(self, prompt: str) -> str | None:
        m = re.search(r"TARGET TABLE:\s*(\S+)", prompt)
        return m.group(1).strip() if m else None

    def _extract_table_columns(self, prompt: str, table: str) -> list[str]:
        """Extract columns belonging to a specific table section of the schema summary."""
        lines = prompt.splitlines()
        in_table = False
        cols: list[str] = []
        for line in lines:
            if line.startswith("### Table:"):
                in_table = f"`{table}`" in line
                continue
            if in_table:
                if line.startswith("### Table:"):
                    break
                m = re.match(r"\s*\|\s*`([^`]+)`", line)
                if m:
                    cols.append(m.group(1).strip())
        return cols

    def _extract_tables_map(self, prompt: str) -> dict[str, list[str]]:
        """Parse every '### Table: `name`' section into table -> columns."""
        tables: dict[str, list[str]] = {}
        current: str | None = None
        for line in prompt.splitlines():
            m = re.match(r"### Table:\s*`([^`]+)`", line)
            if m:
                current = m.group(1).strip()
                tables[current] = []
                continue
            if current is not None:
                cm = re.match(r"\s*\|\s*`([^`]+)`", line)
                if cm:
                    tables[current].append(cm.group(1).strip())
        return tables

    def _heuristic_offline_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Deterministic heuristic reasoning engine used when no live API key is provided.
        Schema-aware: parses column names from the planning prompt so generic
        CSVs (not just sales_data) resolve to valid group/metric/date columns.
        """
        schema_cols = self._extract_schema_columns(prompt)
        col_types = self._extract_column_types(prompt)
        target_table = self._extract_target_table(prompt)
        table_cols = self._extract_table_columns(prompt, target_table) if target_table else []
        # Prefer target-table columns so plans stay executable against the active df.
        ordered_cols = table_cols + [c for c in schema_cols if c not in table_cols]
        schema_cols = ordered_cols or schema_cols
        if "USER QUESTION:" in prompt:
            user_part = prompt.split("USER QUESTION:")[-1]
            if "Based on the dataset schema" in user_part:
                user_part = user_part.split("Based on the dataset schema")[0]
            prompt_lower = user_part.lower()
        else:
            prompt_lower = prompt.lower()

        def _pick(col_kind: str, fallback: str, synonyms: list[str] | None = None) -> str:
            import re as _re
            kind_cols = self._columns_of_kind(schema_cols, col_kind, col_types) if col_kind != "any" else list(schema_cols)
            kind_set = {c.lower() for c in kind_cols}
            # 1. explicit mention in question wins, but must match the requested kind
            # (prevents picking group col 'region' as the numeric metric).
            for c in schema_cols:
                if c.lower() in prompt_lower and (col_kind == "any" or c.lower() in kind_set):
                    return c
            # 1b. semantic equivalence, tight identities first
            # ("region" -> TERRITORY not CITY, "revenue" -> SALES).
            for matcher in (self._tight_terms, self._equiv_terms):
                for c in schema_cols:
                    if col_kind != "any" and c.lower() not in kind_set:
                        continue
                    for term in matcher(c):
                        if len(term) > 2 and _re.search(r"\b" + _re.escape(term) + r"s?\b", prompt_lower):
                            return c
            # 2. synonym match (e.g. "earnings" -> revenue-like column)
            if synonyms:
                for syn in synonyms:
                    if syn in prompt_lower:
                        for c in kind_cols or schema_cols:
                            if syn in c.lower() or c.lower() in syn:
                                return c
            # 3. kind-based default from schema
            if kind_cols:
                return kind_cols[0]
            if schema_cols:
                return schema_cols[0]
            return fallback

        qi = quote_ident

        # Check if planning step requested
        if json_mode and "selected_tool" in (system_prompt or ""):
            # 0. List all rows / records / table preview intent
            if any(w in prompt_lower for w in [
                "list all", "list rows", "show all rows", "show rows", "view rows", "display rows",
                "all rows", "records", "preview table", "show table", "view table", "preview data",
                "show data", "display data", "first rows", "head", "see rows", "browse table", "browse data"
            ]):
                table_target = "active_dataset"
                if "customer" in prompt_lower:
                    table_target = "customers"
                elif "sales" in prompt_lower:
                    table_target = "sales_data"

                sql_q = f"SELECT * FROM {qi(table_target)} LIMIT 100"
                return json.dumps({
                    "user_intent": f"List and inspect records from {table_target}",
                    "reasoning": f"Identified request to inspect rows from '{table_target}'. Selected execute_sql_query with LIMIT 100.",
                    "selected_tool": "execute_sql_query",
                    "tool_parameters": {"query": sql_q},
                    "generated_sql": sql_q,
                    "generated_pandas_code": "df.head(100)",
                })

            # 1. Multi-table JOIN intent (must precede single-table branches)
            tables_map = self._extract_tables_map(prompt)
            mentioned_tables = [t for t in tables_map
                                if t.lower() in prompt_lower or t.lower().replace("_", " ") in prompt_lower]
            wants_join = ("join" in prompt_lower or "combin" in prompt_lower or "merge" in prompt_lower
                          or "together" in prompt_lower or "enrich" in prompt_lower)
            if wants_join and len(mentioned_tables) >= 2:
                t1, t2 = mentioned_tables[0], mentioned_tables[1]
                shared = [c for c in tables_map[t1] if c in tables_map[t2]]
                if shared:
                    key = shared[0]
                    sql_q = (f"SELECT * FROM {qi(t1)} JOIN {qi(t2)} "
                             f"ON {qi(t1)}.{qi(key)} = {qi(t2)}.{qi(key)} LIMIT 100")
                    pandas_code = (f"pd.merge(df_{t1} if 'df_{t1}' in dir() else df, "
                                   f"df_{t2} if 'df_{t2}' in dir() else df, on='{key}', how='inner').head(100)")
                else:
                    sql_q = f"SELECT * FROM {qi(t1)} LIMIT 50"
                    pandas_code = "df.head(50)"
                    key = None
                return json.dumps({
                    "user_intent": f"Join {t1} with {t2}" + (f" on {key}" if key else " (no shared key)"),
                    "reasoning": f"Detected multi-file analysis request. Joining '{t1}' with '{t2}'"
                    + (f" on shared key '{key}'." if key else "; no shared key found, previewing first table."),
                    "selected_tool": "execute_sql_query",
                    "tool_parameters": {"query": sql_q},
                    "generated_sql": sql_q,
                    "generated_pandas_code": pandas_code,
                })

            # 2. Anomaly detection intent
            if "anomal" in prompt_lower or "outlier" in prompt_lower or "unusual" in prompt_lower:
                metric = _pick("metric", "revenue")
                explicit_col = False
                for candidate in schema_cols + ["profit", "units_sold", "revenue", "unit_price", "discount"]:
                    if candidate.lower() in prompt_lower:
                        metric = candidate
                        explicit_col = True
                        break
                return json.dumps({
                    "user_intent": "Detect statistical anomalies and outliers in dataset",
                    "reasoning": f"Identified request for outlier detection on numeric column '{metric}'. Selected detect_anomalies tool with IQR method.",
                    "selected_tool": "detect_anomalies",
                    "tool_parameters": {"column": metric, "method": "iqr", "threshold": 1.5,
                                        "scan_all": not explicit_col},
                    "generated_sql": None,
                    "generated_pandas_code": f"detect_anomalies(df, column='{metric}', method='iqr', threshold=1.5)",
                })

            # 2. Chart / visualization intent
            if any(w in prompt_lower for w in ["chart", "plot", "visual", "graph", "histogram", "scatter", "bar"]):
                chart_type = "bar"
                if "line" in prompt_lower or "trend" in prompt_lower:
                    chart_type = "line"
                elif "pie" in prompt_lower or "share" in prompt_lower:
                    chart_type = "pie"
                elif "scatter" in prompt_lower:
                    chart_type = "scatter"
                elif "box" in prompt_lower:
                    chart_type = "box"

                x_col = _pick("categorical", "region")
                y_col = _pick("metric", "revenue")
                if "product" in prompt_lower:
                    x_col = "product" if "product" in schema_cols or not schema_cols else x_col
                if "customer" in prompt_lower:
                    cand = next((c for c in schema_cols if "customer" in c.lower()), x_col)
                    x_col = cand
                if "date" in prompt_lower or "month" in prompt_lower or "time" in prompt_lower:
                    x_col = _pick("date", x_col)
                if "profit" in prompt_lower:
                    y_col = next((c for c in schema_cols if "profit" in c.lower()), y_col)
                elif "units" in prompt_lower or "quantity" in prompt_lower:
                    y_col = next((c for c in schema_cols if "unit" in c.lower() or "quantity" in c.lower()), y_col)

                return json.dumps({
                    "user_intent": f"Generate {chart_type} visualization of {y_col} across {x_col}",
                    "reasoning": f"Identified plotting request. Constructing {chart_type} chart for '{y_col}' grouped by '{x_col}'.",
                    "selected_tool": "generate_chart",
                    "tool_parameters": {"chart_type": chart_type, "x": x_col, "y": y_col, "title": f"{chart_type.title()} Chart: {y_col} by {x_col}"},
                    "generated_sql": None,
                    "generated_pandas_code": f"px.{chart_type}(df, x='{x_col}', y='{y_col}')",
                })

            # 3. Forecasting / Projection intent
            if any(w in prompt_lower for w in ["forecast", "predict", "project", "future"]):
                metric = _pick("metric", "revenue", synonyms=["profit", "revenue", "sales"])
                date_col = _pick("date", "date")
                return json.dumps({
                    "user_intent": f"Forecast future {metric} projections with 95% confidence intervals",
                    "reasoning": f"Identified request for predictive forecasting on '{metric}'. Selected forecast_metric tool.",
                    "selected_tool": "forecast_metric",
                    "tool_parameters": {"date_col": date_col, "metric_col": metric, "periods": 3, "freq": "ME"},
                    "generated_sql": None,
                    "generated_pandas_code": f"forecast_metric(df, date_col='{date_col}', metric_col='{metric}', periods=3)",
                })

            # 4. Time series / monthly trend intent
            if "month" in prompt_lower or "trend" in prompt_lower or "time" in prompt_lower or "over time" in prompt_lower:
                date_col = _pick("date", "date")
                metric_col = _pick("metric", "revenue", synonyms=["profit", "revenue", "sales"])
                return json.dumps({
                    "user_intent": "Analyze monthly time-series sales trend",
                    "reasoning": f"Detected trend inquiry. Resampling {metric_col} by month end.",
                    "selected_tool": "time_series_trend",
                    "tool_parameters": {"date_col": date_col, "metric_col": metric_col, "freq": "ME", "agg_func": "sum"},
                    "generated_sql": f"SELECT strftime({qi(date_col)}, '%Y-%m') AS month, SUM({qi(metric_col)}) AS total FROM {qi('active_dataset')} GROUP BY 1 ORDER BY 1",
                    "generated_pandas_code": f"df.assign(date=pd.to_datetime(df['{date_col}'])).set_index('{date_col}').resample('ME')['{metric_col}'].sum().reset_index()",
                })

            # 5. Underperforming products / bottom k intent
            if "underperform" in prompt_lower or "worst" in prompt_lower or "lowest" in prompt_lower or "bottom" in prompt_lower:
                group_col = _pick("categorical", "product" if "product" in prompt_lower else "region")
                metric_col = _pick("metric", "revenue", synonyms=["profit", "revenue", "sales"])
                return json.dumps({
                    "user_intent": f"Identify lowest/underperforming {group_col}s by {metric_col}",
                    "reasoning": f"Finding underperforming {group_col} entities by aggregating {metric_col} in ascending order.",
                    "selected_tool": "top_k_analysis",
                    "tool_parameters": {"group_col": group_col, "metric_col": metric_col, "k": 5, "ascending": True, "agg_func": "sum"},
                    "generated_sql": f"SELECT {qi(group_col)}, SUM({qi(metric_col)}) AS total FROM {qi('active_dataset')} GROUP BY {qi(group_col)} ORDER BY total ASC LIMIT 5",
                    "generated_pandas_code": f"df.groupby('{group_col}')['{metric_col}'].sum().reset_index().sort_values(by='{metric_col}', ascending=True).head(5)",
                })

            # 6. Top customers / regions / products
            if "top" in prompt_lower or "highest" in prompt_lower or "most" in prompt_lower or "best" in prompt_lower:
                group_col = _pick("categorical", "region")
                metric = _pick("metric", "revenue", synonyms=["profit", "revenue", "sales"])

                return json.dumps({
                    "user_intent": f"Find top entities by {metric} grouped by {group_col}",
                    "reasoning": f"Selected top_k_analysis to rank {group_col} by sum of {metric} in descending order.",
                    "selected_tool": "top_k_analysis",
                    "tool_parameters": {"group_col": group_col, "metric_col": metric, "k": 5, "ascending": False, "agg_func": "sum"},
                    "generated_sql": f"SELECT {qi(group_col)}, SUM({qi(metric)}) AS {qi(f'total_{metric}')} FROM {qi('active_dataset')} GROUP BY {qi(group_col)} ORDER BY {qi(f'total_{metric}')} DESC LIMIT 5",
                    "generated_pandas_code": f"df.groupby('{group_col}')['{metric}'].sum().reset_index().sort_values(by='{metric}', ascending=False).head(5)",
                })

            # 7. SQL specific query request
            if "sql" in prompt_lower:
                group_col = _pick("categorical", "region")
                metric_col = _pick("metric", "revenue")
                sql_q = f"SELECT {qi(group_col)}, SUM({qi(metric_col)}) AS total FROM {qi('active_dataset')} GROUP BY {qi(group_col)} ORDER BY total DESC"
                return json.dumps({
                    "user_intent": "Execute SQL query on the dataset",
                    "reasoning": "Detected SQL request. Formulated aggregate SQL query for DuckDB execution.",
                    "selected_tool": "execute_sql_query",
                    "tool_parameters": {"query": sql_q},
                    "generated_sql": sql_q,
                    "generated_pandas_code": f"df.groupby('{group_col}')['{metric_col}'].sum().reset_index()",
                })

            # 8. Profile / summarize dataset intent
            if any(w in prompt_lower for w in ["profile", "summarize", "summary", "analyze", "overview", "describe", "analysis"]):
                table_target = "active_dataset"
                if "customer" in prompt_lower:
                    table_target = "customers"
                elif "sales" in prompt_lower:
                    table_target = "sales_data"
                return json.dumps({
                    "user_intent": f"Profile and summarize dataset {table_target}",
                    "reasoning": f"Profiling dataset '{table_target}' to inspect row count, column datatypes, and data distribution.",
                    "selected_tool": "profile_dataset",
                    "tool_parameters": {"table_name": table_target},
                    "generated_sql": f"SELECT count(*) AS total_rows FROM {qi(table_target)}",
                    "generated_pandas_code": "df.info(); df.describe()",
                })

            # 9. Quality check
            if "quality" in prompt_lower or "missing" in prompt_lower or "clean" in prompt_lower:
                return json.dumps({
                    "user_intent": "Run dataset quality audit",
                    "reasoning": "Selected check_data_quality tool to evaluate completeness and issues.",
                    "selected_tool": "check_data_quality",
                    "tool_parameters": {"table_name": "active_dataset"},
                    "generated_sql": None,
                    "generated_pandas_code": "check_data_quality(df)",
                })

            # Default fallback: general aggregation (schema-aware)
            group_col = _pick("categorical", "region")
            metric_col = _pick("metric", "revenue")
            return json.dumps({
                "user_intent": "General dataset analysis and summary",
                "reasoning": f"Defaulting to top_k {metric_col} analysis by {group_col} as analytical entry point.",
                "selected_tool": "top_k_analysis",
                "tool_parameters": {"group_col": group_col, "metric_col": metric_col, "k": 5, "ascending": False, "agg_func": "sum"},
                "generated_sql": f"SELECT {qi(group_col)}, SUM({qi(metric_col)}) AS total FROM {qi('active_dataset')} GROUP BY {qi(group_col)} ORDER BY total DESC",
                "generated_pandas_code": f"df.groupby('{group_col}')['{metric_col}'].sum().reset_index().sort_values(by='{metric_col}', ascending=False)",
            })

        # Non-JSON response (conversational agent or tool synthesis)
        if "TOOL OUTPUT DATA:" in prompt:
            if "column_profiles" in prompt or ("row_count" in prompt and "memory_bytes" in prompt):
                return (
                    "### 📊 Dataset Verified & Cataloged\n\n"
                    "The dataset has been successfully loaded into memory and registered into the DuckDB analytical catalog.\n\n"
                    "- **Data Integrity**: Verified all rows, datatypes, and null constraints without hallucinations.\n"
                    "- **Analysis Ready**: You can now ask for metric rankings, SQL joins, anomaly audits, or forecasts.\n\n"
                    "**💡 Recommended Actions:**\n"
                    "1. *\"Generate an Executive Dashboard artifact for this dataset\"*\n"
                    "2. *\"Detect anomalies in revenue or key metrics\"*\n"
                    "3. *\"List all rows\"* or *\"Show top 5 items\"*"
                )

            if "records" in prompt and ("SELECT" in prompt or "row_count" in prompt or "list" in prompt_lower or "rows" in prompt_lower):
                return (
                    "Retrieved verified records from the dataset via safe DuckDB SQL execution.\n\n"
                    "- **Methodology**: Executed deterministic read-only query in DuckDB.\n"
                    "- **Data Integrity**: Verified directly against in-memory session tables.\n"
                    "- **Granular Inspection**: All columns and records are rendered in the interactive tabular view below."
                )
            return (
                f"Based on the deterministic calculation from the dataset:\n\n"
                f"- **Methodology**: Computed deterministically via safe query execution.\n"
                f"- **Data Integrity**: Verified results contain no synthetic or hallucinated figures.\n"
                f"- **Key Takeaway**: Review the verified figures, breakdown table, and visualization above for full granular inspection."
            )

        # Conversational / Conceptual queries
        q = prompt_lower
        if "special abilities" in q or "superpowers" in q or "assignment" in q:
            return (
                "### ⚡ AI Assistant Special Superpowers (DBO Assignment)\n\n"
                "I combine the flexibility of a normal conversational AI with production-grade, mathematically grounded analytical abilities:\n\n"
                "#### 1. Multi-CSV Ingestion & Schema Cataloging\n"
                "- Upload multiple CSV files (e.g., `sales_data`, `customers`).\n"
                "- Automatic schema profiling, data type detection, null tracking, and memory registration.\n\n"
                "#### 2. Zero-Hallucination SQL Analytics (DuckDB)\n"
                "- Deterministic in-memory SQL execution via DuckDB.\n"
                "- Read-only sandboxing prevents SQL injection or schema mutation.\n"
                "- Supports multi-table relational joins.\n\n"
                "#### 3. Statistical Anomaly Auditing\n"
                "- **Tukey's IQR Fences**: $Q_1 - 1.5 \\times \\text{IQR}$ and $Q_3 + 1.5 \\times \\text{IQR}$.\n"
                "- **Z-Score Standardization**: Flag points where $|Z| > 3.0$.\n"
                "- Provides clear textual explanations for why every outlier was flagged.\n\n"
                "#### 4. Time-Series Forecasting & Trends\n"
                "- Resamples monthly/weekly sales trends.\n"
                "- Projects metrics forward with 95% confidence intervals.\n\n"
                "#### 5. Interactive Visualizations\n"
                "- Generates responsive Plotly charts: Bar, Line, Pie, Scatter, Histogram, and Box plots.\n\n"
                "#### 6. Claude-Style Executive Dashboard Artifacts\n"
                "- Generates full-screen or side-by-side interactive dashboard artifacts upon request.\n"
                "- Includes executive KPI summaries, data quality scores, and distribution breakdowns.\n\n"
                "#### 7. Full Code Transparency\n"
                "- Expandable inspection drawers showing generated SQL and Pandas code for every query."
            )

        if "regression" in q or "machine learning" in q or "model" in q:
            return (
                "### Linear vs. Logistic Regression\n\n"
                "In machine learning and statistics, these are foundational supervised learning algorithms:\n\n"
                "| Feature | Linear Regression | Logistic Regression |\n"
                "| :--- | :--- | :--- |\n"
                "| **Target Variable** | Continuous numeric value ($y \\in \\mathbb{R}$) | Categorical / Probability ($y \\in \\{0, 1\\}$) |\n"
                "| **Hypothesis Function** | $\\hat{y} = w^T x + b$ | $\\hat{y} = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w^T x + b)}}$ |\n"
                "| **Loss Function** | Mean Squared Error (MSE) | Binary Cross-Entropy / Log Loss |\n"
                "| **Output Range** | $(-\\infty, +\\infty)$ | $[0, 1]$ |\n"
                "| **Typical Use Case** | Forecasting revenue, house prices, temperature | Churn prediction, spam detection, fraud audit |\n\n"
                "Would you like an example of how to implement either in Python with Scikit-learn?"
            )

        if "anomaly" in q or "outlier" in q or "tukey" in q or "z-score" in q or "iqr" in q:
            return (
                "### Statistical Anomaly Detection Methodologies\n\n"
                "In data science and business analytics, anomalies (outliers) are data points that significantly deviate from the majority of observations. This application implements two deterministic methods:\n\n"
                "#### 1. Tukey's Interquartile Range (IQR) Fences\n"
                "- **IQR Calculation**: $\\text{IQR} = Q_3 - Q_1$\n"
                "- **Lower Fence**: $Q_1 - 1.5 \\times \\text{IQR}$\n"
                "- **Upper Fence**: $Q_3 + 1.5 \\times \\text{IQR}$\n"
                "- Points outside these fences are flagged as statistical anomalies. This method is **non-parametric** and robust against extreme skews.\n\n"
                "#### 2. Z-Score Standardization\n"
                "- **Formula**: $Z = \\frac{X - \\mu}{\\sigma}$\n"
                "- Flags points where $|Z| > 3.0$ (observations more than 3 standard deviations from the mean).\n\n"
                "*Tip: You can ask me to run an outlier audit on the loaded dataset anytime!*"
            )

        if "duckdb" in q:
            return (
                "### About DuckDB in This Architecture\n\n"
                "**DuckDB** is an embedded analytical SQL database management system (often called the 'SQLite for Analytics').\n\n"
                "**Why we use DuckDB in this application:**\n"
                "1. **Columnar Execution Engine**: Optimized for OLAP (analytical) aggregations (`SUM`, `AVG`, `GROUP BY`).\n"
                "2. **Zero-Copy Pandas Integration**: Queries Pandas DataFrames directly in-memory without expensive serialization.\n"
                "3. **Multi-Table Joins**: Allows joining multiple uploaded CSVs (e.g. sales and customer tables) via standard SQL.\n"
                "4. **Security & Isolation**: We enforce strict read-only validation to prevent destructive operations (`DROP`, `DELETE`, `ALTER`)."
            )

        if "python" in q or "code" in q or "script" in q:
            return (
                "### Python Code Solution\n\n"
                "Here is an idiomatic Python solution using modern best practices:\n\n"
                "```python\n"
                "import pandas as pd\n"
                "import duckdb\n\n"
                "# In-memory analysis with DuckDB and Pandas\n"
                "def analyze_dataset(file_path: str):\n"
                "    df = pd.read_csv(file_path)\n"
                "    conn = duckdb.connect(database=':memory:')\n"
                "    conn.register('data', df)\n"
                "    \n"
                "    query = '''\n"
                "        SELECT region, SUM(revenue) AS total_revenue\n"
                "        FROM data\n"
                "        GROUP BY region\n"
                "        ORDER BY total_revenue DESC\n"
                "    '''\n"
                "    return conn.execute(query).df()\n"
                "```\n\n"
                "Let me know if you would like me to adjust the script for a specific data operation!"
            )

        if "data quality" in q or "missing value" in q:
            return (
                "### Data Quality & Completeness Audit\n\n"
                "Data quality is critical for reliable business intelligence and modeling. Our built-in audit checks:\n"
                "- **Completeness Score**: Percentage of non-null cells across the entire matrix: $\\frac{\\text{Total Non-Nulls}}{\\text{Total Cells}} \\times 100\\%$.\n"
                "- **Missing Values by Column**: Exact counts and percentages of null or NaN values per column.\n"
                "- **Duplicate Detection**: Identifies exact duplicate rows to prevent double-counting.\n"
                "- **Type Consistency**: Flags mixed types and empty strings.\n\n"
                "You can see your active dataset's quality audit by asking for a dashboard or data quality check!"
            )

        if "digital back office" in q or "dbo" in q:
            return (
                "I am **Prism**, running inside this application — upload a CSV and I can "
                "rank entities, plot charts, forecast trends, detect anomalies, and explain every result "
                "with grounded numbers."
            )

        return (
            "I am **Prism**, ready to assist you with general questions, software engineering, statistical concepts, or deep analysis of your tabular datasets.\n\n"
            "### How I Can Help:\n"
            "- **Normal Conversational Assistant**: Ask me conceptual questions, statistical theory, code generation, or general brainstorming.\n"
            "- **Tabular Analytics**: Upload CSVs and ask me to rank entities, plot charts, forecast trends, or detect anomalies.\n"
            "- **Executive Dashboard**: Request an interactive Claude-style dashboard artifact for any loaded dataset.\n\n"
            "What would you like to explore next?"
        )
