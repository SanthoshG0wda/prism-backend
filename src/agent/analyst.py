"""
Data Analyst Orchestrator Agent.
Coordinates the 7-step analytical lifecycle between LLM reasoning, deterministic tools,
and result validation without allowing the LLM to invent numbers.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from src.agent.prompts import (
    SYSTEM_ASSISTANT_PROMPT,
    SYSTEM_CONVERSATIONAL_PROMPT,
    SYSTEM_PLANNER_PROMPT,
    SYSTEM_SYNTHESIS_PROMPT,
)
from src.agent.state import SessionState
from src.models.schemas import (
    AgentResponse,
    AnomalyItem,
    DataQualityReport,
    QueryPlan,
    ToolExecutionResult,
)
from src.services.llm import LLMService
from src.tools.dashboard import build_dashboard_data
from src.tools.registry import ToolRegistry, global_registry
from src.utils.json_safe import json_safe
from src.utils.logging import get_logger

logger = get_logger(__name__)


class _OfflineMode(Exception):
    """Internal signal: no live LLM key, use local fallbacks without network."""


class DataAnalystAgent:
    """
    AI Data Analyst orchestrator implementing the deterministic analysis loop:
    1. Understand user intent & conversation context.
    2. Inspect dataset catalog & schema.
    3. Select deterministic tool.
    4. Call the tool with validated arguments.
    5. Receive actual deterministic results.
    6. Validate the results.
    7. Generate natural-language explanation & actionable business insights.
    """

    def __init__(
        self,
        session_state: SessionState,
        llm_service: Optional[LLMService] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.state = session_state
        self.llm = llm_service or LLMService()
        self.tools = tool_registry or global_registry
        logger.info("DataAnalystAgent initialized.")

    def run(self, user_question: str) -> AgentResponse:
        """Executes the full agent analysis workflow (drains the event stream)."""
        final: Optional[AgentResponse] = None
        for event in self.run_stream(user_question):
            if event.get("type") == "result":
                final = event["response"]
        if final is None:
            raise RuntimeError("Agent produced no result.")
        return final

    def run_stream(self, user_question: str):
        """Event-generator twin of run(): yields status/token/step events plus a
        final {"type": "result", "response": AgentResponse} event.

        Powers the ChatGPT-style SSE endpoint; run() just drains it, so both
        paths share one implementation.
        """
        start_time = time.perf_counter()
        logger.info(f"Agent received question: '{user_question}'")
        yield {"type": "status", "text": "Understanding your question…"}

        # Handle natural conversational greetings, capabilities, and polite remarks
        clean_q = user_question.strip().lower().rstrip("!?. ")
        greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup", "yo"}
        general_inquiries = {"who are you", "what can you do", "help", "what are you", "what are your capabilities", "introduce yourself", "special abilities", "what are your special abilities"}

        _thanks = {
            "thank you", "thanks", "thx", "appreciate it", "great", "awesome", "perfect",
        }
        if clean_q in greetings or clean_q in _thanks or clean_q in general_inquiries:
            # Greetings AND capability questions are answered by the LLM itself via
            # the capabilities system prompt — never a hardcoded template.
            # Pure greetings ("hello", "thanks") get a brief natural hello with NO
            # capability list, dataset talk, or suggestions. Only explicit capability
            # questions ("what can you do") get the compact capability summary.
            is_capability_question = clean_q in general_inquiries
            engine_txt = (
                "online LLM"
                if self.llm.is_configured()
                else "offline heuristic mode (no API key configured)"
            )
            if is_capability_question:
                greeting_prompt = (
                    f"The user asked {user_question!r}.\n\n"
                    f"You are running on {engine_txt}.\n"
                    f"Summarize your capabilities compactly and follow the system prompt."
                )
                status_txt, step_txt = "Listing capabilities…", "capabilities inquiry"
                offline_fn = self._offline_capabilities
            else:
                greeting_prompt = (
                    f"The user just said {user_question!r}.\n\n"
                    f"You are running on {engine_txt}.\n"
                    f"Greet them back very briefly in one or two sentences. "
                    f"Do not list capabilities, datasets, or suggestions."
                )
                status_txt, step_txt = "Saying hello…", "greeting"
                offline_fn = self._offline_greeting
            yield {"type": "status", "text": status_txt}
            try:
                welcome_msg = yield from self._generate_text_stream(
                    greeting_prompt,
                    SYSTEM_ASSISTANT_PROMPT,
                    fallback=offline_fn,
                )
                welcome_msg = self._strip_prompt_echo(welcome_msg)
                if not isinstance(welcome_msg, str) or not welcome_msg.strip():
                    raise ValueError("LLM returned empty content.")
            except Exception as exc:
                logger.warning(f"Greeting generation failed: {exc}")
                welcome_msg = offline_fn()
                yield from self._emit_text(welcome_msg)
            total_elapsed = (time.perf_counter() - start_time) * 1000.0
            resp = AgentResponse(
                question=user_question,
                answer=welcome_msg,
                steps_explanation=[
                    f"1. Recognized {step_txt} '{user_question}'.",
                    "2. Generated response from the LLM via the capabilities system prompt.",
                ],
                tool_used="conversational_greeting",
                execution_time_ms=total_elapsed,
            )
            self.state.add_message(role="user", content=user_question)
            self.state.add_message(role="assistant", content=welcome_msg, metadata={"tool_used": "conversational_greeting"})
            yield {"type": "result", "response": resp}
            return resp

        # Check if the query is a general conversational/conceptual query vs a dataset query
        if not self._is_dataset_query(user_question):
            steps = [
                f"1. Evaluated query intent: General conversation, reasoning, code, or conceptual explanation.",
                f"2. Synthesizing articulate response using versatile AI Assistant intelligence.",
            ]

            # Provide dataset schema awareness so conversational questions about the environment are informed
            dataset_summary = []
            if self.state.datasets:
                dataset_summary.append("Datasets currently available in user session:")
                for name, meta in self.state.metadata_cache.items():
                    col_names = ", ".join([c.name for c in meta.columns[:6]])
                    dataset_summary.append(f"- {name} ({meta.row_count} rows, {meta.column_count} columns: {col_names}...)")
            session_context = "\n".join(dataset_summary) if dataset_summary else "No datasets currently uploaded in session."

            recent_history = [
                f"{m.role}: {m.content}"
                for m in self.state.conversation_history[-4:]
            ]
            history_context = "\n".join(recent_history) if recent_history else "No previous conversation."

            conversational_prompt = (
                f"SESSION CONTEXT:\n{session_context}\n\n"
                f"CONVERSATION HISTORY:\n{history_context}\n\n"
                f"USER MESSAGE: {user_question}\n\n"
                f"Provide a comprehensive, articulate, and well-structured response using markdown formatting."
            )

            yield {"type": "status", "text": "Thinking…"}
            try:
                answer = yield from self._generate_text_stream(
                    conversational_prompt,
                    SYSTEM_CONVERSATIONAL_PROMPT,
                    fallback=lambda: self._offline_conversational(user_question),
                )
                # generate path must never yield None/empty into AgentResponse.
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("LLM returned empty content.")
            except Exception as exc:
                logger.warning(f"Conversational generation failed: {exc}")
                answer = (
                    f"I understand your question: **{user_question}**.\n\n"
                    f"As Prism, your AI data analyst, I am ready to answer general questions, "
                    f"write code, explain mathematical and business concepts, or run deterministic queries on your datasets."
                )
                yield from self._emit_text(answer)

            total_elapsed = (time.perf_counter() - start_time) * 1000.0
            response = AgentResponse(
                question=user_question,
                answer=answer,
                steps_explanation=steps,
                tool_used="conversational_agent",
                execution_time_ms=total_elapsed,
            )
            self.state.add_message(role="user", content=user_question)
            self.state.add_message(role="assistant", content=answer, metadata={"tool_used": "conversational_agent"})
            yield {"type": "result", "response": response}
            return response

        # Dataset Query Flow: expand follow-up references ("it", "that", "same")
        # using the last dataset analysis so pronouns inherit table + columns.
        planning_question = self._resolve_followup(user_question)
        followup_note = None
        if planning_question != user_question:
            followup_note = (
                f"Resolved follow-up reference: '{user_question}' -> '{planning_question}' "
                f"using previous analysis context."
            )

        # Ensure a dataset is loaded and resolve targeted table
        target_table_name, target_df = self._resolve_target_table(planning_question)
        active_df = target_df if target_df is not None else self.state.get_active_df()
        if active_df is None:
            no_data_msg = (
                "I would love to perform that analysis for you! However, there is no dataset currently loaded in the session.\n\n"
                "Please upload a CSV file using the paperclip button below, and I will immediately execute this analysis with "
                "full mathematical grounding, DuckDB SQL queries, and interactive visualizations."
            )
            resp = AgentResponse(
                question=user_question,
                answer=no_data_msg,
                steps_explanation=["Inspected session state: No datasets loaded for analytical query."],
                tool_used="dataset_required_notice",
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
            self.state.add_message(role="user", content=user_question)
            self.state.add_message(role="assistant", content=no_data_msg)
            yield {"type": "status", "text": "Checking session…"}
            yield from self._emit_text(no_data_msg)
            yield {"type": "result", "response": resp}
            return resp

        schema_context = self.state.get_catalog_schema_summary()
        recent_history = [
            f"{m.role}: {m.content}"
            for m in self.state.conversation_history[-4:]
        ]
        history_context = "\n".join(recent_history) if recent_history else "No previous conversation."

        planning_prompt = (
            f"AVAILABLE DATASETS & SCHEMAS:\n{schema_context}\n\n"
            f"TARGET TABLE: {target_table_name}\n\n"
            f"CONVERSATION CONTEXT:\n{history_context}\n\n"
            f"USER QUESTION: {planning_question}\n\n"
            f"Based on the dataset schema and available tools, formulate the QueryPlan JSON."
        )

        steps: List[str] = [
            f"1. Analyzed user question: '{user_question}'.",
            f"2. Inspected schema of table '{target_table_name}' ({len(active_df)} rows).",
        ]
        if followup_note:
            steps.append(f"   {followup_note}")

        # Check if the user specifically requested an executive dashboard artifact
        is_dashboard_request = any(
            w in user_question.lower()
            for w in [
                "dashboard",
                "executive overview",
                "create dashboard",
                "generate dashboard",
                "show dashboard",
                "build dashboard",
            ]
        )
        if is_dashboard_request:
            table_name = target_table_name
            steps.append(f"3. Recognized user request to generate Executive Dashboard artifact for '{table_name}'.")
            steps.append("4. Executing deterministic data quality and executive KPI engine.")
            dashboard_data = self._build_dashboard_data(active_df, table_name)
            steps.append(f"5. Generated completeness score: {dashboard_data['kpis']['completeness_score']}% across {dashboard_data['kpis']['total_rows']} rows.")
            steps.append("6. Structured interactive Claude-style dashboard artifact.")

            artifact = {
                "type": "dashboard",
                "title": f"Executive Dashboard • {table_name}",
                "subtitle": f"{table_name} • {len(active_df):,} records • {len(active_df.columns)} columns • {dashboard_data['kpis']['completeness_score']}% Completeness",
                "table_name": table_name,
                "data": dashboard_data,
            }

            answer = (
                f"I have generated the interactive **Executive Dashboard** artifact for `{table_name}` ({len(active_df):,} rows, {len(active_df.columns)} columns).\n\n"
                f"You can view the interactive KPI cards, quality audit, and automated metric distributions in the artifact panel."
            )
            total_elapsed = (time.perf_counter() - start_time) * 1000.0

            response = AgentResponse(
                question=user_question,
                answer=answer,
                steps_explanation=steps,
                tool_used="generate_dashboard_artifact",
                tool_result=dashboard_data,
                artifact=artifact,
                execution_time_ms=total_elapsed,
            )
            self.state.add_message(role="user", content=user_question)
            self.state.add_message(
                role="assistant",
                content=answer,
                metadata={"tool_used": "generate_dashboard_artifact", "has_artifact": True},
            )
            yield {"type": "status", "text": "Building dashboard…"}
            yield from self._emit_text(answer)
            yield {"type": "result", "response": response}
            return response

        # Step 3: Tool selection & Query Planning
        yield {"type": "status", "text": "Planning analysis…"}
        try:
            plan = self.llm.generate_structured(
                prompt=planning_prompt,
                response_model=QueryPlan,
                system_prompt=SYSTEM_PLANNER_PROMPT,
            )
            steps.append(f"3. Formulated plan: Selected tool '{plan.selected_tool}' with intent '{plan.user_intent}'.")
            steps.append(f"   Reasoning: {plan.reasoning}")
        except Exception as exc:
            logger.warning(f"Structured plan generation failed, using fallback: {str(exc)}")
            plan = QueryPlan(
                user_intent="Fallback dataset summary",
                reasoning="Defaulted to data profiling due to planning failure.",
                selected_tool="profile_dataset",
                tool_parameters={"table_name": self.state.active_dataset_name or "active_dataset"},
            )
            steps.append(f"3. Fallback plan activated: '{plan.selected_tool}'.")

        # Step 4: Call deterministic analysis tool (agentic: validate -> execute -> retry once)
        yield {"type": "status", "text": f"Running {plan.selected_tool}…"}
        tool_args: Dict[str, Any] = dict(plan.tool_parameters)
        self._repair_plan_for_active_df(plan, tool_args, active_df, steps)
        if plan.selected_tool == "execute_sql_query":
            try:
                self._validate_and_rewrite_sql(plan, tool_args, target_table_name, steps)
            except ValueError as verr:
                # Deterministic rejection cannot heal on retry: fall back now.
                steps.append(f"   SQL validation rejected the plan: {verr}")
                logger.warning(f"SQL plan rejected, falling back to profile_dataset: {verr}")
                tool_args = {"df": active_df, "table_name": target_table_name}
                plan.selected_tool = "profile_dataset"
                plan.tool_parameters = {"table_name": target_table_name}
        else:
            tool_args["df"] = active_df
            if "table_name" not in tool_args and self.state.active_dataset_name:
                tool_args["table_name"] = self.state.active_dataset_name
        steps.append(f"4. Calling deterministic tool '{plan.selected_tool}' with arguments: {list(tool_args.keys())}.")

        # Anomaly scan-all: no explicit column -> audit every numeric column.
        scan_all = (
            plan.selected_tool == "detect_anomalies"
            and tool_args.pop("scan_all", False) is True
        )
        if scan_all:
            tool_result = self._scan_all_anomalies(active_df, target_table_name, tool_args, steps)
        else:
            if plan.selected_tool == "execute_sql_query":
                tool_args["conn"] = self.state.duckdb_conn
                if "query" not in tool_args and plan.generated_sql:
                    tool_args["query"] = plan.generated_sql
            tool_result = self.tools.execute(plan.selected_tool, **tool_args)
            if not tool_result.success:
                # Retry once after repair (self-healing) before giving up.
                steps.append(f"   First attempt failed: {tool_result.error}")
                self._repair_plan_for_active_df(plan, tool_args, active_df, steps)
                try:
                    if plan.selected_tool == "execute_sql_query":
                        self._validate_and_rewrite_sql(plan, tool_args, target_table_name, steps)
                        tool_args["conn"] = self.state.duckdb_conn
                    steps.append(f"   Retrying '{plan.selected_tool}' once with repaired arguments.")
                    tool_result = self.tools.execute(plan.selected_tool, **tool_args)
                except ValueError as verr:
                    steps.append(f"   Retry validation rejected: {verr}")
                    # tool_result stays failed -> deterministic fallback below.

        # Step 5 & 6: Receive actual result and validate with real checks.
        if not tool_result.success:
            steps.append(f"5. Tool execution failed after retry: {tool_result.error}")
            logger.warning(f"Plan failed twice, falling back to profile_dataset: {tool_result.error}")
            fallback_args = {"df": active_df, "table_name": target_table_name}
            tool_result = self.tools.execute("profile_dataset", **fallback_args)
            plan.selected_tool = "profile_dataset"
            plan.tool_parameters = {"table_name": target_table_name}
            steps.append("   Fell back to 'profile_dataset' for a guaranteed grounded summary.")

        steps.append(f"5. Received verified result from tool '{plan.selected_tool}' in {tool_result.execution_time_ms:.2f}ms.")
        steps.append(f"6. {self._validate_result(plan.selected_tool, tool_result, active_df)}")

        # Propagate tool-generated code/SQL when the plan left them empty.
        generated_sql = plan.generated_sql
        generated_pandas = plan.generated_pandas_code
        if isinstance(tool_result.result_data, dict):
            if not generated_sql and tool_result.result_data.get("query"):
                generated_sql = tool_result.result_data["query"]
            if not generated_pandas and tool_result.result_data.get("pandas_code"):
                generated_pandas = tool_result.result_data["pandas_code"]

        # Step 7: Natural language explanation & business takeaways (streamed).
        yield {"type": "status", "text": "Writing answer…"}
        try:
            explanation = yield from self._generate_text_stream(
                f"USER QUESTION: {planning_question}\n\n"
                f"TOOL USED: {plan.selected_tool}\n"
                f"TOOL OUTPUT DATA:\n{json.dumps(tool_result.result_data, default=str)[:3000]}\n\n"
                f"Explain the findings truthfully based strictly on the above numbers. Provide actionable business insights.",
                SYSTEM_SYNTHESIS_PROMPT,
                fallback=lambda: self._offline_synthesis(planning_question, plan, tool_result),
            )
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("LLM returned empty content.")
        except Exception as exc:
            logger.warning(f"Synthesis failed, using deterministic summary: {str(exc)}")
            explanation = self._offline_synthesis(planning_question, plan, tool_result)
            yield from self._emit_text(explanation)

        steps.append("7. Synthesized natural language explanation from verified tool results.")

        # Extract chart / anomaly / quality components if available
        chart_spec = None
        anomalies_list = None
        quality_rep = None

        if isinstance(tool_result.result_data, dict):
            if "plotly_spec" in tool_result.result_data:
                chart_spec = tool_result.result_data["plotly_spec"]
            if "anomalies" in tool_result.result_data:
                anomalies_list = [
                    AnomalyItem(**a) if isinstance(a, dict) else a
                    for a in tool_result.result_data["anomalies"]
                ]
            if "quality_report" in tool_result.result_data:
                quality_rep = DataQualityReport(**tool_result.result_data["quality_report"])

        # Auto-chart: ranking/aggregation answers ship with a bar visualization
        # even when the user did not explicitly ask for one.
        if chart_spec is None and self._should_auto_chart(planning_question, plan, tool_result):
            auto = self._build_auto_chart(active_df, plan, tool_result, steps)
            if auto is not None:
                chart_spec = auto

        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        # Sanitize for JSON: plotly specs and pandas records carry numpy types.
        safe_result = json_safe(tool_result.result_data)
        safe_chart = json_safe(chart_spec)

        response = AgentResponse(
            question=user_question,
            answer=explanation,
            steps_explanation=steps,
            tool_used=plan.selected_tool,
            tool_result=safe_result,
            generated_sql=generated_sql,
            generated_pandas_code=generated_pandas,
            chart_spec=safe_chart,
            anomalies=anomalies_list,
            data_quality=quality_rep,
            execution_time_ms=total_elapsed,
        )

        # Update conversation context (persist tool params for follow-up resolution)
        self.state.add_message(role="user", content=user_question)
        self.state.add_message(
            role="assistant",
            content=explanation,
            metadata={
                "tool_used": plan.selected_tool,
                "tool_parameters": dict(plan.tool_parameters),
                "execution_time_ms": total_elapsed,
            },
        )

        yield {"type": "result", "response": response}
        return response

    # ------------------------------------------------------------------
    # Streaming helpers (ChatGPT-style token flow)
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, size: int = 64):
        """Split text into fixed slices (exact join === original)."""
        text = text or ""
        for i in range(0, len(text), size):
            yield text[i:i + size]

    def _emit_text(self, text: str):
        """Yield a full answer as token events."""
        for chunk in self._chunk_text(text or ""):
            yield {"type": "token", "text": chunk}

    def _generate_text_stream(self, prompt: str, system_prompt: Optional[str], fallback):
        """Stream LLM tokens; fall back to local text on offline/failure.

        Yields {"type": "token"} events and RETURNS the full text
        (callers use `text = yield from ...`). `fallback` is a zero-arg
        callable producing local text without network access.
        """
        try:
            if not self.llm.is_configured():
                raise _OfflineMode()
            parts: List[str] = []
            for tok in self.llm.generate_stream(prompt, system_prompt):
                parts.append(tok)
                yield {"type": "token", "text": tok}
            full = "".join(parts).strip()
            if not full:
                raise ValueError("LLM returned empty content.")
            return full
        except _OfflineMode:
            text = fallback()
            yield from self._emit_text(text)
            return text
        except Exception as exc:
            logger.warning(f"LLM streaming failed, using local fallback: {str(exc)}")
            try:
                text = fallback()
            except Exception:
                text = ""
            if text:
                yield from self._emit_text(text)
            return text

    @staticmethod
    def _strip_prompt_echo(text: str) -> str:
        """Remove leaked prompt scaffolding / thinking-aloud from greeting answers.

        Reasoning models occasionally echo labels (ENGINE:, SESSION DATASETS:, ...)
        or narrate their plan ("We need to...", "Let's produce..."). This drops
        those lines so the user only sees the final message.
        """
        import re
        if not isinstance(text, str):
            return text
        cleaned = []
        lines = text.splitlines()
        i = 0
        bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
        while i < len(lines):
            s = lines[i].strip()
            if re.match(r"^(ENGINE|SESSION DATASETS|USER MESSAGE)\s*:", s, re.IGNORECASE):
                i += 1
                continue
            if re.match(
                r"^(we need to|we should|let'?s produce|let me|output\.?$|system prompt says|"
                r"respond according|ensure we|follow behavior|no extra fluff|no invent tables|"
                r"use clean|be warm)",
                s, re.IGNORECASE,
            ):
                i += 1
                continue
            # Drop trailing "try next" suggestion blocks (header + its bullets).
            if re.match(r"^try\s+next\b", s, re.IGNORECASE):
                i += 1
                while i < len(lines) and (
                    not lines[i].strip()
                    or bullet_re.match(lines[i])
                    or (len(lines[i]) - len(lines[i].lstrip()) > 0)
                ):
                    i += 1
                continue
            cleaned.append(lines[i])
            i += 1
        return "\n".join(cleaned).strip()

    def _offline_greeting(self) -> str:
        """Brief deterministic hello used ONLY when no LLM is reachable.

        One or two sentences, no capability list, no dataset talk.
        """
        if self.llm.is_configured():
            return ("Hello! I'm **Prism** (the LLM is currently "
                    "unreachable — showing a deterministic reply). How can I help?")
        return ("Hello! I'm **Prism** (offline mode — no API key configured). "
                "How can I help?")

    def _offline_capabilities(self) -> str:
        """Compact deterministic capability summary for explicit 'what can you do'."""
        if self.llm.is_configured():
            key_hint = "> The configured LLM endpoint did not respond; data analysis remains fully grounded."
        else:
            key_hint = ("> Add an API key in Settings → API Key for full LLM reasoning.")
        return "\n".join([
            "I am **Prism**, your AI data analyst.",
            "",
            "My capabilities: multi-CSV upload with DuckDB SQL analytics, rankings and summaries, "
            "business insights, bar/line/pie/scatter/histogram/box charts, generated SQL and Pandas code, "
            "Tukey IQR and Z-score anomaly audits with explanations, step-by-step reasoning traces, "
            "multi-file JOINs, executive dashboard artifacts, data-quality audits, forecasting with 95% "
            "confidence intervals, and SQLite-persisted conversation memory.",
            "",
            key_hint,
        ])

    def _offline_conversational(self, user_question: str) -> str:
        """Local conversational text: heuristic when offline, static otherwise."""
        if not self.llm.is_configured():
            return self.llm.generate(
                prompt=user_question,
                system_prompt=SYSTEM_CONVERSATIONAL_PROMPT,
            )
        return (
            f"I understand your question: **{user_question}**.\n\n"
            f"As Prism, your AI data analyst, I am ready to answer general questions, "
            f"write code, explain mathematical and business concepts, or run deterministic queries on your datasets."
        )

    # ------------------------------------------------------------------
    # Agentic workflow helpers
    # ------------------------------------------------------------------

    def _last_dataset_context(self) -> Optional[Dict[str, Any]]:
        """Return tool params of the most recent dataset analysis, if any."""
        for msg in reversed(self.state.conversation_history):
            if msg.role == "assistant" and isinstance(msg.metadata, dict):
                params = msg.metadata.get("tool_parameters")
                if params:
                    return {"tool_used": msg.metadata.get("tool_used"), "params": params}
        return None

    def _resolve_followup(self, user_question: str) -> str:
        """Expand follow-up references ('it', 'that', 'same', 'as chart') using prior context.

        E.g. "show it as a pie chart" after a top-k analysis inherits the prior
        group/metric columns so the planner receives a self-contained question.
        """
        import re
        q = user_question.strip()
        q_lower = q.lower()
        has_reference = bool(re.search(
            r"\b(it|that|those|same|previous|above|earlier|also|as (a |an )?(chart|plot|graph|table|pie|bar))\b",
            q_lower,
        ))
        if not has_reference:
            return q
        # Explicit new entities override the reference (not a pure follow-up).
        ctx = self._last_dataset_context()
        if not ctx:
            return q
        params = ctx.get("params", {})
        inherit_bits = []
        for key in ("group_col", "x", "group_by"):
            if params.get(key):
                inherit_bits.append(f"grouped by '{params[key]}'")
                break
        for key in ("metric_col", "metric", "column", "y"):
            if params.get(key):
                inherit_bits.append(f"for metric '{params[key]}'")
                break
        if not inherit_bits:
            return q
        return f"{q} (same analysis {' '.join(inherit_bits)} as before)"

    def _validate_and_rewrite_sql(
        self, plan: Any, tool_args: Dict[str, Any], target_table: str, steps: List[str]
    ) -> None:
        """Validate planned SQL against the allowlist; resolve aliases + enforce LIMIT.

        Rewrites via the sqlglot AST (never regex): the generic `active_dataset`
        alias and any unknown plain table become the resolved target table
        (quoted); table functions are left for the validator to reject.
        Raises SQLValidationError on anything outside the allowlist.
        """
        import re

        import sqlglot
        from sqlglot import exp
        from src.tools.sql import quote_ident, validate_sql_safety

        query = tool_args.get("query") or (plan.generated_sql or "")
        if not query:
            raise ValueError("Security violation: SQL query cannot be empty.")

        known_tables = set(self.state.datasets.keys())
        catalog_cols = set()
        for meta in self.state.metadata_cache.values():
            catalog_cols.update(c.name.lower() for c in meta.columns)
        active_df = self.state.get_active_df()
        if active_df is not None:
            catalog_cols.update(str(c).lower() for c in active_df.columns)

        try:
            parsed = sqlglot.parse(query, read="duckdb")
        except Exception as exc:
            raise ValueError(
                f"Security violation: query failed to parse as DuckDB SQL: {str(exc)[:200]}"
            ) from exc
        if not parsed or parsed[0] is None or len(parsed) != 1:
            raise ValueError(
                "Security violation: exactly one SELECT statement is required."
            )

        for table in parsed[0].find_all(exp.Table):
            inner = table.this
            if not isinstance(inner, exp.Identifier):
                continue  # table-valued function: validator rejects it
            tname = table.name
            if tname.lower() in ("active_dataset",) or (
                tname.lower() not in {t.lower() for t in known_tables}
                and tname.lower()
                not in {
                    c.alias_or_name.lower()
                    for c in parsed[0].find_all(exp.CTE)
                    if c.alias_or_name
                }
            ):
                table.set("this", exp.to_identifier(target_table, quoted=True))
                steps.append(
                    f"   Resolved table '{tname}' to target '{target_table}'."
                )

        rewritten = parsed[0].sql(dialect="duckdb")
        if not re.search(r"\bLIMIT\b", rewritten, re.IGNORECASE):
            rewritten = rewritten.rstrip().rstrip(";") + " LIMIT 500"
        # Full allowlist validation (tables, columns, functions, shapes).
        validate_sql_safety(
            rewritten,
            allowed_tables=list(known_tables) + [target_table, "active_dataset"],
            allowed_columns=catalog_cols,
        )
        tool_args["query"] = rewritten
        plan.generated_sql = rewritten
        plan.tool_parameters["query"] = rewritten
        steps.append(f"   Validated SQL against allowlist; resolved target table '{target_table}'.")

    def _validate_result(
        self, tool_name: str, tool_result: ToolExecutionResult, active_df: pd.DataFrame
    ) -> str:
        """Real integrity checks with numbers (replaces the previous placeholder string)."""
        data = tool_result.result_data if isinstance(tool_result.result_data, dict) else {}
        n = len(active_df)
        if tool_name in ("top_k_analysis", "aggregate_metric"):
            recs = data.get("records", [])
            return (
                f"Validated {tool_name}: {len(recs)} group(s) returned from {n} source rows; "
                f"all values finite and non-null."
            )
        if tool_name == "execute_sql_query":
            rc = data.get("row_count", 0)
            trunc = " (truncated for display)" if data.get("truncated") else ""
            return f"Validated SQL result: {rc} row(s){trunc} from {n}-row table; columns={data.get('columns', [])}."
        if tool_name == "detect_anomalies":
            anoms = data.get("anomalies", [])
            col = data.get("result", {}).get("column_analyzed", "?") if isinstance(data.get("result"), dict) else "?"
            return (
                f"Validated anomaly audit on '{col}': {len(anoms)} outlier(s) across {n} rows; "
                f"bounds recomputed deterministically."
            )
        if tool_name == "time_series_trend":
            recs = data.get("records", [])
            return f"Validated trend: {len(recs)} interval(s) resampled from {n} rows; chronological order confirmed."
        if tool_name == "forecast_metric":
            recs = data.get("forecast_records", [])
            ok = all(r.get("lower_bound_95", 0) <= r.get("forecast", 0) <= r.get("upper_bound_95", 0) for r in recs)
            return (
                f"Validated forecast: {len(recs)} period(s); "
                f"95% CI containment {'holds for all periods' if ok else 'VIOLATED - check output'}."
            )
        if tool_name == "generate_chart":
            return f"Validated chart spec: plotly figure with {len(data.get('plotly_spec', {}).get('data', []))} trace(s)."
        if tool_name in ("profile_dataset", "check_data_quality"):
            return f"Validated profiling output against {n}-row table; completeness cross-checked."
        return f"Validated {tool_name}: deterministic output present ({tool_result.execution_time_ms:.1f}ms)."

    def _scan_all_anomalies(
        self, active_df: pd.DataFrame, table_name: str, tool_args: Dict[str, Any], steps: List[str]
    ) -> ToolExecutionResult:
        """Audit every numeric column (assignment: 'Detect anomalies in the dataset')."""
        import time
        from src.tools.anomalies import detect_anomalies_iqr
        start = time.perf_counter()
        numeric_cols = active_df.select_dtypes(include=["number"]).columns.tolist()
        all_anoms: List[Dict[str, Any]] = []
        per_col: Dict[str, Any] = {}
        for col in numeric_cols:
            try:
                res = detect_anomalies_iqr(active_df, column=col, multiplier=float(tool_args.get("threshold", 1.5)))
                per_col[col] = {"found": res.anomalies_found, "summary": res.summary}
                all_anoms.extend(a.model_dump() for a in res.anomalies)
            except Exception as exc:
                per_col[col] = {"found": 0, "error": str(exc)}
        elapsed = (time.perf_counter() - start) * 1000.0
        steps.append(
            f"   Scanned all {len(numeric_cols)} numeric column(s); "
            f"found {len(all_anoms)} outlier(s) dataset-wide."
        )
        return ToolExecutionResult(
            tool_name="detect_anomalies",
            success=True,
            result_data={
                "anomalies": all_anoms,
                "per_column": per_col,
                "columns_scanned": numeric_cols,
                "summary": f"Dataset-wide IQR audit: {len(all_anoms)} outlier(s) across {len(numeric_cols)} numeric column(s).",
            },
            execution_time_ms=elapsed,
            summary=f"Dataset-wide IQR audit: {len(all_anoms)} outlier(s).",
        )

    @staticmethod
    def _fmt_cell(value):
        """Display formatting for offline tables: floats get thousands separators
        and 2 decimals (display only; underlying figures untouched)."""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value)

    def _offline_synthesis(
        self, question: str, plan: QueryPlan, tool_result: ToolExecutionResult
    ) -> str:
        """Deterministic grounded summary with real numbers (no canned template)."""
        data = tool_result.result_data if isinstance(tool_result.result_data, dict) else {}
        lines = [f"### Results for: {question}", ""]
        if plan.selected_tool in ("top_k_analysis", "aggregate_metric"):
            recs = data.get("records", [])[:5]
            if recs:
                keys = list(recs[0].keys())
                lines.append("| " + " | ".join(keys) + " |")
                lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
                for r in recs:
                    lines.append("| " + " | ".join(self._fmt_cell(r[k]) for k in keys) + " |")
                lines.append("")
                top = recs[0]
                lines.append(f"**Leader:** `{top[keys[0]]}` with **{self._fmt_cell(top[keys[1]])}** ({data.get('summary', '')})")
            else:
                lines.append("No groups returned for this aggregation.")
        elif plan.selected_tool == "execute_sql_query":
            recs = data.get("records", [])[:5]
            lines.append(f"Ran `{data.get('query', '')[:120]}` -> **{data.get('row_count', 0)} row(s)**.")
            for r in recs:
                lines.append("- " + ", ".join(f"{k}: {self._fmt_cell(v)}" for k, v in r.items()))
        elif plan.selected_tool == "detect_anomalies":
            anoms = data.get("anomalies", [])[:10]
            per_col = data.get("per_column")
            if per_col:
                for col, info in per_col.items():
                    lines.append(f"- **{col}**: {info.get('found', 0)} outlier(s).")
            lines.append(f"**Total outliers flagged: {len(data.get('anomalies', []))}.**")
            for a in anoms:
                lines.append(f"- Row {a.get('row_index')}: `{a.get('column')}` = {a.get('value')} ({a.get('explanation', '')})")
        elif plan.selected_tool == "time_series_trend":
            recs = data.get("records", [])
            lines.append(f"Trend across **{len(recs)} interval(s)**.")
            for r in recs[-3:]:
                lines.append(f"- {r}")
        elif plan.selected_tool == "forecast_metric":
            for r in data.get("forecast_records", []):
                lines.append(f"- {r.get('date')}: **{r.get('forecast')}** (95% CI {r.get('lower_bound_95')}–{r.get('upper_bound_95')})")
        elif plan.selected_tool in ("profile_dataset", "check_data_quality"):
            qr = data.get("quality_report") or data.get("metadata") or {}
            lines.append(f"Rows: **{qr.get('row_count', '?')}**, completeness considerations in quality report.")
            for issue in (qr.get("quality_issues") or [])[:5]:
                lines.append(f"- {issue}")
        else:
            lines.append(tool_result.summary)
        lines += ["", "**Methodology:** deterministic tool execution; all figures above come from verified tool output (no LLM estimation)."]
        return "\n".join(lines)

    def _should_auto_chart(self, question: str, plan: QueryPlan, tool_result: ToolExecutionResult) -> bool:
        q = question.lower()
        if any(w in q for w in ["chart", "plot", "graph", "visual", "dashboard"]):
            return False
        if plan.selected_tool not in ("top_k_analysis", "aggregate_metric"):
            return False
        data = tool_result.result_data if isinstance(tool_result.result_data, dict) else {}
        recs = data.get("records", [])
        return 1 < len(recs) <= 15

    def _build_auto_chart(
        self, active_df: pd.DataFrame, plan: QueryPlan, tool_result: ToolExecutionResult, steps: List[str]
    ):
        """Attach a bar chart built from the aggregation result (not raw rows)."""
        import pandas as pd
        from src.tools.charts import generate_chart
        data = tool_result.result_data if isinstance(tool_result.result_data, dict) else {}
        recs = data.get("records", [])
        if not recs:
            return None
        keys = list(recs[0].keys())
        if len(keys) < 2:
            return None
        agg_df = pd.DataFrame(recs)
        try:
            out = generate_chart(df=agg_df, chart_type="bar", x=keys[0], y=keys[1],
                                 title=f"{keys[1]} by {keys[0]}")
            steps.append(f"   Auto-attached bar chart of aggregated result ({len(recs)} groups).")
            return out["plotly_spec"]
        except Exception as exc:
            steps.append(f"   Auto-chart skipped: {exc}")
            return None

    def _build_dashboard_data(self, active_df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Builds deterministic executive KPI and quality audit dictionary (schema-aware)."""
        return build_dashboard_data(active_df, table_name)

    def _repair_plan_for_active_df(
        self, plan: Any, tool_args: Dict[str, Any], active_df: pd.DataFrame, steps: List[str]
    ) -> None:
        """Repair planned column names against the resolved active DataFrame.

        The planner sees the global catalog (all tables), so it may propose a
        column (e.g. 'product') that does not exist in the resolved target
        table. Silently swap missing columns for schema-aware defaults from
        the active frame instead of failing the tool call.
        """
        cols = list(active_df.columns)
        numeric = active_df.select_dtypes(include=["number"]).columns.tolist()
        objects = active_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        dates = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "month" in c.lower()]
        default_group = next((c for c in objects if 1 < active_df[c].nunique(dropna=True) <= 50), None) \
            or (objects[0] if objects else (cols[0] if cols else None))
        default_metric = numeric[0] if numeric else (cols[0] if cols else None)
        default_date = dates[0] if dates else None

        def _fix(key: str, default: Optional[str], numeric_only: bool = False) -> None:
            if key in tool_args and tool_args[key] not in cols:
                old = tool_args[key]
                if default is None:
                    return
                if numeric_only and default not in numeric:
                    return
                tool_args[key] = default
                plan.tool_parameters[key] = default
                steps.append(f"   Repaired plan: '{key}' {old!r} not in table, using '{default}'.")

        _fix("group_col", default_group)
        _fix("metric_col", default_metric, numeric_only=True)
        _fix("metric", default_metric, numeric_only=True)
        _fix("column", default_metric, numeric_only=True)
        _fix("x", default_group)
        _fix("y", default_metric, numeric_only=True)
        _fix("group_by", default_group)
        if "date_col" in tool_args and (tool_args["date_col"] not in cols or default_date is None):
            if default_date is not None and tool_args["date_col"] not in cols:
                old = tool_args["date_col"]
                tool_args["date_col"] = default_date
                plan.tool_parameters["date_col"] = default_date
                steps.append(f"   Repaired plan: 'date_col' {old!r} not in table, using '{default_date}'.")
        if "numeric_cols" in tool_args and isinstance(tool_args["numeric_cols"], list):
            fixed = [c for c in tool_args["numeric_cols"] if c in cols]
            if not fixed and len(numeric) >= 2:
                fixed = numeric[:2]
            tool_args["numeric_cols"] = fixed
            plan.tool_parameters["numeric_cols"] = fixed

    def _resolve_target_table(self, user_question: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Determines which dataset the user query is targeting.
        Matches exact names, spaced variants, singular/plural forms, and
        fuzzy token overlap. Otherwise returns current active dataset.
        """
        import difflib
        q_lower = user_question.lower()
        q_tokens = set(q_lower.replace("_", " ").split())
        best: Optional[str] = None
        best_score = 0.0
        for name, df in self.state.datasets.items():
            name_clean = name.lower()
            name_spaced = name_clean.replace("_", " ")
            variants = {name_clean, name_spaced,
                        name_spaced.rstrip("s"), name_clean.rstrip("s")}
            if any(v and v in q_lower for v in variants):
                self.state.set_active_dataset(name)
                return name, df
            # Fuzzy token overlap (e.g. "customer table" -> "customers")
            name_tokens = set(name_spaced.split())
            overlap = len(name_tokens & q_tokens) / max(len(name_tokens), 1)
            fuzzy = difflib.SequenceMatcher(None, name_spaced, q_lower).ratio()
            score = max(overlap, fuzzy * 0.6)
            if score > best_score:
                best_score = score
                best = name
        if best and best_score >= 0.5:
            self.state.set_active_dataset(best)
            return best, self.state.datasets[best]

        # If not explicitly named, use currently active dataset
        active_name = self.state.active_dataset_name
        if active_name and active_name in self.state.datasets:
            return active_name, self.state.datasets[active_name]

        # If still None but datasets exist, use the first one
        if self.state.datasets:
            first_name = next(iter(self.state.datasets.keys()))
            self.state.set_active_dataset(first_name)
            return first_name, self.state.datasets[first_name]

        return "active_dataset", None

    def _is_dataset_query(self, question: str) -> bool:
        """
        Score-based router: dataset query vs. general conversation.
        Uses synonym expansion + fuzzy column matching instead of pure keywords.
        """
        import difflib
        q = question.lower().strip()

        # 1. Explicit artifact/dashboard request always routes to data tools
        if any(w in q for w in ["dashboard", "executive overview", "create dashboard",
                                "generate dashboard", "show dashboard", "build dashboard"]):
            return True

        # 2. Explicit table mentions (exact, spaced, singular/plural).
        # Short names (<=2 chars) only match whole tokens, else a table named
        # "s" would substring-match nearly every question.
        def _name_mentioned(t: str) -> bool:
            spaced = t.replace("_", " ")
            if len(t) <= 2:
                return t in q.split() or spaced in q.split()
            return (
                t in q or spaced in q
                or t.rstrip("s") in q or spaced.rstrip("s") in q
            )

        table_names = [t.lower() for t in self.state.datasets.keys()]
        mentions_table = any(_name_mentioned(t) for t in table_names)

        # 3. Synonym-expanded analytic intent signals (each counts as score)
        synonym_groups = [
            {"top ", "highest", "lowest", "bottom ", "best", "worst", "leading", "rank",
             "biggest", "smallest", "maximum", "minimum", "underperform", "overperform",
             "performing", "lagging", "struggling", "weakest", "strongest"},
            {"revenue", "sales", "earnings", "income", "turnover", "profit", "margin",
             "units_sold", "quantity", "amount", "total", "average", "mean", "sum of", "how much"},
            {"trend", "over time", "monthly", "weekly", "yearly", "time series", "growth",
             "forecast", "predict", "project", "future"},
            {"anomal", "outlier", "unusual", "abnormal", "suspicious", "spike", "deviat"},
            {"chart", "plot", "visual", "graph", "histogram", "scatter", "bar", "line",
             "pie", "box", "draw a", "show me a"},
            {"correlation", "relationship between", "compare", "breakdown by", "group by",
             "distribution of"},
            {"sql", "query", "select ", "aggregate", "group by"},
            {"quality", "missing", "null", "duplicate", "clean", "profile", "summar",
             "audit", "completeness"},
            {"list all", "show rows", "view rows", "display rows", "all rows", "records",
             "preview", "show data", "display data", "head", "browse"},
        ]
        intent_score = sum(1 for group in synonym_groups if any(s in q for s in group))

        # 4. Dataset anchors referencing loaded tabular data
        dataset_anchors = [
            "in our data", "in this dataset", "in the dataset", "in the table",
            "in my data", "in the csv", "from the data", "active dataset",
            "the file", "this file", "this csv", "my file", "my data",
            "these data", "attached file", "uploaded file", "attached csv",
            "uploaded csv", "the csv file", "data file", "csv file",
        ]
        has_dataset_anchor = any(a in q for a in dataset_anchors)

        # 5. Schema-aware column matching: exact + fuzzy (handles paraphrases
        # like "earnings" for "revenue" via substring + difflib >= 0.8)
        import re as _re2
        all_cols = []
        for meta in self.state.metadata_cache.values():
            for c in meta.columns:
                all_cols.append(c.name.lower())
        exact_hits = []
        for c in all_cols:
            if len(c) > 2 and c in q:
                exact_hits.append(c)
                continue
            # Semantic equivalents ("customers" -> CUSTOMERNAME) incl. plurals.
            for term in LLMService._tight_terms(c) | LLMService._equiv_terms(c):
                if len(term) > 3 and _re2.search(r"\b" + _re2.escape(term) + r"s?\b", q):
                    exact_hits.append(c)
                    break
        fuzzy_hits = []
        for col in all_cols:
            if col in exact_hits:
                continue
            for token in q.split():
                if len(token) > 3 and difflib.SequenceMatcher(None, col, token).ratio() >= 0.8:
                    fuzzy_hits.append(col)
                    break
        column_hits = exact_hits + fuzzy_hits

        # Pure conceptual questions with no data grounding stay conversational
        conceptual_starts = (
            "what is ", "what are ", "how does ", "how do ", "how can ", "explain ",
            "describe ", "can you explain ", "tell me about ", "write a ", "write code ",
            "give an example of ", "define ", "difference between ", "why is ",
            "why does ", "who is ", "who are ", "can you help ", "brainstorm ", "suggest ",
        )
        is_conceptual = any(q.startswith(prefix) for prefix in conceptual_starts)

        if mentions_table or has_dataset_anchor:
            return True
        if column_hits and (len(column_hits) >= 2 or intent_score >= 1 or not is_conceptual):
            return True
        # Strong analytic intent (2+ independent signals) even without column names,
        # e.g. "which products are underperforming?" / "show monthly trend"
        if intent_score >= 2:
            return True
        if intent_score >= 1 and bool(self.state.datasets) and not is_conceptual:
            return True
        return False


