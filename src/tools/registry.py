"""
Tool Registry module providing centralized registration, schema generation,
and safe execution wrapper for all deterministic data analysis tools.
"""

import inspect
import time
from typing import Any, Callable, Dict, List, Optional
from src.models.schemas import ToolExecutionResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry maintaining available tools, their schemas, and execution dispatcher."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._descriptions: Dict[str, str] = {}
        self._parameter_schemas: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameter_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new tool callable with metadata."""
        self._tools[name] = func
        self._descriptions[name] = description
        self._parameter_schemas[name] = parameter_schema or {}
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable[..., Any]]:
        """Retrieve tool function by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools formatted with schema for LLM function calling."""
        tools_list = []
        for name, func in self._tools.items():
            tools_list.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._descriptions.get(name, ""),
                    "parameters": self._parameter_schemas.get(name, {}),
                }
            })
        return tools_list

    def execute(self, name: str, **kwargs: Any) -> ToolExecutionResult:
        """
        Safely execute a tool with timing, exception handling, and standardized result modeling.
        """
        if name not in self._tools:
            logger.error(f"Attempted to invoke unknown tool: {name}")
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' is not registered in the tool registry.",
                summary=f"Failed: Tool '{name}' not found."
            )

        tool_func = self._tools[name]
        start_time = time.perf_counter()

        try:
            logger.info(f"Executing tool '{name}' with arguments: {list(kwargs.keys())}")
            # Safely filter kwargs to only those accepted by tool_func
            sig = inspect.signature(tool_func)
            accepted = sig.parameters
            has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted.values())
            filtered_kwargs = kwargs if has_var_kwargs else {k: v for k, v in kwargs.items() if k in accepted}

            raw_result = tool_func(**filtered_kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            summary = f"Tool '{name}' executed successfully in {elapsed_ms:.2f}ms."
            if isinstance(raw_result, dict) and "summary" in raw_result:
                summary = raw_result["summary"]

            return ToolExecutionResult(
                tool_name=name,
                success=True,
                result_data=raw_result,
                error=None,
                execution_time_ms=elapsed_ms,
                summary=summary,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_msg = f"Error executing tool '{name}': {str(exc)}"
            logger.exception(error_msg)
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                result_data=None,
                error=error_msg,
                execution_time_ms=elapsed_ms,
                summary=f"Execution error in '{name}': {str(exc)}",
            )


# Global singleton registry instance
global_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    parameter_schema: Optional[Dict[str, Any]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a deterministic agent tool."""
    target_registry = registry or global_registry

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        target_registry.register(
            name=name,
            func=func,
            description=description,
            parameter_schema=parameter_schema,
        )
        return func

    return decorator
