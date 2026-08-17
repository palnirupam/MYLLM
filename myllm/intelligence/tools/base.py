"""
myllm.intelligence.tools.base — Base Tool abstractions and ToolRegistry.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import time
from myllm.intelligence.schemas import ToolResult, ToolError


class BaseTool(ABC):
    """
    Abstract interface for all Dhruva tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema format describing accepted parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Executes the tool with given arguments and returns a structured ToolResult.
        """
        pass


class ToolRegistry:
    """
    Registry for managing, discovering, and invoking tools.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance."""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, got {type(tool)}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieves a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Returns list of registered tool names."""
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns JSON schema definitions for all registered tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Invokes a tool by name with exception handling and execution timing.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                call_id="err",
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' is not registered in ToolRegistry. Available: {self.list_tools()}",
                execution_time_ms=0.0,
            )

        start_time = time.time()
        try:
            result = tool.execute(**kwargs)
            result.execution_time_ms = (time.time() - start_time) * 1000.0
            return result
        except ToolError as te:
            return ToolResult(
                call_id="err",
                tool_name=name,
                success=False,
                output=None,
                error=te.message,
                execution_time_ms=(time.time() - start_time) * 1000.0,
            )
        except Exception as e:
            return ToolResult(
                call_id="err",
                tool_name=name,
                success=False,
                output=None,
                error=f"Unexpected error executing tool '{name}': {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000.0,
            )
