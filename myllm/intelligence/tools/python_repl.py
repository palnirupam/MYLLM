"""
myllm.intelligence.tools.python_repl — Sandboxed Python execution tool.
"""

import uuid
from typing import Any, Dict, Optional
from myllm.intelligence.tools.base import BaseTool
from myllm.intelligence.tools.sandbox import ProcessSandbox, SandboxConfig, SandboxExecutionResult
from myllm.intelligence.schemas import ToolResult


class SandboxedPythonTool(BaseTool):
    """
    Executes Python scripts in an isolated subprocess with strict memory, timeout,
    and filesystem protections.
    """

    def __init__(self, sandbox: Optional[ProcessSandbox] = None):
        self.sandbox = sandbox or ProcessSandbox(SandboxConfig(timeout_seconds=3.0))

    @property
    def name(self) -> str:
        return "python_repl"

    @property
    def description(self) -> str:
        return (
            "Executes Python 3 code in an isolated sandbox. "
            "Returns stdout, stderr, and execution status. "
            "Use for calculations, data transformation, algorithms, and logic tests."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python 3 script to execute.",
                }
            },
            "required": ["code"],
        }

    def execute(self, code: str = "", **kwargs: Any) -> ToolResult:
        call_id = str(uuid.uuid4())[:8]
        code_str = code.strip()

        if not code_str:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=None,
                error="Empty code string provided to Python sandbox.",
            )

        res: SandboxExecutionResult = self.sandbox.run_python_snippet(code_str)

        if res.timed_out:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=res.stdout,
                error=f"Execution timed out after {self.sandbox.config.timeout_seconds}s.",
                execution_time_ms=res.duration_ms,
            )

        if res.exit_code != 0:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=res.stdout,
                error=res.stderr or f"Process failed with exit code {res.exit_code}",
                execution_time_ms=res.duration_ms,
            )

        return ToolResult(
            call_id=call_id,
            tool_name=self.name,
            success=True,
            output=res.stdout.strip(),
            error=None,
            execution_time_ms=res.duration_ms,
        )
