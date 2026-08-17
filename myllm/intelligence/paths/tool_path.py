"""
myllm.intelligence.paths.tool_path — Tool execution path.
Coordinates tool invocation, observation synthesis, and tool result propagation.
"""

import re
from typing import Optional, List, Dict, Any
from myllm.intelligence.paths.base import BasePath, PathOutput
from myllm.intelligence.schemas import ExecutionState, ToolResult
from myllm.intelligence.tools.base import ToolRegistry
from myllm.runtime.interfaces.base import InferenceRuntime


class ToolPath(BasePath):
    """
    Executes tool operations, captures observations, and synthesizes concise verified answers.
    """

    MATH_EXPR_EXTRACTOR = re.compile(
        r"(?:calculate|compute|solve|what is|evaluate)?\s*([0-9\.\s\+\-\*\/\(\)\^\%]+(?:sqrt|sin|cos|tan|log|exp|abs|pi|e)?.*)",
        re.IGNORECASE
    )

    CODE_BLOCK_EXTRACTOR = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tool_registry = tool_registry or ToolRegistry()

    def execute(
        self,
        query: str,
        state: ExecutionState,
        runtime: InferenceRuntime,
    ) -> PathOutput:
        tool_results: List[Dict[str, Any]] = []
        synthesized_text = ""
        est_tokens = 0

        # Check if query requests Python code execution
        code_match = self.CODE_BLOCK_EXTRACTOR.search(query)
        if code_match and "python_repl" in self.tool_registry.list_tools():
            code_snippet = code_match.group(1).strip()
            state.tool_calls_made += 1
            res: ToolResult = self.tool_registry.invoke("python_repl", code=code_snippet)
            state.tool_history.append(res)
            tool_results.append({
                "tool": "python_repl",
                "success": res.success,
                "output": str(res.output),
                "error": res.error,
                "duration_ms": res.execution_time_ms,
            })

            if res.success:
                synthesized_text = f"Execution Output:\n{res.output}"
            else:
                synthesized_text = f"Execution Failed: {res.error}"

        # Otherwise check if calculator can evaluate expression
        elif "calculator" in self.tool_registry.list_tools():
            # Extract expression or pass query
            math_match = self.MATH_EXPR_EXTRACTOR.search(query)
            expr_to_eval = math_match.group(1).strip() if math_match else query

            # Remove trailing question marks or punctuation
            expr_to_eval = expr_to_eval.rstrip("?=: ")

            state.tool_calls_made += 1
            res: ToolResult = self.tool_registry.invoke("calculator", expression=expr_to_eval)
            state.tool_history.append(res)
            tool_results.append({
                "tool": "calculator",
                "success": res.success,
                "output": str(res.output),
                "error": res.error,
                "duration_ms": res.execution_time_ms,
            })

            if res.success:
                synthesized_text = f"{expr_to_eval} = {res.output}"
            else:
                # Fallback to model generation if calculator tool couldn't parse expression
                gen = runtime.generate(prompt=query, max_new_tokens=state.budget.max_tokens)
                synthesized_text = gen.strip()

        else:
            # Fallback if no specific tool matches
            gen = runtime.generate(prompt=query, max_new_tokens=state.budget.max_tokens)
            synthesized_text = gen.strip()

        est_tokens = max(1, len(synthesized_text.split()))
        state.tokens_consumed += est_tokens

        return PathOutput(
            text=synthesized_text,
            tokens_generated=est_tokens,
            tool_results=tool_results,
            confidence_estimate=0.95 if any(r.get("success") for r in tool_results) else 0.60,
            metadata={"path_type": "TOOL", "tools_invoked": len(tool_results)},
        )
