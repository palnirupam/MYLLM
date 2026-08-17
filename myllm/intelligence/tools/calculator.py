"""
myllm.intelligence.tools.calculator — Safe AST-based mathematical evaluation tool.
"""

import ast
import math
import operator
import uuid
from typing import Any, Dict
from myllm.intelligence.tools.base import BaseTool
from myllm.intelligence.schemas import ToolResult, ToolError

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return float(_ALLOWED_CONSTANTS[node.id])
        raise ToolError("calculator", f"Unknown constant or variable: '{node.id}'")
    elif isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("calculator", "Division or modulo by zero.")
        # Prevent runaway exponentiation
        if isinstance(node.op, ast.Pow) and (abs(left) > 10000 or abs(right) > 1000):
            raise ToolError("calculator", "Exponentiation result exceeds safe computation bounds.")
        return float(_ALLOWED_OPERATORS[type(node.op)](left, right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _eval_node(node.operand)
        return float(_ALLOWED_OPERATORS[type(node.op)](operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            args = [_eval_node(arg) for arg in node.args]
            func = _ALLOWED_FUNCTIONS[node.func.id]
            try:
                return float(func(*args))
            except Exception as e:
                raise ToolError("calculator", f"Math function error '{node.func.id}': {str(e)}")
        raise ToolError("calculator", f"Disallowed function call in calculator expression.")
    else:
        raise ToolError("calculator", f"Unsupported mathematical syntax: {type(node).__name__}")


class SafeCalculatorTool(BaseTool):
    """
    Evaluates pure mathematical and arithmetic expressions securely without eval() or arbitrary code.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Safely calculates mathematical expressions. "
            "Supports basic arithmetic (+, -, *, /, //, %, **), constants (pi, e), "
            "and functions (sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil)."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '25 * 40 + sqrt(144)'.",
                }
            },
            "required": ["expression"],
        }

    def execute(self, expression: str = "", **kwargs: Any) -> ToolResult:
        call_id = str(uuid.uuid4())[:8]
        expr = expression.strip()

        if not expr:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=None,
                error="Empty expression provided to calculator.",
            )

        try:
            tree = ast.parse(expr, mode="eval")
            result_val = _eval_node(tree)

            # Format cleanly (int if integer value)
            if math.isfinite(result_val) and result_val.is_integer():
                formatted_output = int(result_val)
            else:
                formatted_output = result_val

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                output=formatted_output,
                error=None,
            )
        except SyntaxError as se:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Syntax error in mathematical expression: {se.msg}",
            )
        except ToolError as te:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=None,
                error=te.message,
            )
        except Exception as e:
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Evaluation failed: {str(e)}",
            )
