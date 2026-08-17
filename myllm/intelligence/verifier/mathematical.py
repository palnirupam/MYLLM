"""
myllm.intelligence.verifier.mathematical — Mathematical consistency verifier.
"""

import re
import ast
import operator
from typing import Optional, List, Dict, Any
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.schemas import VerificationStatus


_SAFE_OPERATORS = {
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


def _safe_eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_math(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        return _SAFE_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        operand = _safe_eval_math(node.operand)
        return _SAFE_OPERATORS[type(node.op)](operand)
    else:
        raise ValueError("Unsupported mathematical AST node.")


class MathematicalVerifier(BaseVerifier):
    """
    Verifies mathematical and arithmetic consistency in candidate answers.
    """

    EQUATION_REGEX = re.compile(r"(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)")

    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        # Look for explicit equations in answer
        matches = self.EQUATION_REGEX.findall(candidate_answer)
        if not matches:
            # If no arithmetic equation is asserted, return UNVERIFIED (neutral for non-math)
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                score=0.5,
                verifier_name="MathematicalVerifier",
                critique="No explicit arithmetic equations found in response to verify.",
            )

        mismatches = []
        for expr_str, asserted_val_str in matches:
            try:
                tree = ast.parse(expr_str, mode='eval')
                computed = _safe_eval_math(tree)
                asserted = float(asserted_val_str)
                if abs(computed - asserted) > 1e-4:
                    mismatches.append(f"{expr_str} equals {computed}, not {asserted}")
            except Exception as e:
                mismatches.append(f"Could not parse equation '{expr_str}': {e}")

        if mismatches:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=0.1,
                verifier_name="MathematicalVerifier",
                critique=f"Arithmetic contradictions detected: {'; '.join(mismatches)}",
                details={"mismatches": mismatches},
            )

        return VerificationResult(
            status=VerificationStatus.PASS,
            score=1.0,
            verifier_name="MathematicalVerifier",
            details={"verified_equations_count": len(matches)},
        )
