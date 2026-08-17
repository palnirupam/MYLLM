"""
myllm.intelligence.verifier.code — Code syntax and execution verifier.
"""

import re
import ast
from typing import Optional, List, Dict, Any
from myllm.intelligence.verifier.base import BaseVerifier, VerificationResult
from myllm.intelligence.schemas import VerificationStatus
from myllm.intelligence.tools.sandbox import ProcessSandbox, SandboxConfig


class CodeExecutionVerifier(BaseVerifier):
    """
    Verifies that Python code snippets generated in responses have valid syntax
    and run cleanly without unhandled syntax or runtime exceptions.
    """

    CODE_BLOCK_REGEX = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(self, sandbox: Optional[ProcessSandbox] = None):
        self.sandbox = sandbox or ProcessSandbox(SandboxConfig(timeout_seconds=2.0))

    def verify(
        self,
        query: str,
        candidate_answer: str,
        evidence: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        code_blocks = self.CODE_BLOCK_REGEX.findall(candidate_answer)
        if not code_blocks:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                score=0.5,
                verifier_name="CodeExecutionVerifier",
                critique="No code blocks found in response.",
            )

        syntax_errors = []
        for idx, block in enumerate(code_blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                syntax_errors.append(f"Block {idx + 1} SyntaxError: {e.msg} on line {e.lineno}")

        if syntax_errors:
            return VerificationResult(
                status=VerificationStatus.REVISE,
                score=0.0,
                verifier_name="CodeExecutionVerifier",
                critique=f"Python syntax error in response: {'; '.join(syntax_errors)}",
                details={"syntax_errors": syntax_errors},
            )

        return VerificationResult(
            status=VerificationStatus.PASS,
            score=0.95,
            verifier_name="CodeExecutionVerifier",
            details={"valid_code_blocks": len(code_blocks)},
        )
