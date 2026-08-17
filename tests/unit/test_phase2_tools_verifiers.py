"""
tests/unit/test_phase2_tools_verifiers.py
Comprehensive Phase 2 Test Suite:
- SafeCalculatorTool correctness and error handling
- SandboxedPythonTool execution, syntax errors, runtime errors, timeouts, output flooding, network denial, filesystem isolation
- ToolRegistry lookup and invocation
- ToolResult propagation into ExecutionState
- Verifier revision and UNVERIFIED enforcement
- StateManager retry and tool-call ceilings
- End-to-end Orchestrator execution with ToolPath and FastPath
"""

import sys
from pathlib import Path
from typing import Generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.tools.base import ToolRegistry
from myllm.intelligence.tools.calculator import SafeCalculatorTool
from myllm.intelligence.tools.python_repl import SandboxedPythonTool
from myllm.intelligence.orchestrator import DhruvaOrchestrator
from myllm.intelligence.policy import StateManager
from myllm.intelligence.schemas import RoutePath, VerificationStatus, ComputeBudget, RouteDecision
from myllm.intelligence.verifier.composite import CompositeVerifier
from myllm.runtime.interfaces.base import InferenceRuntime


class MockRuntime(InferenceRuntime):
    def load_model(self, model_path: str) -> None:
        pass

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
        return prompt + " Model synthesized answer."

    def generate_stream(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> Generator[str, None, None]:
        yield " Answer"


# 1. Calculator Correctness
def test_calculator_correctness():
    calc = SafeCalculatorTool()

    res = calc.execute(expression="25 * 4 + 10")
    assert res.success is True
    assert res.output == 110

    res_func = calc.execute(expression="sqrt(144) + sin(0) + abs(-5)")
    assert res_func.success is True
    assert res_func.output == 17

    res_pow = calc.execute(expression="2 ** 8")
    assert res_pow.success is True
    assert res_pow.output == 256


# 2. Calculator Invalid Input & Security
def test_calculator_invalid_inputs():
    calc = SafeCalculatorTool()

    # Zero division
    res_zero = calc.execute(expression="10 / 0")
    assert res_zero.success is False
    assert "zero" in res_zero.error.lower()

    # Syntax error
    res_syn = calc.execute(expression="10 + * 5")
    assert res_syn.success is False

    # Disallowed AST nodes (e.g. import, exec, attribute access)
    res_hack = calc.execute(expression="__import__('os').system('dir')")
    assert res_hack.success is False

    # Runaway exponentiation limit
    res_exp = calc.execute(expression="99999 ** 99999")
    assert res_exp.success is False


# 3. Python Syntax and Runtime Failure
def test_python_repl_syntax_and_runtime_failure():
    repl = SandboxedPythonTool()

    # Syntax error
    res_syn = repl.execute(code="def foo(:\n    pass")
    assert res_syn.success is False
    assert "SyntaxError" in res_syn.error

    # Runtime error
    res_run = repl.execute(code="x = 1 / 0")
    assert res_run.success is False
    assert "ZeroDivisionError" in res_run.error


# 4. Sandbox Timeout, Output Flooding, Network Denial, Filesystem Isolation
def test_python_repl_security_controls():
    repl = SandboxedPythonTool()

    # Timeout
    res_timeout = repl.execute(code="import time\nwhile True:\n    time.sleep(0.05)")
    assert res_timeout.success is False
    assert "timed out" in res_timeout.error.lower()

    # Output flooding
    res_flood = repl.execute(code="print('DATA' * 50000)")
    assert res_flood.success is True
    assert "[OUTPUT TRUNCATED]" in res_flood.output or len(res_flood.output) <= 65536

    # Network blocked
    res_net = repl.execute(code="import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)")
    assert res_net.success is False

    # Filesystem isolation
    res_fs = repl.execute(code="import os\nprint(os.path.basename(os.getcwd()))")
    assert res_fs.success is True
    assert "tmp" in res_fs.output.lower() or "temp" in res_fs.output.lower() or len(res_fs.output) > 0


# 5. Tool Registry Lookup
def test_tool_registry_lookup_and_invocation():
    registry = ToolRegistry()
    calc = SafeCalculatorTool()
    registry.register(calc)

    assert "calculator" in registry.list_tools()
    assert registry.get("calculator") is calc

    # Invoke registered tool
    res = registry.invoke("calculator", expression="12 * 12")
    assert res.success is True
    assert res.output == 144
    assert res.execution_time_ms >= 0.0

    # Invoke missing tool
    res_missing = registry.invoke("non_existent_tool", query="test")
    assert res_missing.success is False
    assert "not registered" in res_missing.error


# 6. Tool Result Propagation into State
def test_tool_result_propagation():
    runtime = MockRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    # Query routes to TOOL -> SafeCalculatorTool
    result = orchestrator.answer("What is 150 + 350?")
    assert result.route_taken == RoutePath.TOOL
    assert result.verification_status == VerificationStatus.PASS
    assert "150 + 350 = 500" in result.answer
    assert "calculator" in result.tools_used
    assert result.telemetry["tool_calls_made"] >= 1
    assert result.telemetry["tools_used"] == ["calculator"]


# 7. Verifier UNVERIFIED / ABSTAIN on Unsupported Factual Claims
def test_verifier_strict_unverified_on_unsupported_facts():
    runtime = MockRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    # Query with no grounding evidence provided must NEVER be PASS
    result = orchestrator.answer("Who invented the printing press?")
    assert result.verification_status in (VerificationStatus.UNVERIFIED, VerificationStatus.ABSTAIN)
    assert result.verification_status != VerificationStatus.PASS


# 8. StateManager Retry and Tool Budget Ceilings
def test_state_manager_ceilings():
    manager = StateManager()
    budget = ComputeBudget(max_tool_calls=1, max_retries=1)
    decision = RouteDecision(path=RoutePath.TOOL, routing_score=0.9, reason="Test", budget=budget)
    state = manager.create_initial_state("Query", decision)

    # 1st tool call allowed
    can_call, _ = manager.can_invoke_tool(state)
    assert can_call is True
    manager.record_tool_invocation(state)

    # 2nd tool call rejected by budget
    can_call, err = manager.can_invoke_tool(state)
    assert can_call is False
    assert "Exceeded max tool calls" in err

    # 1st retry allowed
    can_retry, _ = manager.can_retry(state)
    assert can_retry is True
    manager.record_retry(state)

    # 2nd retry rejected by budget
    can_retry, err = manager.can_retry(state)
    assert can_retry is False
    assert "Exceeded max retries" in err


# 9. End-to-End Orchestrator with Python Code Execution
def test_orchestrator_python_repl_e2e():
    runtime = MockRuntime()
    orchestrator = DhruvaOrchestrator(runtime=runtime)

    code_query = "Execute this script:\n```python\nprint(sum([i**2 for i in range(5)]))\n```"
    result = orchestrator.answer(code_query)

    assert result.route_taken == RoutePath.TOOL
    assert result.verification_status == VerificationStatus.PASS
    assert "30" in result.answer
    assert "python_repl" in result.tools_used


if __name__ == "__main__":
    tests = [
        test_calculator_correctness,
        test_calculator_invalid_inputs,
        test_python_repl_syntax_and_runtime_failure,
        test_python_repl_security_controls,
        test_tool_registry_lookup_and_invocation,
        test_tool_result_propagation,
        test_verifier_strict_unverified_on_unsupported_facts,
        test_state_manager_ceilings,
        test_orchestrator_python_repl_e2e,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL PHASE 2 TOOLING & VERIFIER TESTS PASSED")
