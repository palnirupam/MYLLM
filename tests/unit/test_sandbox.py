"""
tests/unit/test_sandbox.py
Verifies ProcessSandbox security controls: execution, timeouts, output limits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.tools.sandbox import ProcessSandbox, SandboxConfig


def test_sandbox_clean_execution():
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0))
    code = "print(sum([x * 2 for x in range(10)]))"
    res = sandbox.run_python_snippet(code)

    assert res.exit_code == 0
    assert res.timed_out is False
    assert res.stdout.strip() == "90"
    assert res.duration_ms > 0


def test_sandbox_timeout_enforcement():
    # Hard 1.0s timeout
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1.0))
    infinite_loop = "import time\nwhile True:\n    time.sleep(0.1)"
    res = sandbox.run_python_snippet(infinite_loop)

    assert res.timed_out is True
    assert res.exit_code == 124
    assert "timed out" in res.stderr


def test_sandbox_output_truncation():
    # 256 bytes output limit
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0, max_output_bytes=256))
    large_output = "print('A' * 1000)"
    res = sandbox.run_python_snippet(large_output)

    assert res.output_truncated is True
    assert "[OUTPUT TRUNCATED]" in res.stdout


if __name__ == "__main__":
    tests = [
        test_sandbox_clean_execution,
        test_sandbox_timeout_enforcement,
        test_sandbox_output_truncation,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL SANDBOX TESTS PASSED")
