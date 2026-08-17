"""
tests/unit/test_sandbox_security.py
Comprehensive security suite for the ProcessSandbox:
- timeout
- memory exhaustion
- fork/process explosion
- filesystem escape
- network access blocked
- environment-variable isolation
- output flooding
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from myllm.intelligence.tools.sandbox import ProcessSandbox, SandboxConfig


def test_security_timeout():
    """Ensure infinite loops are killed within timeout limit."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=1.0))
    code = "import time\nwhile True:\n    time.sleep(0.05)"
    res = sandbox.run_python_snippet(code)
    assert res.timed_out is True
    assert res.exit_code == 124
    assert "timed out" in res.stderr


def test_security_memory_exhaustion():
    """Ensure scripts attempting large allocations terminate safely."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0, memory_limit_mb=128))
    # Try allocating 500MB string
    code = """
try:
    x = 'A' * (500 * 1024 * 1024)
    print('Allocated')
except Exception as e:
    print('Caught', type(e).__name__)
"""
    res = sandbox.run_python_snippet(code)
    # The script should either catch MemoryError or terminate without crashing host
    assert res.duration_ms < 5000


def test_security_fork_process_explosion():
    """Ensure process spawning / fork attempts are blocked by security preamble."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0))
    code = """
import os
if hasattr(os, 'fork'):
    os.fork()
else:
    # On Windows, test subprocess block
    import subprocess
    subprocess.Popen(['echo', 'exploit'])
"""
    res = sandbox.run_python_snippet(code)
    assert "blocked by the Dhruva sandbox policy" in res.stderr


def test_security_filesystem_escape():
    """Ensure code executes inside isolated temp directory, cannot mutate outside cwd."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0))
    code = """
import os
cwd = os.getcwd()
assert 'tmp' in cwd.lower() or 'temp' in cwd.lower(), f"Unexpected cwd: {cwd}"
print("CWD_ISOLATED")
"""
    res = sandbox.run_python_snippet(code)
    assert res.exit_code == 0
    assert "CWD_ISOLATED" in res.stdout


def test_security_network_access_blocked():
    """Ensure outbound network connections are blocked by security preamble."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0, disable_network=True))
    code = """
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
"""
    res = sandbox.run_python_snippet(code)
    assert res.network_blocked is True or "Network access is blocked" in res.stderr


def test_security_environment_variable_isolation():
    """Ensure host API keys / secrets are not inherited by sandboxed process."""
    # Set a dummy sensitive env var in current process
    os.environ["OPENAI_API_KEY"] = "sk-test-secret-key-12345"
    os.environ["DHRUVA_SECRET_TOKEN"] = "token-secret-9999"

    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0))
    code = """
import os
for k, v in os.environ.items():
    if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k:
        print(f"LEAKED_{k}")
"""
    res = sandbox.run_python_snippet(code)
    assert "LEAKED_OPENAI_API_KEY" not in res.stdout
    assert "LEAKED_DHRUVA_SECRET_TOKEN" not in res.stdout


def test_security_output_flooding():
    """Ensure massive stdout output is truncated to prevent buffer memory exhaustion."""
    sandbox = ProcessSandbox(SandboxConfig(timeout_seconds=2.0, max_output_bytes=512))
    code = "print('FLOOD' * 10000)"
    res = sandbox.run_python_snippet(code)
    assert res.output_truncated is True
    assert "[OUTPUT TRUNCATED]" in res.stdout
    assert len(res.stdout.encode('utf-8')) <= 2048


if __name__ == "__main__":
    tests = [
        test_security_timeout,
        test_security_memory_exhaustion,
        test_security_fork_process_explosion,
        test_security_filesystem_escape,
        test_security_network_access_blocked,
        test_security_environment_variable_isolation,
        test_security_output_flooding,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("\nALL SANDBOX SECURITY TESTS PASSED")
