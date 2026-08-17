"""
myllm.intelligence.tools.sandbox — Security sandbox baseline for code and tool execution.

WARNING:
This sandbox provides process-level isolation, strict execution timeouts, output size
limits, network access blocks, and sanitized environment variables using subprocess execution.
It is an application-level isolation boundary, NOT a kernel-level hardware virtualization
sandbox (e.g. gVisor, Firecracker microVM, or rootless OCI container).
In production multi-tenant environments with untrusted adversarial code, this process should
be wrapped inside a dedicated container or microVM sandbox.
"""

import sys
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field


@dataclass
class SandboxConfig:
    timeout_seconds: float = 3.0
    max_output_bytes: int = 65536  # 64 KB limit
    disable_network: bool = True
    memory_limit_mb: int = 256
    max_processes: int = 1
    whitelisted_env_vars: Set[str] = field(default_factory=lambda: {
        "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TZ", "LANG", "LC_ALL"
    })


@dataclass
class SandboxExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    timed_out: bool = False
    output_truncated: bool = False
    network_blocked: bool = False


# Security preamble injected into isolated script execution
SECURITY_PREAMBLE = """# Dhruva Sandbox Security Preamble
import sys

# 1. Disable network access if configured
def _block_network():
    import socket
    def _disabled_socket(*args, **kwargs):
        raise PermissionError("Network access is blocked by the Dhruva sandbox policy.")
    socket.socket = _disabled_socket
    socket.create_connection = _disabled_socket
    if hasattr(socket, 'getaddrinfo'):
        socket.getaddrinfo = _disabled_socket

_block_network()

# 2. Block process spawning / fork explosion
def _block_subprocesses():
    import os
    if hasattr(os, 'fork'):
        def _disabled_fork(*args, **kwargs):
            raise PermissionError("Process spawning (fork) is blocked by the Dhruva sandbox policy.")
        os.fork = _disabled_fork
    try:
        import subprocess
        def _disabled_popen(*args, **kwargs):
            raise PermissionError("Subprocess spawning is blocked by the Dhruva sandbox policy.")
        subprocess.Popen = _disabled_popen
    except Exception:
        pass

_block_subprocesses()

# 3. Soft memory guard
try:
    import resource
    # Set virtual memory limit on POSIX systems (256 MB)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
except (ImportError, AttributeError, ValueError, Exception):
    pass
# End of Preamble
"""


class ProcessSandbox:
    """
    Subprocess sandbox enforcing execution boundaries, timeouts, environment isolation,
    and resource limits.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()

    def _build_sanitized_env(self, sandbox_dir: str) -> Dict[str, str]:
        """
        Constructs a minimal, sanitized environment containing NO API keys or sensitive secrets.
        """
        sanitized = {}
        for k, v in os.environ.items():
            k_upper = k.upper()
            # Explicitly exclude all secret-bearing environment keys
            if any(secret_term in k_upper for secret_term in ["KEY", "TOKEN", "SECRET", "AUTH", "PASS", "CRED", "API"]):
                continue
            if k_upper in self.config.whitelisted_env_vars:
                sanitized[k] = v

        # Set strict Python runtime isolation flags
        sanitized["PYTHONDONTWRITEBYTECODE"] = "1"
        sanitized["PYTHONUNBUFFERED"] = "1"
        sanitized["PYTHONPATH"] = sandbox_dir
        sanitized["TEMP"] = sandbox_dir
        sanitized["TMP"] = sandbox_dir
        return sanitized

    def run_python_snippet(self, code_str: str) -> SandboxExecutionResult:
        """
        Executes a Python snippet with timeout, filesystem isolation, network disablement,
        and sanitized environment.
        """
        start_time = time.time()

        with tempfile.TemporaryDirectory() as sandbox_dir:
            script_path = Path(sandbox_dir) / "snippet.py"
            full_script = SECURITY_PREAMBLE + "\n" + code_str

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            env = self._build_sanitized_env(sandbox_dir)

            # shell=False is strictly enforced
            cmd = [sys.executable, "-I", str(script_path)]

            timed_out = False
            stdout_str = ""
            stderr_str = ""
            exit_code = -1
            truncated = False
            network_blocked = False

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=sandbox_dir,
                    env=env,
                    shell=False,  # Explicitly False
                    text=True,
                )

                try:
                    stdout, stderr = proc.communicate(timeout=self.config.timeout_seconds)
                    exit_code = proc.returncode

                    if "Network access is blocked" in stderr:
                        network_blocked = True

                    if len(stdout.encode("utf-8")) > self.config.max_output_bytes:
                        stdout = stdout[:self.config.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
                        truncated = True

                    if len(stderr.encode("utf-8")) > self.config.max_output_bytes:
                        stderr = stderr[:self.config.max_output_bytes] + "\n[STDERR TRUNCATED]"
                        truncated = True

                    stdout_str = stdout
                    stderr_str = stderr

                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    stdout_str = stdout
                    stderr_str = f"Execution timed out after {self.config.timeout_seconds}s"
                    exit_code = 124

            except Exception as e:
                stderr_str = f"Sandbox execution failure: {str(e)}"
                exit_code = 1

            duration_ms = (time.time() - start_time) * 1000.0

            return SandboxExecutionResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                duration_ms=duration_ms,
                timed_out=timed_out,
                output_truncated=truncated,
                network_blocked=network_blocked,
            )
