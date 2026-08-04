"""Resource-limited subprocess runner for generated test suites.

Generated code is validated before this runner is invoked.  The subprocess has
no inherited environment secrets, a timeout, and best-effort CPU/address-space
limits on Unix.  It is a test sandbox, not a claim of perfect OS isolation;
production deployments should run it in a container/VM as well.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: float = 5.0
    max_memory_mb: int = 500
    max_cpu_seconds: int = 5


@dataclass
class SandboxResult:
    success: bool
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


class SandboxExecutor:
    def __init__(self, policy: Optional[SandboxPolicy] = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(self, command: Iterable[str], *, cwd: str | Path) -> SandboxResult:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process_group = os.name != "nt"
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=process_group,
                preexec_fn=self._unix_limits if process_group else None,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=self.policy.timeout_seconds
                )
            except subprocess.TimeoutExpired:
                if process_group:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        process.kill()
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                return SandboxResult(
                    False,
                    process.returncode,
                    stdout[-20_000:],
                    stderr[-20_000:],
                    "timeout",
                )
            return SandboxResult(
                process.returncode == 0,
                process.returncode,
                (stdout or "")[-20_000:],
                (stderr or "")[-20_000:],
                "" if process.returncode == 0 else "sandbox command failed",
            )
        except OSError as exc:
            return SandboxResult(False, error=str(exc))

    def _unix_limits(self) -> None:
        try:
            import resource

            memory = int(self.policy.max_memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.policy.max_cpu_seconds, self.policy.max_cpu_seconds),
            )
        except (ImportError, OSError, ValueError):
            # The parent timeout remains active on platforms without rlimits.
            return


__all__ = ["SandboxExecutor", "SandboxPolicy", "SandboxResult"]
