"""Compatibility import for the resource-limited test sandbox."""

from autonomy.sandbox import SandboxExecutor, SandboxPolicy, SandboxResult

SafeExecutor = SandboxExecutor

__all__ = ["SafeExecutor", "SandboxExecutor", "SandboxPolicy", "SandboxResult"]
