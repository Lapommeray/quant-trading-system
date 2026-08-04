"""Compatibility import for the strict generated-code validator."""

from autonomy.self_coding import SafeCodeValidator, ValidationReport

CodeValidator = SafeCodeValidator

__all__ = ["CodeValidator", "SafeCodeValidator", "ValidationReport"]
