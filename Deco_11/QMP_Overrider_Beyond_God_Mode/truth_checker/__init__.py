"""
Truth Checker Package

This package contains the Truth Checker integration for QMP Overrider.
The Truth Checker compares signals from Aurora, Phoenix, and QMP for
agreement or conflict, providing a higher-level decision mechanism.

Modules:
- truth_checker: Cross-validation engine for signal verification
"""

from .truth_checker import TruthChecker

__all__ = ['TruthChecker']
