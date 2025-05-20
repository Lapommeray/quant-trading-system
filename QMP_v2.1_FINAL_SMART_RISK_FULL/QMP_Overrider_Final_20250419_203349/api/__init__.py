"""
API Package

This package provides API functionality for the QMP Overrider system.
It includes a FastAPI server for signal generation, order execution, and system monitoring.
"""

from .api_server import app

__all__ = ['app']
