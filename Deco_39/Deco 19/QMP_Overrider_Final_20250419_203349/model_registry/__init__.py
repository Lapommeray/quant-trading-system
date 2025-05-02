"""
Model Registry Package

This package contains the model registry components for the QMP Overrider system.
It includes the MLFlow integration for model versioning and tracking.
"""

from .mlflow_registry import MLFlowModelRegistry

__all__ = ['MLFlowModelRegistry']
