"""
Aurora Gateway Package

This package contains the Aurora Gateway integration for QMP Overrider.
The Aurora Gateway provides advanced signal fusion and market intelligence
capabilities by integrating multiple data sources and analysis methods.

Modules:
- aurora_micro_node: Minimal micro-node to integrate Aurora Gateway
"""

from .aurora_micro_node import AuroraGateway, AuroraModule

__all__ = ['AuroraGateway', 'AuroraModule']
