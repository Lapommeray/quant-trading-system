"""
Phoenix Protocol Package

This package contains the Phoenix Protocol integration for QMP Overrider.
The Phoenix Protocol provides market collapse detection, regime classification,
and anti-failure decision making capabilities.

Modules:
- phoenix_micro_node: Minimal integration interface for Phoenix Protocol
"""

from .phoenix_micro_node import PhoenixProtocol

__all__ = ['PhoenixProtocol']
