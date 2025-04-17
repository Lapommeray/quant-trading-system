"""
Ritual Lock Package

This package contains the Ritual Lock integration for QMP Overrider.
The Ritual Lock prevents trades when cosmic/weather cycles signal instability,
providing an additional layer of protection against adverse market conditions.

Modules:
- ritual_lock: Spiritual filter for cosmic/weather cycle analysis
"""

from .ritual_lock import RitualLock

__all__ = ['RitualLock']
