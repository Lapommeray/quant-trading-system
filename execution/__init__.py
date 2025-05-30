"""
Execution Algorithms Package

Provides both basic and advanced execution algorithms for the Lapommeray Quantum Trading System.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from execution import VWAPExecution, OptimalExecution
except ImportError:
    VWAPExecution = None
    OptimalExecution = None

try:
    from .advanced.smart_routing import InstitutionalOptimalExecution, AdvancedVWAPExecution
    if VWAPExecution and OptimalExecution:
        __all__ = ['VWAPExecution', 'OptimalExecution', 'InstitutionalOptimalExecution', 'AdvancedVWAPExecution']
    else:
        __all__ = ['InstitutionalOptimalExecution', 'AdvancedVWAPExecution']
except ImportError:
    if VWAPExecution and OptimalExecution:
        __all__ = ['VWAPExecution', 'OptimalExecution']
    else:
        __all__ = []
