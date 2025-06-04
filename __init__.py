"""
Quant Trading System - Advanced Quantum Finance Integration Trading System

A sophisticated quantitative trading system with 11-dimensional strategy perception,
quantum neural processing, and never-loss protection mechanisms.
"""

__version__ = "2.5.0"
__author__ = "Lapommeray"
__email__ = "lapommerayr@gmail.com"

from core import (
    HestonVolatility,
    ML_RSI, 
    OrderFlowImbalance,
    RegimeDetector
)

try:
    from core import QMPOverrider
except ImportError:
    QMPOverrider = None

try:
    from core import QMPAIAgent
except ImportError:
    QMPAIAgent = None

try:
    from core import QuantumOrchestrator
except ImportError:
    QuantumOrchestrator = None

try:
    from core import OversoulDirector
except ImportError:
    OversoulDirector = None

__all__ = [
    'HestonVolatility',
    'ML_RSI', 
    'OrderFlowImbalance',
    'RegimeDetector'
]

if QMPOverrider is not None:
    __all__.append('QMPOverrider')
if QMPAIAgent is not None:
    __all__.append('QMPAIAgent')
if QuantumOrchestrator is not None:
    __all__.append('QuantumOrchestrator')
if OversoulDirector is not None:
    __all__.append('OversoulDirector')
