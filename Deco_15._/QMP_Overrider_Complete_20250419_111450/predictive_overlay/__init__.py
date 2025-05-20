"""
Predictive Overlay Module for QMP Overrider

This module integrates advanced forecasting features into the QMP Overrider system:

Features:
1. Neural Forecasting Overlay using LSTM/Transformer
2. Synthetic Candle Projection (Ghost Candles)
3. Timeline Warp Plot for alternate path simulation
4. Future Zone Sensory Line based on gate and alignment consensus
5. Self-Correction Arcs with predictive error feedback loop
"""

from .neural_forecaster import NeuralForecaster
from .ghost_candle_projector import GhostCandleProjector
from .timeline_warp_plot import TimelineWarpPlot
from .future_zone_sensory import FutureZoneSensory
from .predictive_overlay_integration import PredictiveOverlaySystem

__all__ = [
    'NeuralForecaster',
    'GhostCandleProjector',
    'TimelineWarpPlot',
    'FutureZoneSensory',
    'PredictiveOverlaySystem'
]
