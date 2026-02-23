"""Predictive overlay package.

Keep imports lazy to avoid import-time failures when optional components
(e.g., qiskit-backed forecasters) are unavailable in minimal test environments.
"""

__all__ = [
    "NeuralForecaster",
    "GhostCandleProjector",
    "TimelineWarpPlot",
    "FutureZoneSensory",
    "PredictiveOverlaySystem",
]


def __getattr__(name):
    if name == "NeuralForecaster":
        from .neural_forecaster import NeuralForecaster

        return NeuralForecaster
    if name == "GhostCandleProjector":
        from .ghost_candle_projector import GhostCandleProjector

        return GhostCandleProjector
    if name == "TimelineWarpPlot":
        from .timeline_warp_plot import TimelineWarpPlot

        return TimelineWarpPlot
    if name == "FutureZoneSensory":
        from .future_zone_sensory import FutureZoneSensory

        return FutureZoneSensory
    if name == "PredictiveOverlaySystem":
        from .predictive_overlay_integration import PredictiveOverlaySystem

        return PredictiveOverlaySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
