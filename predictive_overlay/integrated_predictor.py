"""
Unified Predictor

Integrates multiple predictive components into a single unified predictor for the QMP Overrider system.
"""

from .neural_forecaster import NeuralForecaster
from .ghost_candle_projector import GhostCandleProjector
from .timeline_warp_plot import TimelineWarpPlot

class UnifiedPredictor:
    """
    Unified predictor that integrates multiple predictive components.
    """
    
    def __init__(self):
        """
        Initialize the unified predictor with its components.
        """
        self.components = {
            'neural': NeuralForecaster(),
            'ghost': GhostCandleProjector(),
            'temporal': TimelineWarpPlot()
        }
    
    def predict(self, market_data):
        """
        Generate predictions using all components.
        
        Parameters:
        - market_data: The market data to use for predictions
        
        Returns:
        - Dictionary containing predictions from all components
        """
        neural_prediction = self.components['neural'].forecast(market_data)
        ghost_prediction = self.components['ghost'].project(market_data)
        temporal_prediction = self.components['temporal'].analyze(market_data)
        
        return {
            'neural': neural_prediction,
            'ghost': ghost_prediction,
            'temporal': temporal_prediction
        }
