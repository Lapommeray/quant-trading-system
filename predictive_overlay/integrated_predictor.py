"""
Unified Predictor

Integrates multiple predictive components into a single unified predictor for the QMP Overrider system.
"""

from .neural_forecaster import NeuralForecaster
from .ghost_candle_projector import GhostCandleProjector
from .timeline_warp_plot import TimelineWarpPlot
from datetime import datetime

class MockAlgorithm:
    """Mock algorithm for testing purposes"""
    def __init__(self):
        self.Time = datetime.now()

class UnifiedPredictor:
    """
    Unified predictor that integrates multiple predictive components.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the unified predictor with its components.
        
        Parameters:
        - algorithm: QuantConnect algorithm instance (optional, uses mock for testing)
        """
        if algorithm is None:
            algorithm = MockAlgorithm()
            
        self.components = {
            'neural': NeuralForecaster(algorithm),
            'ghost': GhostCandleProjector(algorithm),
            'temporal': TimelineWarpPlot(algorithm)
        }
    
    def predict(self, market_data):
        """
        Generate predictions using all components.
        
        Parameters:
        - market_data: The market data to use for predictions
        
        Returns:
        - Dictionary containing predictions from all components
        """
        neural_result = self.components['neural'].forecast('SPY', market_data)
        ghost_result = self.components['ghost'].project_ghost_candles('SPY', market_data)
        temporal_result = self.components['temporal'].generate_timelines('SPY', market_data)
        
        neural_value = neural_result.get('latest_price', 100.0) if isinstance(neural_result, dict) else 100.0
        ghost_value = ghost_result[0]['Close'] if ghost_result and len(ghost_result) > 0 else neural_value
        temporal_value = temporal_result.get('latest_price', neural_value) if isinstance(temporal_result, dict) else neural_value
        
        return {
            'neural': neural_value,
            'ghost': ghost_value,
            'temporal': temporal_value
        }
