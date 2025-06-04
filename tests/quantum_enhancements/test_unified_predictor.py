import pytest
from predictive_overlay.integrated_predictor import UnifiedPredictor
import pandas as pd
import numpy as np

def load_test_market_data():
    """Load test market data for unified predictor testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    data = {
        '5m': pd.DataFrame({
            'Open': np.random.normal(100, 2, 100),
            'High': np.random.normal(102, 2, 100),
            'Low': np.random.normal(98, 2, 100),
            'Close': np.random.normal(101, 2, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
    }
    return data

class TestUnifiedPredictor:
    @pytest.fixture
    def predictor(self):
        return UnifiedPredictor()
    
    def test_component_integration(self, predictor):
        assert 'neural' in predictor.components, "Missing neural forecaster"
        assert 'ghost' in predictor.components, "Missing ghost candle projector"
        assert 'temporal' in predictor.components, "Missing timeline warp plot"
    
    def test_projection_consistency(self, predictor):
        """Test that prediction components maintain consistency within tolerance."""
        market_data = load_test_market_data()
        projection = predictor.predict(market_data)
        
        assert projection['neural'] == pytest.approx(projection['ghost'], abs=0.5), "Component divergence detected"
        assert projection['neural'] == pytest.approx(projection['temporal'], abs=0.5), "Component divergence detected"
