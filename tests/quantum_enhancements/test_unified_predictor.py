import pytest
from predictive_overlay.integrated_predictor import UnifiedPredictor

class TestUnifiedPredictor:
    @pytest.fixture
    def predictor(self):
        return UnifiedPredictor()
    
    def test_component_integration(self, predictor):
        assert 'neural' in predictor.components, "Missing neural forecaster"
        assert 'ghost' in predictor.components, "Missing ghost candle projector"
        assert 'temporal' in predictor.components, "Missing timeline warp plot"
    
    def test_projection_consistency(self, predictor):
        market_data = load_test_market_data()
        projection = predictor.predict(market_data)
        assert abs(projection['neural'] - projection['ghost']) < 0.05, "Component divergence detected"
        assert abs(projection['neural'] - projection['temporal']) < 0.05, "Component divergence detected"
