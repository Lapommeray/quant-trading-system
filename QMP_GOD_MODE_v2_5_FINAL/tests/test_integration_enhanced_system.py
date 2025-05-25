"""
Integration test for the enhanced quant-trading-system with super high confidence requirements.
Tests the integration of all enhanced modules including AdvancedNoiseFilter, MarketGlitchDetector,
and ImperceptiblePatternDetector with the OverSoulDirector.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.advanced_noise_filter import AdvancedNoiseFilter
from ai.market_glitch_detector import MarketGlitchDetector
from ai.imperceptible_pattern_detector import ImperceptiblePatternDetector
from core.oversoul_director import OverSoulDirector
from core.anti_loss_guardian import AntiLossGuardian

class MockQuantConnectAlgorithm:
    """Mock QuantConnect Algorithm for testing"""
    
    def __init__(self):
        self.portfolio = {"value": 100000.0}
        self.positions = {}
        self.orders = []
        self.securities = {}
        self.insights = []
        self.debug_messages = []
        self.error_messages = []
        self.warning_messages = []
        
    def Debug(self, message):
        self.debug_messages.append(message)
        
    def Error(self, message):
        self.error_messages.append(message)
        
    def Log(self, message):
        self.debug_messages.append(message)
        
    def SetHoldings(self, symbol, percentage):
        self.positions[symbol] = percentage
        return True
        
    def Liquidate(self):
        self.positions = {}
        return True
        
    def GetPortfolioValue(self):
        return self.portfolio["value"]
        
    def EmitInsights(self, insights):
        self.insights.extend(insights)

class TestIntegrationEnhancedSystem(unittest.TestCase):
    """Test the integration of all enhanced modules in the system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.algorithm = MockQuantConnectAlgorithm()
        self.oversoul = OverSoulDirector(self.algorithm)
        self.anti_loss = AntiLossGuardian(self.algorithm)
        self.noise_filter = AdvancedNoiseFilter()
        self.glitch_detector = MarketGlitchDetector()
        self.pattern_detector = ImperceptiblePatternDetector()
        
        self.market_data = self._create_sample_market_data()
        self.noisy_market_data = self._create_noisy_market_data()
        self.glitch_market_data = self._create_glitch_market_data()
        
    def _create_sample_market_data(self):
        """Create sample market data for testing"""
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(100)]
        timestamps.reverse()
        
        base_price = 100.0
        prices = []
        for i in range(100):
            price = base_price + i * 0.1 + np.random.normal(0, 0.1)
            prices.append(price)
        
        ohlcv = []
        for i, ts in enumerate(timestamps):
            price = prices[i]
            candle = (
                int(ts.timestamp() * 1000),  # timestamp in milliseconds
                price - 0.1,                 # open
                price + 0.2,                 # high
                price - 0.2,                 # low
                price,                       # close
                1000 + np.random.normal(0, 100)  # volume
            )
            ohlcv.append(candle)
        
        return {
            "ohlcv": ohlcv,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
    
    def _create_noisy_market_data(self):
        """Create noisy market data for testing"""
        noisy_data = self._create_sample_market_data()
        ohlcv = list(noisy_data["ohlcv"])
        
        for i in range(5):
            idx = np.random.randint(10, 90)
            spike_candle = list(ohlcv[idx])
            spike_candle[4] = spike_candle[4] * (1 + np.random.choice([-1, 1]) * 0.1)  # 10% spike
            ohlcv[idx] = tuple(spike_candle)
        
        for i in range(10):
            idx = np.random.randint(10, 90)
            low_vol_candle = list(ohlcv[idx])
            low_vol_candle[5] = low_vol_candle[5] * 0.1  # 90% volume reduction
            ohlcv[idx] = tuple(low_vol_candle)
        
        for i in range(20):
            idx = np.random.randint(10, 90)
            hf_candle = list(ohlcv[idx])
            hf_candle[4] = hf_candle[4] * (1 + np.random.normal(0, 0.02))  # Small random noise
            ohlcv[idx] = tuple(hf_candle)
        
        noisy_data["ohlcv"] = ohlcv
        return noisy_data
    
    def _create_glitch_market_data(self):
        """Create market data with glitches for testing"""
        glitch_data = self._create_sample_market_data()
        ohlcv = list(glitch_data["ohlcv"])
        
        idx = 50
        gap_candle = list(ohlcv[idx])
        gap_candle[4] = gap_candle[4] * 1.15  # 15% gap
        ohlcv[idx] = tuple(gap_candle)
        
        for i in range(3):
            idx = 60 + i
            low_vol_candle = list(ohlcv[idx])
            low_vol_candle[5] = low_vol_candle[5] * 0.05  # 95% volume reduction
            ohlcv[idx] = tuple(low_vol_candle)
        
        glitch_data["ohlcv"] = ohlcv
        
        glitch_data["order_book"] = {
            "bids": [(99.5, 5000), (99.0, 3000), (98.5, 2000)],
            "asks": [(100.0, 500), (100.5, 300), (101.0, 200)]
        }
        
        return glitch_data
    
    def test_noise_filtering_integration(self):
        """Test integration of noise filtering with the system"""
        filter_result = self.noise_filter.filter_noise(self.noisy_market_data)
        
        self.assertTrue(filter_result["filtered"])
        self.assertGreater(filter_result["final_quality"], filter_result["initial_quality"])
        self.assertGreaterEqual(filter_result["final_quality"], 0.8)
        
        pattern_data = filter_result["data"].copy()
        ohlcv = list(pattern_data["ohlcv"])
        
        for i in range(5):
            idx = len(ohlcv) - 10 + i
            if idx < len(ohlcv):
                candle = list(ohlcv[idx])
                candle[4] = candle[4] * (1 + 0.02)  # Up
                candle[1] = candle[4] * 0.99  # Open lower than close (bullish)
                ohlcv[idx] = tuple(candle)
        
        idx = len(ohlcv) - 5
        if idx < len(ohlcv):
            candle = list(ohlcv[idx])
            candle[4] = candle[4] * 0.95  # Down 5%
            candle[1] = candle[4] * 1.02  # Open higher than close (bearish)
            candle[3] = candle[4] * 0.98  # Low below close
            ohlcv[idx] = tuple(candle)
        
        for i in range(3):
            idx = len(ohlcv) - 3 + i
            if idx < len(ohlcv):
                candle = list(ohlcv[idx])
                candle[5] = candle[5] * 3.0  # Triple volume
                ohlcv[idx] = tuple(candle)
        
        pattern_data["ohlcv"] = ohlcv
        
        pattern_data["order_book"] = {
            "bids": [(99.5, 5000), (99.0, 3000), (98.5, 2000)],
            "asks": [(100.0, 500), (100.5, 300), (101.0, 200)]
        }
        
        pattern_result = self.pattern_detector.detect_patterns(pattern_data)
        
        # Adjust assertions to match the actual response format from ImperceptiblePatternDetector
        self.assertTrue(pattern_result.get("detected", False))
        self.assertIn("confidence", pattern_result)
        self.assertGreaterEqual(pattern_result.get("confidence", 0), 0.7)
    
    def test_glitch_detection_integration(self):
        """Test integration of glitch detection with the system"""
        glitch_result = self.glitch_detector.detect_glitches(self.glitch_market_data)
        
        self.assertTrue(glitch_result["glitches_detected"])
        self.assertGreaterEqual(glitch_result["confidence"], 0.85)
        self.assertIn("signal", glitch_result)
        
        pattern_result = self.pattern_detector.detect_patterns(self.glitch_market_data)
        
        self.assertTrue(pattern_result.get("detected", False))
        self.assertIn("confidence", pattern_result)
    
    def test_anti_loss_integration(self):
        """Test integration of anti-loss guardian with the system"""
        anti_loss_result = self.anti_loss.check_anti_loss_conditions(100000.0, {"SPY": 0.05, "QQQ": 0.05})
        
        self.assertTrue(anti_loss_result["allowed"])
        self.assertEqual(anti_loss_result["action"], "none")
        
        anti_loss_result = self.anti_loss.check_anti_loss_conditions(99000.0, {"SPY": 0.4, "QQQ": 0.1})
        
        self.assertFalse(anti_loss_result["allowed"])
        self.assertIn(anti_loss_result["action"], ["reduce_concentration", "reduce_position", "halt_new_trades"])
    
    def test_oversoul_integration(self):
        """Test integration of oversoul director with the system"""
        gate_results = {
            "emotion_dna": True,
            "fractal_resonance": True,
            "intention_decoder": True,
            "timeline_fork": True,
            "astro_sync": True,
            "black_swan_protector": True,
            "quantum_tremor": True,
            "future_shadow": True,
            "market_thought": True,
            "reality_matrix": True,
            "human_lag": True,
            "invisible_data": True,
            "meta_adaptive": True,
            "quantum_sentiment": True,
            "imperceptible_pattern": True,
            "market_glitch_detector": True,
            "advanced_noise_filter": True
        }
        
        oversoul_result = self.oversoul.evaluate_state(gate_results)
        
        self.assertEqual(oversoul_result["action"], "EXECUTE")
        self.assertIn("modules", oversoul_result)
        self.assertTrue(oversoul_result["modules"]["advanced_noise_filter"])
        self.assertTrue(oversoul_result["modules"]["market_glitch_detector"])
        self.assertTrue(oversoul_result["modules"]["imperceptible_pattern"])
    
    def test_full_integration_cycle(self):
        """Test full integration cycle with all components"""
        filter_result = self.noise_filter.filter_noise(self.noisy_market_data)
        filtered_data = filter_result["data"]
        
        glitch_result = self.glitch_detector.detect_glitches(filtered_data)
        
        pattern_result = self.pattern_detector.detect_patterns(filtered_data)
        
        anti_loss_result = self.anti_loss.check_anti_loss_conditions(100000.0, {"SPY": 0.2})
        
        gate_results = {
            "emotion_dna": True,
            "fractal_resonance": True,
            "intention_decoder": True,
            "timeline_fork": True,
            "astro_sync": True,
            "black_swan_protector": True,
            "quantum_tremor": True,
            "future_shadow": True,
            "market_thought": True,
            "reality_matrix": True,
            "human_lag": True,
            "invisible_data": True,
            "meta_adaptive": True,
            "quantum_sentiment": True,
            "imperceptible_pattern": pattern_result.get("detected", False),
            "market_glitch_detector": glitch_result["glitches_detected"],
            "advanced_noise_filter": filter_result.get("high_quality", filter_result.get("final_quality", 0) >= 0.95)
        }
        
        oversoul_result = self.oversoul.evaluate_state(gate_results)
        
        self.assertGreaterEqual(filter_result["final_quality"], 0.8)
        self.assertIn("confidence", pattern_result)
        self.assertIn("action", oversoul_result)
        
        if (pattern_result.get("detected", False) and 
            filter_result["high_quality"] and 
            anti_loss_result["allowed"]):
            self.assertEqual(oversoul_result["action"], "EXECUTE")
        
    def test_super_high_confidence_requirements(self):
        """Test that the system enforces super high confidence requirements"""
        self.assertGreaterEqual(self.noise_filter.min_quality_threshold, 0.95)
        
        self.assertGreaterEqual(self.glitch_detector.confidence_threshold, 0.95)
        
        self.assertGreaterEqual(self.pattern_detector.confidence_threshold, 0.85)
        
        self.assertEqual(self.anti_loss.max_consecutive_losses, 1)
        self.assertLessEqual(self.anti_loss.protection_levels["level_1"]["drawdown"], 0.01)
        
        low_conf_pattern = {
            "patterns_detected": True,
            "confidence": 0.7,
            "high_confidence": False
        }
        
        gate_results = {
            "emotion_dna": True,
            "fractal_resonance": True,
            "intention_decoder": True,
            "timeline_fork": True,
            "astro_sync": True,
            "black_swan_protector": True,
            "quantum_tremor": True,
            "future_shadow": True,
            "market_thought": True,
            "reality_matrix": True,
            "human_lag": True,
            "invisible_data": True,
            "meta_adaptive": True,
            "quantum_sentiment": True,
            "imperceptible_pattern": low_conf_pattern["high_confidence"],
            "market_glitch_detector": True,
            "advanced_noise_filter": True
        }
        
        oversoul_result = self.oversoul.evaluate_state(gate_results)
        self.assertNotEqual(oversoul_result["action"], "EXECUTE")

if __name__ == "__main__":
    unittest.main()
