import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QMP_GOD_MODE_v2_5_FINAL.ai.advanced_noise_filter import AdvancedNoiseFilter
from QMP_GOD_MODE_v2_5_FINAL.ai.market_glitch_detector import MarketGlitchDetector

class TestAdvancedNoiseFilter(unittest.TestCase):
    """Test the Advanced Noise Filter module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.noise_filter = AdvancedNoiseFilter()
        self.glitch_detector = MarketGlitchDetector()
        
        self.market_data = self._create_sample_market_data()
        self.noisy_market_data = self._create_noisy_market_data()
        
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
    
    def test_filter_noise_clean_data(self):
        """Test noise filtering on clean data"""
        result = self.noise_filter.filter_noise(self.market_data)
        
        self.assertIn("filtered", result)
        self.assertIn("initial_quality", result)
        self.assertIn("final_quality", result)
        
        self.assertGreaterEqual(result["initial_quality"], 0.9)
        self.assertGreaterEqual(result["final_quality"], 0.9)
    
    def test_filter_noise_noisy_data(self):
        """Test noise filtering on noisy data"""
        result = self.noise_filter.filter_noise(self.noisy_market_data)
        
        self.assertIn("filtered", result)
        self.assertIn("initial_quality", result)
        self.assertIn("final_quality", result)
        
        self.assertTrue(result["filtered"])
        self.assertGreater(result["final_quality"], result["initial_quality"])
        self.assertGreaterEqual(result["final_quality"], 0.8)
    
    def test_filter_stats(self):
        """Test filter statistics"""
        self.noise_filter.filter_noise(self.noisy_market_data)
        
        stats = self.noise_filter.get_filter_stats()
        
        self.assertIn("total_checks", stats)
        self.assertIn("noise_detected", stats)
        self.assertIn("noise_removed", stats)
        self.assertGreaterEqual(stats["total_checks"], 1)
    
    def test_glitch_detection_clean_data(self):
        """Test glitch detection on clean data"""
        result = self.glitch_detector.detect_glitches(self.market_data)
        
        self.assertIn("glitches_detected", result)
        
        self.assertFalse(result["glitches_detected"])
    
    def test_glitch_detection_with_discontinuity(self):
        """Test glitch detection with price discontinuity"""
        glitch_data = self._create_sample_market_data()
        ohlcv = list(glitch_data["ohlcv"])
        
        idx = 50
        gap_candle = list(ohlcv[idx])
        gap_candle[4] = gap_candle[4] * 1.15  # 15% gap
        ohlcv[idx] = tuple(gap_candle)
        
        glitch_data["ohlcv"] = ohlcv
        
        result = self.glitch_detector.detect_glitches(glitch_data)
        
        self.assertIn("glitches_detected", result)
        
        self.assertTrue(result["glitches_detected"])
        self.assertIn("glitches", result)
        
        found_discontinuity = False
        for glitch in result["glitches"]:
            if glitch["type"] == "price_discontinuity":
                found_discontinuity = True
                break
        
        self.assertTrue(found_discontinuity)
    
    def test_glitch_detection_stats(self):
        """Test glitch detection statistics"""
        glitch_data = self._create_sample_market_data()
        ohlcv = list(glitch_data["ohlcv"])
        
        idx = 50
        gap_candle = list(ohlcv[idx])
        gap_candle[4] = gap_candle[4] * 1.15  # 15% gap
        ohlcv[idx] = tuple(gap_candle)
        
        glitch_data["ohlcv"] = ohlcv
        
        self.glitch_detector.detect_glitches(glitch_data)
        
        stats = self.glitch_detector.get_detection_stats()
        
        self.assertIn("total_checks", stats)
        self.assertIn("glitches_detected", stats)
        self.assertGreaterEqual(stats["total_checks"], 1)
        self.assertGreaterEqual(stats["glitches_detected"], 1)

if __name__ == "__main__":
    unittest.main()
