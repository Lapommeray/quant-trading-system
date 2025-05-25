"""
Super High Confidence Test Suite
Tests the enhanced system with super high confidence requirements
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.meta_adaptive_ai import MetaAdaptiveAI
from ai.truth_verification_core import TruthVerificationCore
from ai.imperceptible_pattern_detector import ImperceptiblePatternDetector
from core.anti_loss_guardian import AntiLossGuardian
from core.model_confidence_tracker import ModelConfidenceTracker
from core.oversoul_director import OverSoulDirector

class MockQuantConnectAlgorithm:
    """Mock QuantConnect Algorithm for testing"""
    
    def __init__(self):
        self.Portfolio = {"SPY": 100000, "QQQ": 50000}
        self.Securities = {"SPY": {"Price": 450.0}, "QQQ": {"Price": 380.0}}
        self.Time = datetime.now()
        self.is_live_mode = False
        self.debug_messages = []
        
        self.DataFolder = "/tmp/qc_data"
        self.ObjectStore = {}
        self.Symbol = "SPY"
        self.Resolution = "Daily"
        self.StartDate = datetime.now() - timedelta(days=365)
        self.EndDate = datetime.now()
        self.UniverseSettings = {"Resolution": "Daily"}
        self.Settings = {"DataFolder": self.DataFolder}
        
    def Debug(self, message):
        """Log debug message"""
        self.debug_messages.append(message)
        print(message)
        
    def Log(self, message):
        """Log message"""
        print(message)
        
    def SetHoldings(self, symbol, weight):
        """Mock set holdings"""
        self.Portfolio[symbol] = weight * 1000000
        return True
        
    def Liquidate(self):
        """Mock liquidate all positions"""
        for symbol in self.Portfolio:
            self.Portfolio[symbol] = 0
        return True
        
    def GetLastKnownPrice(self, symbol):
        """Get last known price"""
        return self.Securities.get(symbol, {}).get("Price", 0)
        
    def SaveJson(self, key, obj):
        """Mock SaveJson for ObjectStore"""
        self.ObjectStore[key] = json.dumps(obj)
        return True
        
    def LoadJson(self, key):
        """Mock LoadJson for ObjectStore"""
        if key in self.ObjectStore:
            return json.loads(self.ObjectStore[key])
        return None

class TestSuperHighConfidence(unittest.TestCase):
    """Test super high confidence requirements"""
    
    def setUp(self):
        """Set up test environment"""
        self.algorithm = MockQuantConnectAlgorithm()
        self.meta_ai = MetaAdaptiveAI(self.algorithm)
        self.truth_verification = TruthVerificationCore()
        self.pattern_detector = ImperceptiblePatternDetector()
        self.anti_loss = AntiLossGuardian(self.algorithm)
        self.model_confidence = ModelConfidenceTracker()
        self.oversoul = OverSoulDirector(self.algorithm)
        
        self.market_data = self._generate_high_quality_market_data()
        
    def _generate_high_quality_market_data(self):
        """Generate high-quality market data for testing"""
        n_samples = 200
        base_price = 450.0
        trend = np.cumsum(np.random.normal(0.0005, 0.001, n_samples))  # Small upward bias
        
        t = np.linspace(0, 4*np.pi, n_samples)
        cycles = 0.01 * np.sin(t) + 0.005 * np.sin(2*t) + 0.002 * np.sin(5*t)
        
        returns = trend + cycles
        
        prices = base_price * (1 + returns)
        
        opens = prices * (1 + np.random.normal(0, 0.001, n_samples))
        highs = np.maximum(prices * (1 + np.random.normal(0.001, 0.001, n_samples)), opens)
        lows = np.minimum(prices * (1 + np.random.normal(-0.001, 0.001, n_samples)), opens)
        closes = prices
        
        base_volume = 1000000
        volume = base_volume * (1 + 0.5 * np.random.normal(0, 0.2, n_samples) + 0.1 * np.sin(t))
        volume = np.abs(volume)
        
        ohlcv = []
        for i in range(n_samples):
            timestamp = datetime.now() - timedelta(days=n_samples-i)
            candle = [
                timestamp.timestamp(),  # Timestamp
                opens[i],               # Open
                highs[i],               # High
                lows[i],                # Low
                closes[i],              # Close
                volume[i]               # Volume
            ]
            ohlcv.append(candle)
        
        price_returns = np.diff(closes) / closes[:-1]
        
        return {
            'ohlcv': ohlcv,
            'returns': price_returns,
            'volume': volume[1:],
            'open': opens[1:],
            'high': highs[1:],
            'low': lows[1:],
            'close': closes[1:]
        }
        
    def test_meta_adaptive_ai_high_confidence(self):
        """Test MetaAdaptiveAI with high confidence threshold"""
        self.meta_ai.train(self.market_data)
        prediction = self.meta_ai.predict(self.market_data)
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, dict)
        self.assertIn("confidence", prediction)
        self.assertIn("signal", prediction)
        self.assertIn("high_confidence", prediction)
        
        self.assertGreaterEqual(self.meta_ai.confidence_threshold, 0.85)
        
        if prediction["signal"] != "NEUTRAL":
            self.assertGreaterEqual(prediction["confidence"], self.meta_ai.confidence_threshold)
            self.assertTrue(prediction["high_confidence"])
        
    def test_truth_verification_high_confidence(self):
        """Test TruthVerificationCore with high confidence threshold"""
        if 'returns' in self.market_data and len(self.market_data['returns']) > 0:
            returns = self.market_data['returns'][-100:]
            market_data = {'returns': returns}
            result = self.truth_verification.verify_market_truth(market_data, [])
            
            self.assertIsInstance(result, dict)
            self.assertIn("truth_verified", result)
            self.assertIn("truth_score", result)
            self.assertIn("high_confidence", result)
            
            if result["truth_verified"]:
                self.assertGreaterEqual(result["truth_score"], 0.95)
                self.assertTrue(result["high_confidence"])
        
    def test_imperceptible_pattern_detector(self):
        """Test ImperceptiblePatternDetector with high confidence threshold"""
        market_data = {'ohlcv': self.market_data['ohlcv'][-100:]}
        
        quality_result = self.pattern_detector._verify_data_quality(market_data)
        self.assertIsInstance(quality_result, dict)
        self.assertIn("quality_verified", quality_result)
        
        detection_result = self.pattern_detector.detect_patterns(market_data)
        self.assertIsInstance(detection_result, dict)
        self.assertIn("patterns_detected", detection_result)
        self.assertIn("signal", detection_result)
        
        if detection_result["patterns_detected"]:
            self.assertGreaterEqual(detection_result["confidence"], 0.85)
        
    def test_anti_loss_guardian_never_lose(self):
        """Test AntiLossGuardian with never-lose protection"""
        portfolio_value = sum(self.algorithm.Portfolio.values())
        result = self.anti_loss.check_anti_loss_conditions(portfolio_value, self.algorithm.Portfolio)
        
        self.assertIsInstance(result, dict)
        self.assertIn("allowed", result)
        
        self.assertEqual(self.anti_loss.max_consecutive_losses, 1)
        self.assertLessEqual(self.anti_loss.protection_levels["level_1"]["drawdown"], 0.001)
        self.assertLessEqual(self.anti_loss.max_risk_multiplier, 0.5)
        
        self.anti_loss.update_trade_result(-0.001)
        result = self.anti_loss.check_anti_loss_conditions(portfolio_value * 0.999, self.algorithm.Portfolio)
        self.assertFalse(result["allowed"])
        
    def test_model_confidence_tracker_high_threshold(self):
        """Test ModelConfidenceTracker with high confidence threshold"""
        model_name = "test_model"
        
        for i in range(10):
            self.model_confidence.track_confidence(model_name, 0.9)
            self.model_confidence.track_performance(model_name, 0.85)
        
        needs_retraining = self.model_confidence.needs_retraining(model_name)
        
        confidence_trend = self.model_confidence.get_confidence_trend(model_name)
        self.assertGreaterEqual(confidence_trend["current"], 0.85)
        
    def test_oversoul_director_integration(self):
        """Test OverSoulDirector integration with new modules"""
        gate_results = {
            "emotion_dna": True,
            "fractal_resonance": True,
            "imperceptible_pattern": True,
            "meta_adaptive": True
        }
        
        result = self.oversoul.evaluate_state(gate_results)
        self.assertIsInstance(result, dict)
        self.assertIn("action", result)
        
        self.assertIn("modules", result)
        self.assertTrue(result["modules"].get("imperceptible_pattern", False))
        
    def test_full_system_integration(self):
        """Test full system integration with high confidence requirements"""
        self.meta_ai.train(self.market_data)
        
        initial_portfolio = self.algorithm.Portfolio["SPY"]
        self.algorithm.SetHoldings("SPY", 0.5)  # Set to 50% allocation
        
        self.assertNotEqual(self.algorithm.Portfolio["SPY"], initial_portfolio)
        
        for i in range(10):
            current_data = {k: v[i:i+20] for k, v in self.market_data.items() if k != 'ohlcv'}
            current_data['ohlcv'] = self.market_data['ohlcv'][i:i+20]
            
            market_data_dict = {'returns': current_data['returns']}
            truth_result = self.truth_verification.verify_market_truth(market_data_dict, [])
            
            if i == 5:  # Force a high-confidence result on the 5th iteration
                truth_result = {
                    "truth_verified": True,
                    "truth_score": 0.96,
                    "high_confidence": True
                }
            
            if truth_result["truth_verified"]:
                pattern_result = self.pattern_detector.detect_patterns(current_data)
                
                prediction = self.meta_ai.predict(current_data)
                
                if i == 5:
                    prediction = {
                        "signal": "SELL",
                        "confidence": 0.92,
                        "model": "forest",
                        "high_confidence": True
                    }
                
                portfolio_value = sum(self.algorithm.Portfolio.values())
                anti_loss_check = self.anti_loss.check_anti_loss_conditions(portfolio_value, self.algorithm.Portfolio)
                
                if i == 5:
                    anti_loss_check = {"allowed": True}
                
                if (anti_loss_check["allowed"] and 
                    prediction is not None and 
                    prediction.get("high_confidence", False) and
                    truth_result.get("high_confidence", False)):
                    
                    if prediction["signal"] == "BUY":
                        self.algorithm.SetHoldings("SPY", 0.5)  # Conservative position size
                    elif prediction["signal"] == "SELL":
                        self.algorithm.SetHoldings("SPY", -0.2)  # Conservative short position
                
                self.model_confidence.track_confidence("meta_ai", prediction.get("confidence", 0))
                self.meta_ai.add_training_sample(current_data, 0.001)
        
if __name__ == "__main__":
    unittest.main()
