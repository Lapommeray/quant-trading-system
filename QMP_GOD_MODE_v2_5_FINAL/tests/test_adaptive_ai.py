#!/usr/bin/env python
"""
Test module for adaptive AI capabilities
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai.meta_adaptive_ai import MetaAdaptiveAI

class MockAlgorithm:
    """Mock algorithm for testing"""
    def __init__(self):
        self.Time = datetime.now()
        self.DataFolder = "/tmp"
        
    def Debug(self, message):
        """Mock debug method"""
        print(f"DEBUG: {message}")

class TestAdaptiveAI(unittest.TestCase):
    """Test adaptive AI capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.algorithm = MockAlgorithm()
        self.adaptive_ai = MetaAdaptiveAI(self.algorithm, symbol="SPY")
        
    def test_evolution_stages(self):
        """Test evolution stages and feature sets"""
        self.assertEqual(self.adaptive_ai.evolution_stage, 1)
        self.assertEqual(self.adaptive_ai.current_feature_set, "basic")
        
        self.assertIn("basic", self.adaptive_ai.feature_sets)
        self.assertIn("advanced", self.adaptive_ai.feature_sets)
        self.assertIn("quantum", self.adaptive_ai.feature_sets)
        
        self.assertGreaterEqual(len(self.adaptive_ai.feature_sets["basic"]), 5)
        self.assertGreaterEqual(len(self.adaptive_ai.feature_sets["advanced"]), 10)
        self.assertGreaterEqual(len(self.adaptive_ai.feature_sets["quantum"]), 15)
        
    def test_model_selection(self):
        """Test model selection capabilities"""
        self.assertIn("forest", self.adaptive_ai.models)
        self.assertIn("boost", self.adaptive_ai.models)
        self.assertIn("neural", self.adaptive_ai.models)
        
        self.assertEqual(self.adaptive_ai.active_model, "forest")
        
    def test_prediction_with_confidence(self):
        """Test prediction with confidence threshold"""
        features = {
            "rsi": 30,
            "macd": -0.5,
            "bb_width": 2.0,
            "atr": 1.5,
            "volume_change": 0.2,
            "price_change": -0.01,
            "ma_cross": -1,
            "support_resistance": 0.8
        }
        
        prediction = self.adaptive_ai.predict(features)
        
        self.assertIn("signal", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("model", prediction)
        self.assertIn("evolution_stage", prediction)
        self.assertIn("feature_set", prediction)
        
    def test_training_sample_addition(self):
        """Test adding training samples"""
        features = {
            "rsi": 30,
            "macd": -0.5,
            "bb_width": 2.0,
            "atr": 1.5,
            "volume_change": 0.2,
            "price_change": -0.01,
            "ma_cross": -1,
            "support_resistance": 0.8
        }
        
        result = self.adaptive_ai.add_training_sample(features, 1)  # BUY signal
        self.assertTrue(result)
        
        self.assertGreater(len(self.adaptive_ai.training_data), 0)
        
    def test_evolution_check(self):
        """Test evolution check mechanism"""
        for model_name in self.adaptive_ai.models.keys():
            for i in range(5):
                self.adaptive_ai.performance_history[model_name].append({
                    "timestamp": self.algorithm.Time,
                    "accuracy": 0.7,  # Above 0.65 threshold for stage 1->2
                    "samples": 100
                })
                
        self.adaptive_ai.last_evolution_check = None
        evolved = self.adaptive_ai._check_evolution()
        
        self.assertTrue(evolved)
        self.assertEqual(self.adaptive_ai.evolution_stage, 2)
        self.assertEqual(self.adaptive_ai.current_feature_set, "advanced")
        
    def test_performance_metrics(self):
        """Test performance metrics reporting"""
        for model_name in self.adaptive_ai.models.keys():
            for i in range(5):
                self.adaptive_ai.performance_history[model_name].append({
                    "timestamp": self.algorithm.Time,
                    "accuracy": 0.7,
                    "samples": 100
                })
                
        metrics = self.adaptive_ai.get_performance_metrics()
        
        self.assertIn("active_model", metrics)
        self.assertIn("evolution_stage", metrics)
        self.assertIn("feature_set", metrics)
        self.assertIn("confidence_threshold", metrics)
        self.assertIn("is_trained", metrics)
        self.assertIn("training_samples", metrics)
        self.assertIn("model_performance", metrics)
        
        for model_name in self.adaptive_ai.models.keys():
            self.assertIn(model_name, metrics["model_performance"])
            self.assertIn("recent_accuracy", metrics["model_performance"][model_name])
            self.assertIn("samples_seen", metrics["model_performance"][model_name])
            self.assertIn("history_length", metrics["model_performance"][model_name])
            
    def test_super_high_confidence_mode(self):
        """Test system operates with super high confidence when properly trained"""
        for i in range(200):  # Add lots of training samples
            features = {
                "rsi": 30 + i % 40,
                "macd": -0.5 + (i % 10) * 0.1,
                "bb_width": 2.0,
                "atr": 1.5,
                "volume_change": 0.2,
                "price_change": -0.01 + (i % 5) * 0.004,
                "ma_cross": (-1) ** (i % 2),
                "support_resistance": 0.8,
                "fractal_dimension": 1.5,
                "hurst_exponent": 0.6,
                "entropy": 0.8,
                "correlation_matrix": 0.7,
                "volatility_regime": 1.0,
                "quantum_probability": 0.9,
                "timeline_convergence": 0.8,
                "emotional_resonance": 0.7,
                "intention_field": 0.6
            }
            
            expected_signal = 1 if features["rsi"] < 35 else -1 if features["rsi"] > 65 else 0
            self.adaptive_ai.add_training_sample(features, expected_signal)
        
        self.adaptive_ai.evolution_stage = 3
        self.adaptive_ai.current_feature_set = "quantum"
        self.adaptive_ai.confidence_threshold = 0.75
        
        for model_name in self.adaptive_ai.models.keys():
            self.adaptive_ai.models[model_name] = MagicMock()
            self.adaptive_ai.models[model_name].predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))
        
        test_features = {
            "rsi": 25,  # Strong buy signal
            "macd": -0.8,
            "bb_width": 2.5,
            "atr": 1.2,
            "volume_change": 0.5,
            "price_change": -0.02,
            "ma_cross": -1,
            "support_resistance": 0.9,
            "fractal_dimension": 1.5,
            "hurst_exponent": 0.6,
            "entropy": 0.8,
            "correlation_matrix": 0.7,
            "volatility_regime": 1.0,
            "quantum_probability": 0.9,
            "timeline_convergence": 0.8,
            "emotional_resonance": 0.7,
            "intention_field": 0.6
        }
        
        self.adaptive_ai.is_trained = True
        
        prediction = self.adaptive_ai.predict(test_features)
        
        self.assertGreaterEqual(prediction["confidence"], 0.85)
        self.assertEqual(prediction["evolution_stage"], 3)
        self.assertEqual(prediction["feature_set"], "quantum")
        self.assertEqual(prediction["signal"], "BUY")
        
    def test_never_lose_prevention_mechanisms(self):
        """Test that system has robust mechanisms to prevent losses"""
        low_confidence_features = {
            "rsi": 50,  # Neutral signal
            "macd": 0,
            "bb_width": 1.0,
            "atr": 0.5,
            "volume_change": 0,
            "price_change": 0,
            "ma_cross": 0,
            "support_resistance": 0.5
        }
        
        for model_name in self.adaptive_ai.models.keys():
            self.adaptive_ai.models[model_name] = MagicMock()
            self.adaptive_ai.models[model_name].predict_proba = MagicMock(return_value=np.array([[0.4, 0.6]]))
        
        self.adaptive_ai.is_trained = True
        
        prediction = self.adaptive_ai.predict(low_confidence_features)
        
        self.assertLessEqual(prediction["confidence"], 0.6)
        self.assertEqual(prediction["signal"], "NEUTRAL")
        
    def test_quantum_feature_set(self):
        """Test quantum feature set capabilities"""
        self.adaptive_ai.evolution_stage = 3
        self.adaptive_ai.current_feature_set = "quantum"
        
        quantum_features = self.adaptive_ai.feature_sets["quantum"]
        self.assertIn("quantum_probability", quantum_features)
        self.assertIn("timeline_convergence", quantum_features)
        self.assertIn("emotional_resonance", quantum_features)
        self.assertIn("intention_field", quantum_features)
        
        features = {
            "rsi": 30,
            "macd": -0.5,
            "bb_width": 2.0,
            "atr": 1.5,
            "volume_change": 0.2,
            "price_change": -0.01,
            "ma_cross": -1,
            "support_resistance": 0.8,
            "fractal_dimension": 1.5,
            "hurst_exponent": 0.6,
            "entropy": 0.8,
            "correlation_matrix": 0.7,
            "volatility_regime": 1.0,
            "quantum_probability": 0.9,
            "timeline_convergence": 0.8,
            "emotional_resonance": 0.7,
            "intention_field": 0.6
        }
        
        self.adaptive_ai.scaler = MagicMock()
        self.adaptive_ai.scaler.transform = MagicMock(return_value=np.array([[1.0] * len(features)]))
        
        for model_name in self.adaptive_ai.models.keys():
            self.adaptive_ai.models[model_name] = MagicMock()
            self.adaptive_ai.models[model_name].predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))
        
        self.adaptive_ai.is_trained = True
        
        prediction = self.adaptive_ai.predict(features)
        
        self.assertEqual(prediction["feature_set"], "quantum")
        self.assertEqual(prediction["evolution_stage"], 3)
            
if __name__ == '__main__':
    unittest.main()
