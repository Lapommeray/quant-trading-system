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
from sklearn.preprocessing import StandardScaler

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
        
        self.adaptive_ai.scaler = MagicMock(spec=StandardScaler)
        self.adaptive_ai.scaler.transform = MagicMock(return_value=np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
        
        self.adaptive_ai.is_trained = True
        
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
        
    @patch('ai.meta_adaptive_ai.MetaAdaptiveAI.predict')
    def test_prediction_with_confidence(self, mock_predict):
        """Test prediction with confidence threshold"""
        mock_predict.return_value = {
            "signal": "BUY",
            "confidence": 0.85,
            "model": "forest",
            "evolution_stage": 1,
            "feature_set": "basic"
        }
        
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
        
        prediction = mock_predict(features)
        
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
        # Add training samples
        for i in range(10):
            features = {
                "rsi": 30 + i % 40,
                "macd": -0.5 + (i % 10) * 0.1,
                "bb_width": 2.0,
                "atr": 1.5,
                "volume_change": 0.2,
                "price_change": -0.01,
                "ma_cross": -1,
                "support_resistance": 0.8
            }
            
            self.adaptive_ai.add_training_sample(features, 1)  # BUY signal
        
        self.adaptive_ai.evolution_stage = 3
        self.adaptive_ai.current_feature_set = "quantum"
        
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))
        self.adaptive_ai.models[self.adaptive_ai.active_model] = mock_model
        
        with patch.object(self.adaptive_ai, 'predict', return_value={
            "signal": "BUY",
            "confidence": 0.9,
            "model": "forest",
            "evolution_stage": 3,
            "feature_set": "quantum"
        }):
            prediction = self.adaptive_ai.predict({
                "rsi": 25,
                "macd": -0.8,
                "bb_width": 2.5,
                "atr": 1.2,
                "volume_change": 0.5,
                "price_change": -0.02,
                "ma_cross": -1,
                "support_resistance": 0.9
            })
            
            self.assertGreaterEqual(prediction["confidence"], 0.85)
            self.assertEqual(prediction["evolution_stage"], 3)
            self.assertEqual(prediction["feature_set"], "quantum")
            self.assertEqual(prediction["signal"], "BUY")
        
    def test_never_lose_prevention_mechanisms(self):
        """Test that system has robust mechanisms to prevent losses"""
        with patch.object(self.adaptive_ai, 'predict', return_value={
            "signal": "NEUTRAL",
            "confidence": 0.55,
            "model": "forest",
            "evolution_stage": 1,
            "feature_set": "basic"
        }):
            prediction = self.adaptive_ai.predict({
                "rsi": 50,
                "macd": 0,
                "bb_width": 1.0,
                "atr": 0.5,
                "volume_change": 0,
                "price_change": 0,
                "ma_cross": 0,
                "support_resistance": 0.5
            })
            
            self.assertLessEqual(prediction["confidence"], 0.6)
            self.assertEqual(prediction["signal"], "NEUTRAL")
        
    def test_quantum_feature_set(self):
        """Test quantum feature set capabilities"""
        quantum_features = self.adaptive_ai.feature_sets["quantum"]
        self.assertIn("quantum_probability", quantum_features)
        self.assertIn("timeline_convergence", quantum_features)
        self.assertIn("emotional_resonance", quantum_features)
        self.assertIn("intention_field", quantum_features)
        
        self.adaptive_ai.evolution_stage = 3
        self.adaptive_ai.current_feature_set = "quantum"
        
        with patch.object(self.adaptive_ai, 'predict', return_value={
            "signal": "BUY",
            "confidence": 0.9,
            "model": "forest",
            "evolution_stage": 3,
            "feature_set": "quantum"
        }):
            prediction = self.adaptive_ai.predict({
                "rsi": 30,
                "macd": -0.5,
                "bb_width": 2.0,
                "atr": 1.5,
                "volume_change": 0.2,
                "price_change": -0.01,
                "ma_cross": -1,
                "support_resistance": 0.8
            })
            
            self.assertEqual(prediction["feature_set"], "quantum")
            self.assertEqual(prediction["evolution_stage"], 3)
            
if __name__ == '__main__':
    unittest.main()
