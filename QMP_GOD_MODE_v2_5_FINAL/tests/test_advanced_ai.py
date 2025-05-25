"""
Test Advanced AI Modules
Comprehensive testing for all advanced AI capabilities
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.meta_adaptive_ai import MetaAdaptiveAI
from ai.market_intelligence import (
    LatencyCancellationField, 
    EmotionHarvestAI, 
    QuantumLiquiditySignatureReader,
    SovereignQuantumOracle
)

class MockAlgorithm:
    def __init__(self):
        self.Time = datetime.now()
        self.DataFolder = "/tmp/test_data"
        self.debug_messages = []
        
    def Debug(self, message):
        self.debug_messages.append(message)
        print(message)

class TestAdvancedAI(unittest.TestCase):
    """Test all advanced AI modules"""
    
    def setUp(self):
        self.algorithm = MockAlgorithm()
        self.meta_ai = MetaAdaptiveAI(self.algorithm)
        
        self.market_data = {
            'returns': np.random.normal(0.001, 0.02, 100).tolist(),
            'volume': np.random.normal(1000000, 200000, 100).tolist(),
            'open': np.random.normal(100, 2, 100).tolist(),
            'high': np.random.normal(102, 2, 100).tolist(),
            'low': np.random.normal(98, 2, 100).tolist(),
            'close': np.random.normal(100, 2, 100).tolist()
        }
        
    def test_time_resonant_neural_lattice(self):
        """Test Time-Resonant Neural Lattice functionality"""
        result = self.meta_ai.time_resonant_neural_lattice(self.market_data)
        
        self.assertIn("resonance", result)
        self.assertIn("temporal_patterns", result)
        self.assertIn("lattice_state", result)
        self.assertIsInstance(result["resonance"], float)
        self.assertIn(result["lattice_state"], ["resonating", "dormant"])
        
    def test_dna_self_rewrite(self):
        """Test Self-Rewriting DNA-AI functionality"""
        performance_metrics = {
            'recent_accuracy': 0.4,  # Low performance to trigger mutation
            'samples_seen': 1000
        }
        market_conditions = {
            'volatility': 0.06  # High volatility
        }
        
        result = self.meta_ai.dna_self_rewrite(performance_metrics, market_conditions)
        
        self.assertIn("mutation_triggered", result)
        self.assertIn("evolutionary_state", result)
        self.assertIn("genetic_diversity", result)
        self.assertTrue(result["mutation_triggered"])  # Should trigger with low performance
        
    def test_causal_quantum_reasoning(self):
        """Test Causal Quantum Reasoning Engine"""
        result = self.meta_ai.causal_quantum_reasoning(self.market_data)
        
        self.assertIn("causality_factors", result)
        self.assertIn("quantum_consciousness", result)
        self.assertIn("quantum_state", result)
        self.assertIn(result["quantum_state"], ["coherent", "decoherent"])
        self.assertIsInstance(result["quantum_consciousness"], float)
        
    def test_latency_cancellation_field(self):
        """Test Latency-Cancellation Field"""
        lcf = LatencyCancellationField()
        result = lcf.cancel_latency(self.market_data)
        
        self.assertIn("latency_cancelled", result)
        if result["latency_cancelled"]:
            self.assertIn("predicted_move", result)
            self.assertIn("temporal_advantage", result)
            
    def test_emotion_harvest_ai(self):
        """Test Emotion Harvest AI"""
        eha = EmotionHarvestAI()
        result = eha.harvest_emotions(self.market_data)
        
        self.assertIn("emotion", result)
        self.assertIn("intensity", result)
        self.assertIn(result["emotion"], ["fear", "greed", "complacency", "neutral"])
        self.assertGreaterEqual(result["intensity"], 0.0)
        self.assertLessEqual(result["intensity"], 1.0)
        
    def test_quantum_liquidity_signature_reader(self):
        """Test Quantum Liquidity Signature Reader"""
        qlsr = QuantumLiquiditySignatureReader()
        result = qlsr.read_liquidity_signature(self.market_data)
        
        self.assertIn("signature", result)
        self.assertIn("confidence", result)
        self.assertIn(result["signature"], [
            "market_maker_accumulation", 
            "institutional_momentum", 
            "mixed_flow", 
            "retail_dominated"
        ])
        
    def test_sovereign_quantum_oracle(self):
        """Test Sovereign Quantum Oracle"""
        sqo = SovereignQuantumOracle(self.algorithm)
        target_outcome = 0.005  # 0.5% positive return
        
        result = sqo.write_market_reality(target_outcome, self.market_data)
        
        self.assertIn("reality_written", result)
        if result["reality_written"]:
            self.assertIn("influence_strength", result)
            self.assertIn("quantum_coherence", result)
            
    def test_integration_with_anti_loss_systems(self):
        """Test that new AI modules work with existing anti-loss systems"""
        crash_data = {
            'returns': [-0.05, -0.08, -0.12, -0.10, -0.15] * 20,  # Market crash pattern
            'volume': [1000000, 1500000, 3000000, 2500000, 4000000] * 20
        }
        
        lattice_result = self.meta_ai.time_resonant_neural_lattice(crash_data)
        self.assertIsNotNone(lattice_result)
        
        crisis_performance = {'recent_accuracy': 0.3, 'error_rate': 0.7}
        crisis_conditions = {'volatility': 0.15}
        
        dna_result = self.meta_ai.dna_self_rewrite(crisis_performance, crisis_conditions)
        self.assertTrue(dna_result["mutation_triggered"])
        
        quantum_result = self.meta_ai.causal_quantum_reasoning(crash_data)
        self.assertLessEqual(quantum_result["quantum_consciousness"], 1.0)

if __name__ == "__main__":
    unittest.main()
