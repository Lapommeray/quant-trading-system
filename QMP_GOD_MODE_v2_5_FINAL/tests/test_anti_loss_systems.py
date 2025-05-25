#!/usr/bin/env python
"""
Comprehensive testing for anti-loss systems
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.anti_loss_guardian import AntiLossGuardian
from ai.meta_adaptive_ai import MetaAdaptiveAI
from emergency_stop import EmergencyStop

class TestAntiLossSystems(unittest.TestCase):
    """Test comprehensive anti-loss protection systems"""
    
    def setUp(self):
        """Set up test environment"""
        self.algorithm = MagicMock()
        self.algorithm.Time = datetime.now()
        self.algorithm.Debug = MagicMock()
        
        self.guardian = AntiLossGuardian(self.algorithm)
        self.adaptive_ai = MetaAdaptiveAI(self.algorithm, symbol="SPY")
        self.emergency_stop = EmergencyStop(dry_run=True)
        
    def test_drawdown_protection_levels(self):
        """Test multi-level drawdown protection"""
        self.guardian.peak_portfolio_value = 100000
        
        result = self.guardian.check_anti_loss_conditions(portfolio_value=95000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "reduce_position")
        
        result = self.guardian.check_anti_loss_conditions(portfolio_value=90000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "halt_new_trades")
        
        result = self.guardian.check_anti_loss_conditions(portfolio_value=85000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "emergency_liquidation")
        
    def test_consecutive_loss_protection(self):
        """Test consecutive loss protection"""
        for i in range(3):
            self.guardian.update_trade_result(-100)
            
        result = self.guardian.check_anti_loss_conditions(portfolio_value=100000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "emergency_mode")
        self.assertTrue(self.guardian.emergency_mode)
        
    def test_position_concentration_protection(self):
        """Test position concentration protection"""
        current_positions = {
            "SPY": 60000,
            "QQQ": 20000,
            "AAPL": 20000
        }
        
        result = self.guardian.check_anti_loss_conditions(portfolio_value=100000, current_positions=current_positions)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "reduce_concentration")
        
    def test_ai_self_modification(self):
        """Test AI self-modification capabilities"""
        performance_metrics = {
            "recent_accuracy": 0.5,
            "samples": 100
        }
        
        original_threshold = self.adaptive_ai.confidence_threshold
        
        self.adaptive_ai.self_modify_code(performance_metrics)
        self.assertGreater(self.adaptive_ai.confidence_threshold, original_threshold)
        
        performance_metrics["recent_accuracy"] = 0.85
        self.adaptive_ai.evolution_stage = 1
        self.adaptive_ai.self_modify_code(performance_metrics)
        self.assertEqual(self.adaptive_ai.evolution_stage, 2)
        
    def test_future_market_adaptation(self):
        """Test adaptation to unknown market conditions"""
        market_data = {
            'returns': np.random.normal(0, 0.06, 100),  # High volatility
            'volume': np.random.normal(1000000, 200000, 100)
        }
        
        self.adaptive_ai.adapt_to_future_markets(market_data)
        self.assertEqual(self.adaptive_ai.risk_multiplier, 0.5)  # Should reduce risk
        
        trending_returns = np.cumsum(np.random.normal(0.01, 0.02, 100))  # Trending upward
        market_data = {
            'returns': trending_returns,
            'volume': np.random.normal(1000000, 200000, 100)
        }
        
        self.adaptive_ai.adapt_to_future_markets(market_data)
        self.assertIn("trend_strength", self.adaptive_ai.feature_sets[self.adaptive_ai.current_feature_set])
        
    def test_emergency_protocols(self):
        """Test emergency protocol activation"""
        self.guardian.emergency_mode = True
        protocols = self.guardian.emergency_protocols()
        
        self.assertTrue(protocols["liquidate_all"])
        self.assertTrue(protocols["block_new_trades"])
        self.assertTrue(protocols["notify_admin"])
        self.assertTrue(protocols["create_backup"])
        
    def test_ai_driven_emergency_detection(self):
        """Test AI-driven emergency detection"""
        market_data = {
            'returns': np.array([-0.05, -0.08, -0.12, -0.15, -0.20]),  # 20% crash
            'volume': np.array([2000000, 3000000, 5000000, 8000000, 10000000])
        }
        
        ai_metrics = {
            'confidence': 0.2,
            'recent_accuracy': 0.3
        }
        
        emergency, conditions = self.emergency_stop.ai_driven_emergency_check(market_data, ai_metrics)
        self.assertTrue(emergency)
        self.assertGreaterEqual(len(conditions), 2)
        
    def test_risk_multiplier_adjustment(self):
        """Test dynamic risk multiplier adjustment"""
        for i in range(5):
            self.guardian.update_trade_result(100)
            
        original_multiplier = self.guardian.risk_multiplier
        
        for i in range(5):
            self.guardian.update_trade_result(-100)
            
        self.assertLess(self.guardian.risk_multiplier, original_multiplier)
        
    def test_unusual_pattern_detection(self):
        """Test unusual trading pattern detection"""
        for i in range(10):
            self.guardian.update_trade_result(100 if i % 2 == 0 else -100)
            
        result = self.guardian.check_anti_loss_conditions(portfolio_value=100000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "pause_trading")
        
    def test_integration_of_all_systems(self):
        """Test integration of all anti-loss systems"""
        market_data = {
            'returns': np.array([-0.05, -0.08, -0.12, -0.15, -0.20]),  # 20% crash
            'volume': np.array([2000000, 3000000, 5000000, 8000000, 10000000])
        }
        
        self.adaptive_ai.adapt_to_future_markets(market_data)
        
        self.guardian.peak_portfolio_value = 100000
        result = self.guardian.check_anti_loss_conditions(portfolio_value=80000, current_positions={})
        self.assertFalse(result["allowed"])
        self.assertEqual(result["action"], "emergency_liquidation")
        
        ai_metrics = {
            'confidence': 0.2,
            'recent_accuracy': 0.3
        }
        emergency, conditions = self.emergency_stop.ai_driven_emergency_check(market_data, ai_metrics)
        self.assertTrue(emergency)
        
        self.assertGreaterEqual(self.adaptive_ai.confidence_threshold, 0.65)

if __name__ == '__main__':
    unittest.main()
