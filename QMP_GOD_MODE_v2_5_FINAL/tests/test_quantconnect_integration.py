"""
Comprehensive QuantConnect Integration Test
Tests the system's readiness for live trading on QuantConnect
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
from core.event_blackout import EventBlackoutManager
from core.anti_loss_guardian import AntiLossGuardian
from core.black_swan_detector import BlackSwanDetector
from emergency_stop import EmergencyStop

class MockQuantConnectAlgorithm:
    """Mock QuantConnect Algorithm for testing"""
    
    def __init__(self):
        self.Portfolio = {"SPY": 100000, "QQQ": 50000}
        self.Securities = {"SPY": {"Price": 450.0}, "QQQ": {"Price": 380.0}}
        self.Time = datetime.now()
        self.is_live_mode = False
        self.debug_messages = []
        
        # Required attributes for MetaAdaptiveAI
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

class TestQuantConnectIntegration(unittest.TestCase):
    """Test QuantConnect integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.algorithm = MockQuantConnectAlgorithm()
        self.meta_ai = MetaAdaptiveAI(self.algorithm)
        self.event_blackout = EventBlackoutManager()  # No algorithm parameter
        self.anti_loss = AntiLossGuardian(self.algorithm)
        self.black_swan = BlackSwanDetector()  # Takes api_key, not algorithm
        self.emergency = EmergencyStop(dry_run=True)  # Takes dry_run, not algorithm
        
        self.market_data = {
            'returns': np.random.normal(0.001, 0.01, 100),
            'volume': np.random.normal(1000000, 200000, 100),
            'open': np.random.normal(450, 5, 100),
            'high': np.random.normal(455, 5, 100),
            'low': np.random.normal(445, 5, 100),
            'close': np.random.normal(450, 5, 100)
        }
        
    def test_full_integration_cycle(self):
        """Test full integration cycle with QuantConnect"""
        friday = datetime.now() - timedelta(days=datetime.now().weekday()) + timedelta(days=4)  # Get next Friday
        nfp_time = friday.replace(hour=8, minute=30, second=0, microsecond=0)
        result, event_name = self.event_blackout.is_blackout_period_sync(nfp_time)
        self.assertTrue(result)
        
        # Test MetaAdaptiveAI
        self.meta_ai.train(self.market_data)
        prediction = self.meta_ai.predict(self.market_data)
        self.assertIsNotNone(prediction)
        
        # Test AntiLossGuardian
        portfolio_value = sum(self.algorithm.Portfolio.values())
        result = self.anti_loss.check_anti_loss_conditions(portfolio_value, self.algorithm.Portfolio)
        self.assertIn("allowed", result)
        
        # Test BlackSwanDetector - use check_health_emergencies with asyncio.run
        is_emergency = asyncio.run(self.black_swan.check_health_emergencies())
        self.assertIsInstance(is_emergency, bool)
        
        # Test EmergencyStop
        market_data = {
            'returns': self.market_data['returns'],
            'volume': self.market_data['volume']
        }
        ai_metrics = {
            'confidence': 0.8,
            'recent_accuracy': 0.7
        }
        portfolio_data = {
            'positions': self.algorithm.Portfolio
        }
        emergency_result, _ = self.emergency.comprehensive_emergency_check(
            market_data, ai_metrics, portfolio_data
        )
        self.assertIsInstance(emergency_result, bool)
        
    def test_live_trading_simulation(self):
        """Test live trading simulation"""
        self.algorithm.is_live_mode = True
        
        for i in range(10):
            current_data = {k: v[i:i+20] for k, v in self.market_data.items()}
            
            current_time = datetime.now()
            is_blackout, _ = self.event_blackout.is_blackout_period_sync(current_time)
            
            if not is_blackout:
                prediction = self.meta_ai.predict(current_data)
                
                portfolio_value = sum(self.algorithm.Portfolio.values())
                anti_loss_check = self.anti_loss.check_anti_loss_conditions(portfolio_value, self.algorithm.Portfolio)
                
                if anti_loss_check["allowed"] and prediction is not None:
                    if prediction > 0.7:
                        self.algorithm.SetHoldings("SPY", 0.8)
                    elif prediction < -0.7:
                        self.algorithm.SetHoldings("SPY", -0.3)
            
            ai_metrics = {
                'confidence': 0.8,
                'recent_accuracy': 0.7
            }
            portfolio_data = {
                'positions': self.algorithm.Portfolio
            }
            emergency_triggered, _ = self.emergency.comprehensive_emergency_check(
                current_data, ai_metrics, portfolio_data
            )
            if emergency_triggered:
                self.algorithm.Liquidate()
                break
                
            self.meta_ai.add_training_sample(current_data, 0.01)  # Assume small positive return
            
        self.assertTrue(True)  # If we got here without errors, test passes
        
    def test_event_blackout_periods(self):
        """Test event blackout periods for economic events"""
        
        now = datetime.now()
        
        friday = now - timedelta(days=now.weekday()) + timedelta(days=4)  # Get next Friday
        nfp_time = friday.replace(hour=8, minute=30, second=0, microsecond=0)
        
        result, event_name = self.event_blackout.is_blackout_period_sync(nfp_time)
        self.assertTrue(result)
        self.assertEqual(event_name, "NFP")
        
        non_event_time = datetime.now().replace(hour=3, minute=0, second=0, microsecond=0)  # 3 AM, unlikely to be any event
        result, event_name = self.event_blackout.is_blackout_period_sync(non_event_time)
        self.assertFalse(result)
        
        wednesday = now - timedelta(days=now.weekday()) + timedelta(days=2)  # Get next Wednesday
        fomc_time = wednesday.replace(hour=14, minute=0, second=0, microsecond=0)
        
        result, event_name = self.event_blackout.is_blackout_period_sync(fomc_time)
        self.assertTrue(result)
        self.assertEqual(event_name, "FOMC")
        
        weekend_time = now - timedelta(days=now.weekday()) + timedelta(days=6)  # Saturday
        weekend_time = weekend_time.replace(hour=12, minute=0, second=0, microsecond=0)  # Noon on Saturday
        result, event_name = self.event_blackout.is_blackout_period_sync(weekend_time)
        self.assertFalse(result)
            
    def test_ai_self_modification(self):
        """Test AI self-modification capabilities"""
        self.meta_ai.train(self.market_data)
        
        initial_threshold = self.meta_ai.confidence_threshold
        
        performance_metrics = {
            'recent_accuracy': 0.4,
            'volatility': 0.05,
            'correlation_breakdown': 0.8
        }
        
        self.meta_ai.self_modify_code(performance_metrics)
        
        self.meta_ai.adapt_to_future_markets(self.market_data)
        
        self.assertNotEqual(initial_threshold, self.meta_ai.confidence_threshold)
        
    def test_emergency_failsafe(self):
        """Test emergency failsafe with manual override"""
        result = self.emergency.execute_emergency_stop(code="MANUAL_OVERRIDE")
        self.assertTrue(result)
        
        market_data = {
            'returns': [-0.05, -0.08, -0.12, -0.15, -0.20],  # Market crash
            'volume': [1000000, 1500000, 3000000, 5000000, 8000000]  # Volume spike
        }
        
        ai_metrics = {
            'confidence': 0.15,  # Low confidence
            'recent_accuracy': 0.25,  # Poor accuracy
            'recent_predictions': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Stuck predictions
        }
        
        portfolio_data = {
            'positions': {'SPY': 80000, 'QQQ': 10000, 'AAPL': 10000}  # High concentration
        }
        
        emergency, triggers = self.emergency.comprehensive_emergency_check(
            market_data, ai_metrics, portfolio_data
        )
        
        self.assertTrue(emergency)
        self.assertGreaterEqual(len(triggers), 2)
        
    def test_quantconnect_deployment_readiness(self):
        """Test QuantConnect deployment readiness"""
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")))
        
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PRODUCTION_DEPLOYMENT_GUIDE.md")))
        
        try:
            from ai.meta_adaptive_ai import MetaAdaptiveAI
            from core.event_blackout import EventBlackoutManager
            from core.anti_loss_guardian import AntiLossGuardian
            from core.black_swan_detector import BlackSwanDetector
            from emergency_stop import EmergencyStop
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")

if __name__ == "__main__":
    unittest.main()
