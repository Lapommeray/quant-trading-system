#!/usr/bin/env python
"""
Test module for event blackout functionality
"""

import unittest
import sys
import os
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.event_blackout import EventBlackoutManager
from core.black_swan_detector import BlackSwanDetector

class TestEventBlackout(unittest.TestCase):
    """Test event blackout functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.blackout_manager = EventBlackoutManager()
        
    def test_blackout_periods(self):
        """Test that blackout periods meet the 30min-1hr requirement"""
        for event, config in self.blackout_manager.blackout_events.items():
            duration = config["duration"]
            self.assertGreaterEqual(duration, 30, 
                                  f"Event {event} has duration less than 30 minutes")
            self.assertLessEqual(duration, 120, 
                               f"Event {event} has duration greater than 120 minutes")
                               
    def test_nfp_blackout(self):
        """Test NFP blackout period (30min)"""
        test_date = datetime(2025, 5, 2, 8, 30, 0)  # Friday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "NFP": {"time": "08:30", "duration": 30, "days": [4]},  # Friday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "NFP")
        
        test_date = datetime(2025, 5, 2, 8, 59, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 2, 9, 0, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_fomc_blackout(self):
        """Test FOMC blackout period (120min)"""
        test_date = datetime(2025, 5, 7, 14, 0, 0)  # Wednesday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "FOMC": {"time": "14:00", "duration": 120, "days": [2]},  # Wednesday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "FOMC")
        
        test_date = datetime(2025, 5, 7, 15, 59, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 7, 16, 0, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_cpi_blackout(self):
        """Test CPI blackout period (60min)"""
        test_date = datetime(2025, 5, 6, 8, 30, 0)  # Tuesday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "CPI": {"time": "08:30", "duration": 60, "days": [1]},  # Tuesday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "CPI")
        
        test_date = datetime(2025, 5, 6, 9, 29, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 6, 9, 30, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_gdp_blackout(self):
        """Test GDP blackout period (45min)"""
        test_date = datetime(2025, 5, 8, 8, 30, 0)  # Thursday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "GDP": {"time": "08:30", "duration": 45, "days": [3]},  # Thursday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "GDP")
        
        test_date = datetime(2025, 5, 8, 9, 14, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 8, 9, 15, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_retail_sales_blackout(self):
        """Test Retail Sales blackout period (30min)"""
        test_date = datetime(2025, 5, 15, 8, 30, 0)  # Mid-month
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "RETAIL_SALES": {"time": "08:30", "duration": 30, "days": [3]},  # Thursday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "RETAIL_SALES")
        
        test_date = datetime(2025, 5, 15, 8, 59, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 15, 9, 0, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_ecb_rate_blackout(self):
        """Test ECB Rate Decision blackout period (60min)"""
        test_date = datetime(2025, 5, 8, 12, 45, 0)  # Thursday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "ECB_RATE": {"time": "12:45", "duration": 60, "days": [3]},  # Thursday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "ECB_RATE")
        
        test_date = datetime(2025, 5, 8, 13, 44, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 8, 13, 45, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_boe_policy_blackout(self):
        """Test BOE Policy Decision blackout period (30min)"""
        test_date = datetime(2025, 5, 8, 11, 0, 0)  # Thursday
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "BOE_POLICY": {"time": "11:00", "duration": 30, "days": [3]},  # Thursday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "BOE_POLICY")
        
        test_date = datetime(2025, 5, 8, 11, 29, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 8, 11, 30, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_china_pmi_blackout(self):
        """Test China PMI blackout period (30min)"""
        test_date = datetime(2025, 5, 1, 1, 45, 0)  # Monthly
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "CHINA_PMI": {"time": "01:45", "duration": 30, "days": [3]},  # Thursday
        }
        
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        self.assertEqual(event, "CHINA_PMI")
        
        test_date = datetime(2025, 5, 1, 2, 14, 59)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertTrue(is_blackout)
        
        test_date = datetime(2025, 5, 1, 2, 15, 1)
        is_blackout, event = self.blackout_manager.is_blackout_period_sync(test_date)
        self.assertFalse(is_blackout)
        
        self.blackout_manager.blackout_events = original_events
        
    def test_weekend_blackout(self):
        """Test weekend blackout"""
        test_date = datetime(2025, 5, 3, 12, 0, 0)  # Saturday
        is_weekend = self.blackout_manager.check_weekend_market(test_date)
        self.assertTrue(is_weekend)
        
        test_date = datetime(2025, 5, 4, 12, 0, 0)  # Sunday
        is_weekend = self.blackout_manager.check_weekend_market(test_date)
        self.assertTrue(is_weekend)
        
        test_date = datetime(2025, 5, 5, 12, 0, 0)  # Monday
        is_weekend = self.blackout_manager.check_weekend_market(test_date)
        self.assertFalse(is_weekend)
        
    def test_black_swan_simulation(self):
        """Test black swan simulation"""
        import pandas as pd
        import numpy as np
        
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        
        results = self.blackout_manager.simulate_black_swan_events(returns)
        
        self.assertEqual(len(results), 4)  # 4 scenarios
        
        for result in results:
            self.assertIn('scenario', result)
            self.assertIn('impact', result)
            self.assertIn('max_drawdown', result)
            self.assertIn('recovery_time', result)
            
            self.assertGreaterEqual(result['max_drawdown'], -0.19)
            
    def test_black_swan_detector_integration(self):
        """Test black swan detector integration"""
        mock_detector = MagicMock(spec=BlackSwanDetector)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        future = loop.create_future()
        future.set_result(True)
        
        mock_detector.global_risk_check = MagicMock(return_value=future)
        
        self.blackout_manager.set_black_swan_detector(mock_detector)
        
        async def test_async():
            test_date = datetime(2025, 5, 1, 10, 0, 0)  # Normal time
            is_blackout, event = await self.blackout_manager.is_blackout_period(test_date)
            self.assertTrue(is_blackout)
            self.assertEqual(event, "BLACK_SWAN_EVENT")
            
        loop.run_until_complete(test_async())
        loop.close()
        
        mock_detector.global_risk_check.assert_called_once()
        
    def test_async_blackout_check(self):
        """Test async blackout check"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        original_events = self.blackout_manager.blackout_events.copy()
        self.blackout_manager.blackout_events = {
            "NFP": {"time": "08:30", "duration": 30, "days": [4]},  # Friday
        }
        
        async def test_async():
            test_date = datetime(2025, 5, 2, 8, 30, 0)  # NFP
            is_blackout, event = await self.blackout_manager.is_blackout_period(test_date)
            self.assertTrue(is_blackout)
            self.assertEqual(event, "NFP")
            
        loop.run_until_complete(test_async())
        loop.close()
        
        self.blackout_manager.blackout_events = original_events
            
if __name__ == '__main__':
    unittest.main()
