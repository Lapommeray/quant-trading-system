#!/usr/bin/env python
"""
Test module for black swan detector capabilities
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.black_swan_detector import BlackSwanDetector
from core.event_blackout import EventBlackoutManager

class TestBlackSwanDetector(unittest.TestCase):
    """Test black swan detector capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.detector = BlackSwanDetector()
        self.blackout_manager = EventBlackoutManager()
        self.blackout_manager.set_black_swan_detector(self.detector)
        
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector.logger)
        self.assertEqual(len(self.detector.detected_events), 0)
        self.assertIsInstance(self.detector.last_check_time, dict)
        
    def test_health_emergency_detection(self):
        """Test health emergency detection"""
        async def run_test():
            # Mock feedparser response
            with patch('core.black_swan_detector.feedparser.parse') as mock_parse:
                mock_entry = MagicMock()
                mock_entry.title = "WHO declares pandemic emergency"
                mock_parse.return_value = MagicMock(entries=[mock_entry])
                
                # Test detection
                result = await self.detector.check_health_emergencies()
                self.assertTrue(result)
                self.assertEqual(len(self.detector.detected_events), 1)
                self.assertEqual(self.detector.detected_events[0]['type'], 'HEALTH_EMERGENCY')
        
        asyncio.run(run_test())
        
    def test_earthquake_detection(self):
        """Test earthquake detection"""
        async def run_test():
            # Mock aiohttp response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "features": [{
                    "properties": {
                        "mag": 7.5,
                        "place": "Test Location"
                    }
                }]
            })
            
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            with patch('aiohttp.ClientSession', return_value=mock_session):
                # Test detection
                result = await self.detector.check_major_earthquakes()
                self.assertTrue(result)
                self.assertEqual(len(self.detector.detected_events), 1)
                self.assertEqual(self.detector.detected_events[0]['type'], 'MAJOR_EARTHQUAKE')
        
        asyncio.run(run_test())
        
    def test_global_risk_check(self):
        """Test global risk check"""
        async def run_test():
            # Set up mocks
            with patch.object(self.detector, 'check_health_emergencies', AsyncMock(return_value=False)), \
                 patch.object(self.detector, 'check_major_earthquakes', AsyncMock(return_value=False)), \
                 patch.object(self.detector, 'check_solar_flares', AsyncMock(return_value=False)), \
                 patch.object(self.detector, 'check_bank_failures', AsyncMock(return_value=False)), \
                 patch.object(self.detector, 'check_geopolitical_crises', AsyncMock(return_value=True)), \
                 patch.object(self.detector, 'check_crypto_exchange_halts', AsyncMock(return_value=False)):
                
                # Test detection
                result = await self.detector.global_risk_check()
                self.assertTrue(result)
        
        asyncio.run(run_test())
        
    def test_integration_with_event_blackout(self):
        """Test integration with event blackout manager"""
        async def run_test():
            # Set up mock
            with patch.object(self.detector, 'global_risk_check', AsyncMock(return_value=True)):
                # Test integration
                is_blackout, event = await self.blackout_manager.is_blackout_period(datetime.now())
                self.assertTrue(is_blackout)
                self.assertEqual(event, "BLACK_SWAN_EVENT")
        
        asyncio.run(run_test())
        
    def test_rate_limiting(self):
        """Test API rate limiting"""
        # Check that last_check_time is updated
        self.detector.last_check_time['health'] = datetime.now()
        asyncio.run(self.detector.check_health_emergencies())
        self.assertEqual(len(self.detector.detected_events), 0)  # Should be rate limited
        
    def test_multiple_events_detection(self):
        """Test detection of multiple events"""
        # Add some test events
        self.detector.detected_events.append({
            'type': 'HEALTH_EMERGENCY',
            'title': 'Test Pandemic',
            'timestamp': datetime.now().isoformat(),
            'source': 'WHO'
        })
        self.detector.detected_events.append({
            'type': 'MAJOR_EARTHQUAKE',
            'magnitude': 8.0,
            'location': 'Test Location',
            'timestamp': datetime.now().isoformat(),
            'source': 'USGS'
        })
        
        # Test retrieval
        events = self.detector.get_detected_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['type'], 'HEALTH_EMERGENCY')
        self.assertEqual(events[1]['type'], 'MAJOR_EARTHQUAKE')
        
    def test_error_handling(self):
        """Test error handling in global risk check"""
        async def run_test():
            # Set up mocks to raise exceptions
            with patch.object(self.detector, 'check_health_emergencies', AsyncMock(side_effect=Exception("Test exception"))), \
                 patch.object(self.detector, 'check_major_earthquakes', AsyncMock(return_value=True)):
                
                # Test error handling
                result = await self.detector.global_risk_check()
                self.assertTrue(result)  # Should still return True due to earthquake
        
        asyncio.run(run_test())
        
if __name__ == '__main__':
    unittest.main()
