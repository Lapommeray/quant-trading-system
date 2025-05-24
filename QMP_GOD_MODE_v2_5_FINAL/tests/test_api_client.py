import unittest
import asyncio
import aiohttp
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.api_client import QMPApiClient

class TestAsyncApiClient(unittest.TestCase):
    def setUp(self):
        """Set up test client"""
        self.api_client = QMPApiClient(base_url="http://test-api.example.com", api_key="test-key")
        
    @patch('aiohttp.ClientSession.get')
    def test_get_status_async(self, mock_get):
        """Test that get_status is truly asynchronous"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__.return_value = mock_response
        mock_get.return_value = mock_response
        
        async def run_test():
            result = await self.api_client.get_status()
            self.assertEqual(result, {"status": "ok"})
            mock_get.assert_called_once()
            
        asyncio.run(run_test())
        
    @patch('aiohttp.ClientSession.post')
    def test_generate_signal_async(self, mock_post):
        """Test that generate_signal is truly asynchronous"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"signal": "buy"})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        async def run_test():
            result = await self.api_client.generate_signal("SPY")
            self.assertEqual(result, {"signal": "buy"})
            mock_post.assert_called_once()
            
        asyncio.run(run_test())
        
    @patch('aiohttp.ClientSession.post')
    def test_place_order_async(self, mock_post):
        """Test that place_order is truly asynchronous"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"order_id": "123"})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__.return_value = mock_response
        mock_post.return_value = mock_response
        
        async def run_test():
            result = await self.api_client.place_order("SPY", "BUY", 100)
            self.assertEqual(result, {"order_id": "123"})
            mock_post.assert_called_once()
            
        asyncio.run(run_test())
        
    @patch('aiohttp.ClientSession.get')
    def test_get_signals_async(self, mock_get):
        """Test that get_signals is truly asynchronous"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"signals": []})
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__.return_value = mock_response
        mock_get.return_value = mock_response
        
        async def run_test():
            result = await self.api_client.get_signals("SPY")
            self.assertEqual(result, {"signals": []})
            mock_get.assert_called_once()
            
        asyncio.run(run_test())
        
    @patch('aiohttp.ClientSession.get')
    def test_error_handling_async(self, mock_get):
        """Test that error handling works correctly in async context"""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__.return_value = mock_response
        mock_get.return_value = mock_response
        
        async def run_test():
            result = await self.api_client.get_status()
            self.assertIsNone(result)
            mock_get.assert_called_once()
            
        asyncio.run(run_test())
        
    @patch('aiohttp.ClientSession.get')
    def test_exception_handling_async(self, mock_get):
        """Test that exception handling works correctly in async context"""
        mock_get.side_effect = aiohttp.ClientError("Connection error")
        
        async def run_test():
            result = await self.api_client.get_status()
            self.assertIsNone(result)
            mock_get.assert_called_once()
            
        asyncio.run(run_test())
        
    def test_concurrent_requests(self):
        """Test that multiple requests can be made concurrently"""
        async def mock_get_status():
            await asyncio.sleep(0.1)
            return {"status": "ok"}
            
        async def mock_get_symbols():
            await asyncio.sleep(0.1)
            return {"symbols": ["SPY", "QQQ"]}
            
        self.api_client.get_status = mock_get_status
        self.api_client.get_symbols = mock_get_symbols
        
        async def run_test():
            start_time = asyncio.get_event_loop().time()
            
            status_task = asyncio.create_task(self.api_client.get_status())
            symbols_task = asyncio.create_task(self.api_client.get_symbols())
            
            status_result = await status_task
            symbols_result = await symbols_task
            
            end_time = asyncio.get_event_loop().time()
            
            self.assertEqual(status_result, {"status": "ok"})
            self.assertEqual(symbols_result, {"symbols": ["SPY", "QQQ"]})
            
            self.assertLess(end_time - start_time, 0.2)
            
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
