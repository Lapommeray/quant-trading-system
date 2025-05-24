import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class AsyncQMPApiClient:
    """Async QMP API Client with aiohttp for non-blocking calls"""
    
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.logger = self._setup_logger()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AsyncQMPApiClient")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
            
    def _get_headers(self):
        """Get request headers"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        return headers
            
    async def generate_signal(self, symbol, timestamp=None):
        """Generate trading signal asynchronously"""
        url = f"{self.base_url}/signal"
        headers = self._get_headers()
            
        data = {"symbol": symbol}
        if timestamp:
            data["timestamp"] = timestamp.isoformat()
            
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error generating signal: {await response.text()}")
                    return None
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
            
    async def get_market_data(self, symbol):
        """Fetch market data asynchronously"""
        url = f"{self.base_url}/market_data/{symbol}"
        headers = self._get_headers()
            
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error getting market data: {await response.text()}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return None
            
    async def place_order(self, symbol, direction, quantity, order_type="market", limit_price=None, stop_price=None):
        """Place trading order asynchronously"""
        url = f"{self.base_url}/order"
        headers = self._get_headers()
        
        data = {
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "order_type": order_type
        }
        
        if limit_price:
            data["limit_price"] = limit_price
        
        if stop_price:
            data["stop_price"] = stop_price
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error placing order: {await response.text()}")
                    return None
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
            
    async def get_signals(self, symbol=None, start_date=None, end_date=None, limit=100):
        """Get historical signals asynchronously"""
        url = f"{self.base_url}/signals"
        headers = self._get_headers()
        
        params = {
            "limit": limit
        }
        
        if symbol:
            params["symbol"] = symbol
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error getting signals: {await response.text()}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting signals: {e}")
            return None
            
    async def get_performance(self, symbol=None, start_date=None, end_date=None):
        """Get performance metrics asynchronously"""
        url = f"{self.base_url}/performance"
        headers = self._get_headers()
        
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error getting performance: {await response.text()}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting performance: {e}")
            return None
