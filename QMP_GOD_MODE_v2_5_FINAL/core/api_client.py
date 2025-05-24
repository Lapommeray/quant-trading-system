"""
Async API Client

This module provides an asynchronous client for the QMP Overrider API.
It handles non-blocking communication with the API server for signal generation, order execution, and system monitoring.
"""

import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

class QMPApiClient:
    """
    QMP Async API Client
    
    Provides an asynchronous client for the QMP Overrider API.
    It handles non-blocking communication with the API server for signal generation, order execution, and system monitoring.
    Uses aiohttp for true async/await pattern to prevent blocking during market hours.
    """
    
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        """
        Initialize QMP API Client
        
        Parameters:
        - base_url: API base URL
        - api_key: API key
        """
        self.base_url = base_url
        self.api_key = api_key
        
        self.logger = self._setup_logger()
        self.logger.info(f"Initializing QMP API Client with base URL: {base_url}")
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("QMPApiClient")
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
    
    async def get_status(self):
        """
        Get system status asynchronously
        
        Returns:
        - System status
        """
        url = f"{self.base_url}/status"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error getting status: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return None
    
    async def generate_signal(self, symbol, timestamp=None):
        """
        Generate trading signal asynchronously
        
        Parameters:
        - symbol: Symbol to generate signal for
        - timestamp: Timestamp to generate signal for
        
        Returns:
        - Signal data
        """
        url = f"{self.base_url}/signal"
        
        data = {
            "symbol": symbol
        }
        
        if timestamp:
            data["timestamp"] = timestamp.isoformat()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self._get_headers(), json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error generating signal: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    async def place_order(self, symbol, direction, quantity, order_type="market", limit_price=None, stop_price=None):
        """
        Place trading order asynchronously
        
        Parameters:
        - symbol: Symbol to trade
        - direction: Direction of the trade (BUY or SELL)
        - quantity: Quantity to trade
        - order_type: Order type (market, limit, stop, etc.)
        - limit_price: Limit price (for limit orders)
        - stop_price: Stop price (for stop orders)
        
        Returns:
        - Order data
        """
        url = f"{self.base_url}/order"
        
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
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self._get_headers(), json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error placing order: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def get_signals(self, symbol=None, start_date=None, end_date=None, limit=100):
        """
        Get historical signals asynchronously
        
        Parameters:
        - symbol: Symbol to get signals for
        - start_date: Start date
        - end_date: End date
        - limit: Maximum number of signals to return
        
        Returns:
        - Signals data
        """
        url = f"{self.base_url}/signals"
        
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
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(), params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error getting signals: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting signals: {e}")
            return None
    
    async def get_orders(self, symbol=None, start_date=None, end_date=None, limit=100):
        """
        Get historical orders asynchronously
        
        Parameters:
        - symbol: Symbol to get orders for
        - start_date: Start date
        - end_date: End date
        - limit: Maximum number of orders to return
        
        Returns:
        - Orders data
        """
        url = f"{self.base_url}/orders"
        
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
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(), params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error getting orders: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return None
    
    async def get_symbols(self):
        """
        Get available symbols asynchronously
        
        Returns:
        - Symbols data
        """
        url = f"{self.base_url}/symbols"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error getting symbols: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return None
    
    async def get_performance(self, symbol=None, start_date=None, end_date=None):
        """
        Get performance metrics asynchronously
        
        Parameters:
        - symbol: Symbol to get performance for
        - start_date: Start date
        - end_date: End date
        
        Returns:
        - Performance data
        """
        url = f"{self.base_url}/performance"
        
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(), params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Error getting performance: {error_text}")
                        return None
        except Exception as e:
            self.logger.error(f"Error getting performance: {e}")
            return None
