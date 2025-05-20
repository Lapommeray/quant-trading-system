"""
Alpaca Executor

This module provides an execution interface for the Alpaca API to be used with the QMP Overrider system.
It handles order execution, position management, and account information.
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

class AlpacaExecutor:
    """
    Alpaca Executor
    
    Provides an execution interface for the Alpaca API to be used with the QMP Overrider system.
    It handles order execution, position management, and account information.
    """
    
    def __init__(self, api_key=None, api_secret=None, base_url="https://paper-api.alpaca.markets"):
        """
        Initialize Alpaca Executor
        
        Parameters:
        - api_key: Alpaca API key
        - api_secret: Alpaca API secret
        - base_url: Alpaca API base URL
        """
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        self.base_url = base_url
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        self.logger = self._setup_logger()
        self.logger.info("Initializing Alpaca Executor")
        
        self.order_history = []
        self.position_cache = {}
        self.account_cache = None
        self.last_cache_update = None
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AlpacaExecutor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_account(self, force_refresh=False):
        """
        Get account information
        
        Parameters:
        - force_refresh: Force refresh of account cache
        
        Returns:
        - Account information
        """
        if not force_refresh and self.account_cache and self.last_cache_update:
            if (datetime.now() - self.last_cache_update).total_seconds() < 60:
                return self.account_cache
        
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            self.account_cache = response.json()
            self.last_cache_update = datetime.now()
            return self.account_cache
        else:
            self.logger.error(f"Error getting account: {response.text}")
            return None
    
    def get_positions(self, force_refresh=False):
        """
        Get positions
        
        Parameters:
        - force_refresh: Force refresh of position cache
        
        Returns:
        - Positions
        """
        if not force_refresh and self.position_cache and self.last_cache_update:
            if (datetime.now() - self.last_cache_update).total_seconds() < 60:
                return self.position_cache
        
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            positions = response.json()
            self.position_cache = {p["symbol"]: p for p in positions}
            self.last_cache_update = datetime.now()
            return positions
        else:
            self.logger.error(f"Error getting positions: {response.text}")
            return None
    
    def get_position(self, symbol):
        """
        Get position for a specific symbol
        
        Parameters:
        - symbol: Symbol to get position for
        
        Returns:
        - Position information
        """
        positions = self.get_positions()
        
        if positions:
            for position in positions:
                if position["symbol"] == symbol:
                    return position
        
        return None
    
    def place_market_order(self, symbol, qty, side):
        """
        Place market order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (buy or sell)
        
        Returns:
        - Order information
        """
        return self.place_order(symbol, qty, side, "market", "gtc")
    
    def place_limit_order(self, symbol, qty, side, limit_price):
        """
        Place limit order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (buy or sell)
        - limit_price: Limit price
        
        Returns:
        - Order information
        """
        return self.place_order(symbol, qty, side, "limit", "gtc", limit_price=limit_price)
    
    def place_stop_order(self, symbol, qty, side, stop_price):
        """
        Place stop order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (buy or sell)
        - stop_price: Stop price
        
        Returns:
        - Order information
        """
        return self.place_order(symbol, qty, side, "stop", "gtc", stop_price=stop_price)
    
    def place_order(self, symbol, qty, side, type="market", time_in_force="gtc", limit_price=None, stop_price=None):
        """
        Place order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (buy or sell)
        - type: Order type (market, limit, stop, etc.)
        - time_in_force: Time in force (gtc, ioc, etc.)
        - limit_price: Limit price (for limit orders)
        - stop_price: Stop price (for stop orders)
        
        Returns:
        - Order information
        """
        url = f"{self.base_url}/v2/orders"
        data = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        
        if limit_price and type in ["limit", "stop_limit"]:
            data["limit_price"] = str(limit_price)
        
        if stop_price and type in ["stop", "stop_limit"]:
            data["stop_price"] = str(stop_price)
        
        self.logger.info(f"Placing {type} order: {symbol} {side} {qty}")
        
        response = requests.post(url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            order = response.json()
            self.order_history.append(order)
            return order
        else:
            self.logger.error(f"Error placing order: {response.text}")
            return None
    
    def get_orders(self, status="open"):
        """
        Get orders
        
        Parameters:
        - status: Order status (open, closed, all)
        
        Returns:
        - Orders
        """
        url = f"{self.base_url}/v2/orders"
        params = {"status": status}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Error getting orders: {response.text}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel order
        
        Parameters:
        - order_id: Order ID to cancel
        
        Returns:
        - True if successful, False otherwise
        """
        url = f"{self.base_url}/v2/orders/{order_id}"
        
        response = requests.delete(url, headers=self.headers)
        
        if response.status_code == 204:
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        else:
            self.logger.error(f"Error cancelling order: {response.text}")
            return False
    
    def cancel_all_orders(self):
        """
        Cancel all orders
        
        Returns:
        - True if successful, False otherwise
        """
        url = f"{self.base_url}/v2/orders"
        
        response = requests.delete(url, headers=self.headers)
        
        if response.status_code == 204:
            self.logger.info("Cancelled all orders")
            return True
        else:
            self.logger.error(f"Error cancelling all orders: {response.text}")
            return False
    
    def close_position(self, symbol):
        """
        Close position
        
        Parameters:
        - symbol: Symbol to close position for
        
        Returns:
        - True if successful, False otherwise
        """
        url = f"{self.base_url}/v2/positions/{symbol}"
        
        response = requests.delete(url, headers=self.headers)
        
        if response.status_code == 204:
            self.logger.info(f"Closed position: {symbol}")
            return True
        else:
            self.logger.error(f"Error closing position: {response.text}")
            return False
    
    def close_all_positions(self):
        """
        Close all positions
        
        Returns:
        - True if successful, False otherwise
        """
        url = f"{self.base_url}/v2/positions"
        
        response = requests.delete(url, headers=self.headers)
        
        if response.status_code == 204:
            self.logger.info("Closed all positions")
            return True
        else:
            self.logger.error(f"Error closing all positions: {response.text}")
            return False
    
    def get_order_by_id(self, order_id):
        """
        Get order by ID
        
        Parameters:
        - order_id: Order ID
        
        Returns:
        - Order information
        """
        url = f"{self.base_url}/v2/orders/{order_id}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Error getting order: {response.text}")
            return None
    
    def get_portfolio_history(self, period="1M", timeframe="1D"):
        """
        Get portfolio history
        
        Parameters:
        - period: Period (1D, 1M, 3M, 1A, etc.)
        - timeframe: Timeframe (1Min, 15Min, 1H, 1D, etc.)
        
        Returns:
        - Portfolio history
        """
        url = f"{self.base_url}/v2/account/portfolio/history"
        params = {
            "period": period,
            "timeframe": timeframe
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Error getting portfolio history: {response.text}")
            return None
