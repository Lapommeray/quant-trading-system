"""
Alpaca Adapter

This module provides an adapter for the Alpaca API to be used with the QMP Overrider system.
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta

class AlpacaAdapter:
    """
    Alpaca Adapter
    
    Provides an adapter for the Alpaca API to be used with the QMP Overrider system.
    """
    
    def __init__(self, api_key=None, api_secret=None, base_url="https://paper-api.alpaca.markets"):
        """
        Initialize Alpaca Adapter
        
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
        
        print("Initializing Alpaca Adapter")
    
    def get_account(self):
        """
        Get account information
        
        Returns:
        - Account information
        """
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting account: {response.text}")
            return None
    
    def get_positions(self):
        """
        Get positions
        
        Returns:
        - Positions
        """
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting positions: {response.text}")
            return None
    
    def place_order(self, symbol, qty, side, type="market", time_in_force="gtc"):
        """
        Place order
        
        Parameters:
        - symbol: Symbol to trade
        - qty: Quantity to trade
        - side: Side of the trade (buy or sell)
        - type: Order type (market, limit, etc.)
        - time_in_force: Time in force (gtc, ioc, etc.)
        
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
        
        response = requests.post(url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error placing order: {response.text}")
            return None
    
    def get_bars(self, symbol, timeframe="1Min", start=None, end=None, limit=100):
        """
        Get bars
        
        Parameters:
        - symbol: Symbol to get bars for
        - timeframe: Timeframe of the bars
        - start: Start time
        - end: End time
        - limit: Maximum number of bars to return
        
        Returns:
        - Bars
        """
        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        
        if not start:
            start = (datetime.now() - timedelta(days=1)).isoformat()
        
        if not end:
            end = datetime.now().isoformat()
        
        params = {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": limit
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting bars: {response.text}")
            return None
    
    def get_last_trade(self, symbol):
        """
        Get last trade
        
        Parameters:
        - symbol: Symbol to get last trade for
        
        Returns:
        - Last trade
        """
        url = f"{self.base_url}/v2/stocks/{symbol}/trades/latest"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting last trade: {response.text}")
            return None
    
    def get_last_quote(self, symbol):
        """
        Get last quote
        
        Parameters:
        - symbol: Symbol to get last quote for
        
        Returns:
        - Last quote
        """
        url = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting last quote: {response.text}")
            return None
