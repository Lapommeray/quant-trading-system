"""
Exchange Connector - CCXT Integration

This module provides connectivity to cryptocurrency exchanges using CCXT.
It handles market data retrieval, order execution, and exchange information.
"""

import os
import time
import logging
import ccxt
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from live_data.api_vault import APIVault

logger = logging.getLogger(__name__)

class ExchangeConnector:
    """
    Exchange connector using CCXT for cryptocurrency exchange integration.
    
    This class provides methods for retrieving market data, executing orders,
    and managing exchange connections using the CCXT library.
    """
    
    def __init__(self, exchange_id: str, api_vault: Optional[APIVault] = None):
        """
        Initialize the exchange connector.
        
        Parameters:
        - exchange_id: ID of the exchange to connect to (e.g., 'binance', 'coinbase')
        - api_vault: APIVault instance for secure API key management
        """
        self.exchange_id = exchange_id.lower()
        self.api_vault = api_vault or APIVault()
        
        credentials = self.api_vault.get_credentials(self.exchange_id)
        
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(f"Exchange {self.exchange_id} not supported by CCXT")
        
        exchange_class = getattr(ccxt, self.exchange_id)
        
        options = {
            'adjustForTimeDifference': True,
            'recvWindow': 60000,  # For Binance
        }
        
        self.exchange = exchange_class({
            **credentials,
            'options': options,
            'enableRateLimit': True,
        })
        
        logger.info(f"Initialized {self.exchange_id} exchange connector")
    
    def test_connection(self) -> bool:
        """
        Test connection to the exchange.
        
        Returns:
        - True if connection is successful, False otherwise
        """
        try:
            self.exchange.load_markets()
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"🟢 LIVE DATA: {self.exchange_id} connected, BTC/USDT price: {ticker['last']}")
            return True
        except ccxt.NetworkError as e:
            logger.error(f"Network error connecting to {self.exchange_id}: {e}")
            return False
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error with {self.exchange_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error testing connection to {self.exchange_id}: {e}")
            return False
    
    def fetch_markets(self) -> Dict[str, Any]:
        """
        Fetch all markets from the exchange.
        
        Returns:
        - Dictionary of markets
        """
        try:
            self.exchange.load_markets(reload=True)
            return self.exchange.markets
        except Exception as e:
            logger.error(f"Error fetching markets from {self.exchange_id}: {e}")
            return {}
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch ticker data for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
        - Ticker data
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol} from {self.exchange_id}: {e}")
            return {}
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                   since: Optional[int] = None, limit: Optional[int] = None) -> List[List[float]]:
        """
        Fetch OHLCV (candle) data for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - timeframe: Timeframe (e.g., '1m', '1h', '1d')
        - since: Timestamp in milliseconds for start time
        - limit: Number of candles to fetch
        
        Returns:
        - List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        if not self.exchange.has['fetchOHLCV']:
            logger.error(f"Exchange {self.exchange_id} does not support fetchOHLCV")
            return []
        
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} from {self.exchange_id}: {e}")
            return []
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch order book for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - limit: Limit the number of orders returned
        
        Returns:
        - Order book data
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol} from {self.exchange_id}: {e}")
            return {}
    
    def fetch_trades(self, symbol: str, since: Optional[int] = None, 
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - since: Timestamp in milliseconds for start time
        - limit: Number of trades to fetch
        
        Returns:
        - List of trades
        """
        try:
            return self.exchange.fetch_trades(symbol, since, limit)
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol} from {self.exchange_id}: {e}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new order.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - order_type: Order type (e.g., 'limit', 'market')
        - side: Order side ('buy' or 'sell')
        - amount: Order amount
        - price: Order price (required for limit orders)
        
        Returns:
        - Order information
        """
        try:
            return self.exchange.create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            logger.error(f"Error creating {order_type} {side} order for {symbol} on {self.exchange_id}: {e}")
            return {}
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
        - Account balance information
        """
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Error fetching balance from {self.exchange_id}: {e}")
            return {}
    
    def fetch_my_trades(self, symbol: Optional[str] = None, 
                       since: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch user's trades.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - since: Timestamp in milliseconds for start time
        - limit: Number of trades to fetch
        
        Returns:
        - List of user's trades
        """
        try:
            return self.exchange.fetch_my_trades(symbol, since, limit)
        except Exception as e:
            logger.error(f"Error fetching my trades from {self.exchange_id}: {e}")
            return []
    
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
        - List of open orders
        """
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error fetching open orders from {self.exchange_id}: {e}")
            return []
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Parameters:
        - order_id: ID of the order to cancel
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        
        Returns:
        - Cancellation result
        """
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} on {self.exchange_id}: {e}")
            return {}
    
    def close(self) -> None:
        """Close the exchange connection."""
        try:
            self.exchange.close()
            logger.info(f"Closed connection to {self.exchange_id}")
        except Exception as e:
            logger.error(f"Error closing connection to {self.exchange_id}: {e}")
