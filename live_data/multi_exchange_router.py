"""
Multi-Exchange Router - Unified Exchange Interface

This module provides a unified interface for interacting with multiple
cryptocurrency exchanges, with automatic failover and load balancing.
"""

import os
import time
import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from live_data.exchange_connector import ExchangeConnector
from live_data.websocket_streams import WebSocketStreams
from live_data.api_vault import APIVault
from live_data.data_verifier import DataVerifier

logger = logging.getLogger(__name__)

class MultiExchangeRouter:
    """
    Multi-exchange router for unified exchange interface with failover.
    
    This class provides a unified interface for interacting with multiple
    cryptocurrency exchanges, with automatic failover and load balancing.
    """
    
    def __init__(self, exchanges: List[str], api_vault: Optional[APIVault] = None,
                data_verifier: Optional[DataVerifier] = None):
        """
        Initialize the multi-exchange router.
        
        Parameters:
        - exchanges: List of exchange IDs to connect to
        - api_vault: APIVault instance for secure API key management
        - data_verifier: DataVerifier instance for data verification
        """
        self.api_vault = api_vault or APIVault()
        self.data_verifier = data_verifier or DataVerifier()
        
        self.exchanges = {}
        self.websockets = {}
        self.primary_exchange = None
        
        for exchange_id in exchanges:
            try:
                self.exchanges[exchange_id] = ExchangeConnector(exchange_id, self.api_vault)
                logger.info(f"Initialized exchange connector for {exchange_id}")
                
                if not self.primary_exchange:
                    self.primary_exchange = exchange_id
            except Exception as e:
                logger.error(f"Failed to initialize exchange connector for {exchange_id}: {e}")
        
        if not self.exchanges:
            raise ValueError("No exchanges could be initialized")
        
        logger.info(f"Initialized multi-exchange router with {len(self.exchanges)} exchanges")
        logger.info(f"Primary exchange: {self.primary_exchange}")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """
        Test connections to all exchanges.
        
        Returns:
        - Dictionary of exchange IDs and connection status
        """
        results = {}
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                results[exchange_id] = exchange.test_connection()
            except Exception as e:
                logger.error(f"Error testing connection to {exchange_id}: {e}")
                results[exchange_id] = False
        
        if self.primary_exchange in results and not results[self.primary_exchange]:
            for exchange_id, status in results.items():
                if status:
                    self.primary_exchange = exchange_id
                    logger.info(f"Updated primary exchange to {exchange_id}")
                    break
        
        return results
    
    def get_exchange(self, exchange_id: Optional[str] = None) -> ExchangeConnector:
        """
        Get exchange connector for a specific exchange or primary exchange.
        
        Parameters:
        - exchange_id: Exchange ID to get connector for, or None for primary
        
        Returns:
        - Exchange connector
        """
        if exchange_id and exchange_id in self.exchanges:
            return self.exchanges[exchange_id]
        
        return self.exchanges[self.primary_exchange]
    
    def get_websocket(self, exchange_id: Optional[str] = None) -> WebSocketStreams:
        """
        Get WebSocket streams for a specific exchange or primary exchange.
        
        Parameters:
        - exchange_id: Exchange ID to get WebSocket for, or None for primary
        
        Returns:
        - WebSocket streams
        """
        exchange_id = exchange_id or self.primary_exchange
        
        if exchange_id not in self.websockets:
            try:
                self.websockets[exchange_id] = WebSocketStreams(exchange_id)
                self.websockets[exchange_id].start()
            except Exception as e:
                logger.error(f"Failed to initialize WebSocket streams for {exchange_id}: {e}")
                raise
        
        return self.websockets[exchange_id]
    
    def fetch_ticker(self, symbol: str, exchange_id: Optional[str] = None,
                    verify: bool = True) -> Dict[str, Any]:
        """
        Fetch ticker data for a symbol from an exchange.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - exchange_id: Exchange ID to fetch from, or None for automatic
        - verify: Whether to verify the data
        
        Returns:
        - Ticker data
        """
        if exchange_id and exchange_id in self.exchanges:
            try:
                ticker = self.exchanges[exchange_id].fetch_ticker(symbol)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_ticker_data(
                        ticker, symbol, exchange_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Ticker data verification failed: {reason}")
                        raise ValueError(f"Ticker data verification failed: {reason}")
                
                return ticker
            except Exception as e:
                logger.error(f"Error fetching ticker from {exchange_id}: {e}")
        
        errors = {}
        
        for ex_id, exchange in self.exchanges.items():
            if ex_id == exchange_id:
                continue  # Already tried
            
            try:
                ticker = exchange.fetch_ticker(symbol)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_ticker_data(
                        ticker, symbol, ex_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Ticker data verification failed for {ex_id}: {reason}")
                        continue
                
                return ticker
            except Exception as e:
                errors[ex_id] = str(e)
                logger.error(f"Error fetching ticker from {ex_id}: {e}")
        
        raise ValueError(f"Failed to fetch ticker for {symbol} from any exchange: {errors}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m',
                   since: Optional[int] = None, limit: Optional[int] = None,
                   exchange_id: Optional[str] = None, verify: bool = True) -> List[List[float]]:
        """
        Fetch OHLCV (candle) data for a symbol from an exchange.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - timeframe: Timeframe (e.g., '1m', '1h', '1d')
        - since: Timestamp in milliseconds for start time
        - limit: Number of candles to fetch
        - exchange_id: Exchange ID to fetch from, or None for automatic
        - verify: Whether to verify the data
        
        Returns:
        - List of OHLCV candles [timestamp, open, high, low, close, volume]
        """
        if exchange_id and exchange_id in self.exchanges:
            try:
                ohlcv = self.exchanges[exchange_id].fetch_ohlcv(symbol, timeframe, since, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_ohlcv_data(
                        ohlcv, symbol, exchange_id, timeframe
                    )
                    
                    if not is_authentic:
                        logger.warning(f"OHLCV data verification failed: {reason}")
                        raise ValueError(f"OHLCV data verification failed: {reason}")
                
                return ohlcv
            except Exception as e:
                logger.error(f"Error fetching OHLCV from {exchange_id}: {e}")
        
        errors = {}
        
        for ex_id, exchange in self.exchanges.items():
            if ex_id == exchange_id:
                continue  # Already tried
            
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_ohlcv_data(
                        ohlcv, symbol, ex_id, timeframe
                    )
                    
                    if not is_authentic:
                        logger.warning(f"OHLCV data verification failed for {ex_id}: {reason}")
                        continue
                
                return ohlcv
            except Exception as e:
                errors[ex_id] = str(e)
                logger.error(f"Error fetching OHLCV from {ex_id}: {e}")
        
        raise ValueError(f"Failed to fetch OHLCV for {symbol} from any exchange: {errors}")
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None,
                        exchange_id: Optional[str] = None, verify: bool = True) -> Dict[str, Any]:
        """
        Fetch order book for a symbol from an exchange.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - limit: Limit the number of orders returned
        - exchange_id: Exchange ID to fetch from, or None for automatic
        - verify: Whether to verify the data
        
        Returns:
        - Order book data
        """
        if exchange_id and exchange_id in self.exchanges:
            try:
                order_book = self.exchanges[exchange_id].fetch_order_book(symbol, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_order_book_data(
                        order_book, symbol, exchange_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Order book data verification failed: {reason}")
                        raise ValueError(f"Order book data verification failed: {reason}")
                
                return order_book
            except Exception as e:
                logger.error(f"Error fetching order book from {exchange_id}: {e}")
        
        errors = {}
        
        for ex_id, exchange in self.exchanges.items():
            if ex_id == exchange_id:
                continue  # Already tried
            
            try:
                order_book = exchange.fetch_order_book(symbol, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_order_book_data(
                        order_book, symbol, ex_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Order book data verification failed for {ex_id}: {reason}")
                        continue
                
                return order_book
            except Exception as e:
                errors[ex_id] = str(e)
                logger.error(f"Error fetching order book from {ex_id}: {e}")
        
        raise ValueError(f"Failed to fetch order book for {symbol} from any exchange: {errors}")
    
    def fetch_trades(self, symbol: str, since: Optional[int] = None,
                    limit: Optional[int] = None, exchange_id: Optional[str] = None,
                    verify: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a symbol from an exchange.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - since: Timestamp in milliseconds for start time
        - limit: Number of trades to fetch
        - exchange_id: Exchange ID to fetch from, or None for automatic
        - verify: Whether to verify the data
        
        Returns:
        - List of trades
        """
        if exchange_id and exchange_id in self.exchanges:
            try:
                trades = self.exchanges[exchange_id].fetch_trades(symbol, since, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_trade_data(
                        trades, symbol, exchange_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Trade data verification failed: {reason}")
                        raise ValueError(f"Trade data verification failed: {reason}")
                
                return trades
            except Exception as e:
                logger.error(f"Error fetching trades from {exchange_id}: {e}")
        
        errors = {}
        
        for ex_id, exchange in self.exchanges.items():
            if ex_id == exchange_id:
                continue  # Already tried
            
            try:
                trades = exchange.fetch_trades(symbol, since, limit)
                
                if verify:
                    is_authentic, reason = self.data_verifier.verify_trade_data(
                        trades, symbol, ex_id
                    )
                    
                    if not is_authentic:
                        logger.warning(f"Trade data verification failed for {ex_id}: {reason}")
                        continue
                
                return trades
            except Exception as e:
                errors[ex_id] = str(e)
                logger.error(f"Error fetching trades from {ex_id}: {e}")
        
        raise ValueError(f"Failed to fetch trades for {symbol} from any exchange: {errors}")
    
    def subscribe_to_ticker(self, symbol: str, callback, exchange_id: Optional[str] = None) -> bool:
        """
        Subscribe to ticker updates for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - callback: Callback function to handle updates
        - exchange_id: Exchange ID to subscribe to, or None for primary
        
        Returns:
        - True if subscription was successful, False otherwise
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            ws = self.get_websocket(exchange_id)
            
            if exchange_id == 'binance':
                stream_name = f"{symbol.lower().replace('/', '')}@ticker"
            elif exchange_id == 'coinbase':
                stream_name = f"ticker:{symbol.replace('/', '-')}"
            elif exchange_id == 'kraken':
                stream_name = f"ticker:{symbol}"
            else:
                stream_name = f"ticker:{symbol}"
            
            return ws.subscribe(stream_name, callback)
        except Exception as e:
            logger.error(f"Error subscribing to ticker for {symbol} on {exchange_id}: {e}")
            return False
    
    def subscribe_to_trades(self, symbol: str, callback, exchange_id: Optional[str] = None) -> bool:
        """
        Subscribe to trade updates for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - callback: Callback function to handle updates
        - exchange_id: Exchange ID to subscribe to, or None for primary
        
        Returns:
        - True if subscription was successful, False otherwise
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            ws = self.get_websocket(exchange_id)
            
            if exchange_id == 'binance':
                stream_name = f"{symbol.lower().replace('/', '')}@trade"
            elif exchange_id == 'coinbase':
                stream_name = f"matches:{symbol.replace('/', '-')}"
            elif exchange_id == 'kraken':
                stream_name = f"trade:{symbol}"
            else:
                stream_name = f"trades:{symbol}"
            
            return ws.subscribe(stream_name, callback)
        except Exception as e:
            logger.error(f"Error subscribing to trades for {symbol} on {exchange_id}: {e}")
            return False
    
    def subscribe_to_order_book(self, symbol: str, callback, exchange_id: Optional[str] = None) -> bool:
        """
        Subscribe to order book updates for a symbol.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - callback: Callback function to handle updates
        - exchange_id: Exchange ID to subscribe to, or None for primary
        
        Returns:
        - True if subscription was successful, False otherwise
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            ws = self.get_websocket(exchange_id)
            
            if exchange_id == 'binance':
                stream_name = f"{symbol.lower().replace('/', '')}@depth"
            elif exchange_id == 'coinbase':
                stream_name = f"level2:{symbol.replace('/', '-')}"
            elif exchange_id == 'kraken':
                stream_name = f"book:{symbol}"
            else:
                stream_name = f"orderbook:{symbol}"
            
            return ws.subscribe(stream_name, callback)
        except Exception as e:
            logger.error(f"Error subscribing to order book for {symbol} on {exchange_id}: {e}")
            return False
    
    def create_order(self, symbol: str, order_type: str, side: str,
                    amount: float, price: Optional[float] = None,
                    exchange_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new order on an exchange.
        
        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT')
        - order_type: Order type (e.g., 'limit', 'market')
        - side: Order side ('buy' or 'sell')
        - amount: Order amount
        - price: Order price (required for limit orders)
        - exchange_id: Exchange ID to create order on, or None for primary
        
        Returns:
        - Order information
        """
        exchange_id = exchange_id or self.primary_exchange
        
        try:
            return self.exchanges[exchange_id].create_order(symbol, order_type, side, amount, price)
        except Exception as e:
            logger.error(f"Error creating order on {exchange_id}: {e}")
            raise
    
    def close(self) -> None:
        """Close all exchange and WebSocket connections."""
        for exchange_id, ws in list(self.websockets.items()):
            try:
                ws.stop()
                logger.info(f"Closed WebSocket connection for {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection for {exchange_id}: {e}")
        
        for exchange_id, exchange in list(self.exchanges.items()):
            try:
                exchange.close()
                logger.info(f"Closed exchange connection for {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing exchange connection for {exchange_id}: {e}")
        
        self.websockets = {}
        self.exchanges = {}
        self.primary_exchange = None
