"""
Crypto Feed

This module provides a feed for cryptocurrency data to be used with the QMP Overrider system.
It handles fetching market data from various cryptocurrency exchanges.
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

class CryptoFeed:
    """
    Crypto Feed
    
    Provides a feed for cryptocurrency data to be used with the QMP Overrider system.
    It handles fetching market data from various cryptocurrency exchanges.
    """
    
    def __init__(self):
        """Initialize Crypto Feed"""
        self.logger = self._setup_logger()
        self.logger.info("Initializing Crypto Feed")
        
        self.cache = {}
        self.cache_expiry = {}
        
        self.rate_limits = {
            "binance": {"limit": 1200, "window": 60, "count": 0, "reset_time": time.time() + 60},
            "coinbase": {"limit": 10, "window": 1, "count": 0, "reset_time": time.time() + 1},
            "kraken": {"limit": 15, "window": 1, "count": 0, "reset_time": time.time() + 1}
        }
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("CryptoFeed")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_rate_limit(self, exchange):
        """
        Check rate limit
        
        Parameters:
        - exchange: Exchange to check rate limit for
        
        Returns:
        - True if rate limit is not exceeded, False otherwise
        """
        if exchange not in self.rate_limits:
            return True
        
        rate_limit = self.rate_limits[exchange]
        
        if time.time() > rate_limit["reset_time"]:
            rate_limit["count"] = 0
            rate_limit["reset_time"] = time.time() + rate_limit["window"]
        
        if rate_limit["count"] >= rate_limit["limit"]:
            self.logger.warning(f"Rate limit exceeded for {exchange}")
            return False
        
        rate_limit["count"] += 1
        return True
    
    def _get_data(self, url, params=None, headers=None, exchange=None, cache_key=None, cache_duration=60):
        """
        Get data from API
        
        Parameters:
        - url: API URL
        - params: API parameters
        - headers: API headers
        - exchange: Exchange name for rate limiting
        - cache_key: Cache key
        - cache_duration: Cache duration in seconds
        
        Returns:
        - API response
        """
        if cache_key and cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                return self.cache[cache_key]
        
        if exchange and not self._check_rate_limit(exchange):
            time.sleep(1)  # Wait for rate limit to reset
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                if cache_key:
                    self.cache[cache_key] = data
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_duration)
                
                return data
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding JSON: {response.text}")
                return None
        else:
            self.logger.error(f"Error getting data: {response.text}")
            return None
    
    def get_binance_klines(self, symbol, interval="1m", limit=500):
        """
        Get Binance klines
        
        Parameters:
        - symbol: Symbol to get data for (e.g., BTCUSDT)
        - interval: Time interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        - limit: Number of klines to get (max 1000)
        
        Returns:
        - Klines data
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        cache_key = f"binance_klines_{symbol}_{interval}_{limit}"
        
        data = self._get_data(url, params, exchange="binance", cache_key=cache_key)
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        
        for col in df.columns:
            if col not in ["open_time", "close_time"]:
                df[col] = pd.to_numeric(df[col])
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df = df.set_index("open_time")
        
        return df
    
    def get_binance_ticker(self, symbol):
        """
        Get Binance ticker
        
        Parameters:
        - symbol: Symbol to get data for (e.g., BTCUSDT)
        
        Returns:
        - Ticker data
        """
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        cache_key = f"binance_ticker_{symbol}"
        
        data = self._get_data(url, params, exchange="binance", cache_key=cache_key, cache_duration=10)
        
        if not data:
            return None
        
        return data
    
    def get_binance_order_book(self, symbol, limit=100):
        """
        Get Binance order book
        
        Parameters:
        - symbol: Symbol to get data for (e.g., BTCUSDT)
        - limit: Number of bids and asks to get (max 5000)
        
        Returns:
        - Order book data
        """
        url = "https://api.binance.com/api/v3/depth"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        cache_key = f"binance_order_book_{symbol}_{limit}"
        
        data = self._get_data(url, params, exchange="binance", cache_key=cache_key, cache_duration=5)
        
        if not data:
            return None
        
        bids_df = pd.DataFrame(data["bids"], columns=["price", "quantity"])
        asks_df = pd.DataFrame(data["asks"], columns=["price", "quantity"])
        
        bids_df["price"] = pd.to_numeric(bids_df["price"])
        bids_df["quantity"] = pd.to_numeric(bids_df["quantity"])
        asks_df["price"] = pd.to_numeric(asks_df["price"])
        asks_df["quantity"] = pd.to_numeric(asks_df["quantity"])
        
        return {
            "bids": bids_df,
            "asks": asks_df,
            "lastUpdateId": data["lastUpdateId"]
        }
    
    def get_coinbase_candles(self, product_id, granularity=60, start=None, end=None):
        """
        Get Coinbase candles
        
        Parameters:
        - product_id: Product ID to get data for (e.g., BTC-USD)
        - granularity: Granularity in seconds (60, 300, 900, 3600, 21600, 86400)
        - start: Start time (ISO 8601)
        - end: End time (ISO 8601)
        
        Returns:
        - Candles data
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        params = {"granularity": granularity}
        
        if start:
            params["start"] = start
        
        if end:
            params["end"] = end
        
        cache_key = f"coinbase_candles_{product_id}_{granularity}_{start}_{end}"
        
        data = self._get_data(url, params, exchange="coinbase", cache_key=cache_key)
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        
        for col in df.columns:
            if col != "time":
                df[col] = pd.to_numeric(df[col])
        
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        
        df = df.sort_index()
        
        return df
    
    def get_coinbase_ticker(self, product_id):
        """
        Get Coinbase ticker
        
        Parameters:
        - product_id: Product ID to get data for (e.g., BTC-USD)
        
        Returns:
        - Ticker data
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
        
        cache_key = f"coinbase_ticker_{product_id}"
        
        data = self._get_data(url, exchange="coinbase", cache_key=cache_key, cache_duration=10)
        
        if not data:
            return None
        
        return data
    
    def get_coinbase_order_book(self, product_id, level=2):
        """
        Get Coinbase order book
        
        Parameters:
        - product_id: Product ID to get data for (e.g., BTC-USD)
        - level: Level of detail (1, 2, 3)
        
        Returns:
        - Order book data
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/book"
        params = {"level": level}
        
        cache_key = f"coinbase_order_book_{product_id}_{level}"
        
        data = self._get_data(url, params, exchange="coinbase", cache_key=cache_key, cache_duration=5)
        
        if not data:
            return None
        
        bids_df = pd.DataFrame(data["bids"], columns=["price", "size", "num_orders"])
        asks_df = pd.DataFrame(data["asks"], columns=["price", "size", "num_orders"])
        
        bids_df["price"] = pd.to_numeric(bids_df["price"])
        bids_df["size"] = pd.to_numeric(bids_df["size"])
        asks_df["price"] = pd.to_numeric(asks_df["price"])
        asks_df["size"] = pd.to_numeric(asks_df["size"])
        
        return {
            "bids": bids_df,
            "asks": asks_df,
            "sequence": data["sequence"]
        }
    
    def get_kraken_ohlc(self, pair, interval=1, since=None):
        """
        Get Kraken OHLC data
        
        Parameters:
        - pair: Pair to get data for (e.g., XBTUSD)
        - interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        - since: Return committed OHLC data since given ID
        
        Returns:
        - OHLC data
        """
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            "pair": pair,
            "interval": interval
        }
        
        if since:
            params["since"] = since
        
        cache_key = f"kraken_ohlc_{pair}_{interval}_{since}"
        
        data = self._get_data(url, params, exchange="kraken", cache_key=cache_key)
        
        if not data:
            return None
        
        if "error" in data and data["error"]:
            self.logger.error(f"Kraken API error: {data['error']}")
            return None
        
        if "result" not in data:
            self.logger.error("Kraken API result not found")
            return None
        
        result = data["result"]
        
        if pair not in result:
            self.logger.error(f"Pair not found in result: {pair}")
            return None
        
        ohlc_data = result[pair]
        
        df = pd.DataFrame(ohlc_data, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        
        for col in df.columns:
            if col != "time":
                df[col] = pd.to_numeric(df[col])
        
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        
        df = df.sort_index()
        
        return df
    
    def get_kraken_ticker(self, pair):
        """
        Get Kraken ticker
        
        Parameters:
        - pair: Pair to get data for (e.g., XBTUSD)
        
        Returns:
        - Ticker data
        """
        url = "https://api.kraken.com/0/public/Ticker"
        params = {"pair": pair}
        
        cache_key = f"kraken_ticker_{pair}"
        
        data = self._get_data(url, params, exchange="kraken", cache_key=cache_key, cache_duration=10)
        
        if not data:
            return None
        
        if "error" in data and data["error"]:
            self.logger.error(f"Kraken API error: {data['error']}")
            return None
        
        if "result" not in data:
            self.logger.error("Kraken API result not found")
            return None
        
        result = data["result"]
        
        if pair not in result:
            self.logger.error(f"Pair not found in result: {pair}")
            return None
        
        ticker_data = result[pair]
        
        return ticker_data
    
    def get_kraken_order_book(self, pair, count=100):
        """
        Get Kraken order book
        
        Parameters:
        - pair: Pair to get data for (e.g., XBTUSD)
        - count: Maximum number of asks/bids to return
        
        Returns:
        - Order book data
        """
        url = "https://api.kraken.com/0/public/Depth"
        params = {
            "pair": pair,
            "count": count
        }
        
        cache_key = f"kraken_order_book_{pair}_{count}"
        
        data = self._get_data(url, params, exchange="kraken", cache_key=cache_key, cache_duration=5)
        
        if not data:
            return None
        
        if "error" in data and data["error"]:
            self.logger.error(f"Kraken API error: {data['error']}")
            return None
        
        if "result" not in data:
            self.logger.error("Kraken API result not found")
            return None
        
        result = data["result"]
        
        if pair not in result:
            self.logger.error(f"Pair not found in result: {pair}")
            return None
        
        order_book = result[pair]
        
        bids_df = pd.DataFrame(order_book["bids"], columns=["price", "volume", "timestamp"])
        asks_df = pd.DataFrame(order_book["asks"], columns=["price", "volume", "timestamp"])
        
        bids_df["price"] = pd.to_numeric(bids_df["price"])
        bids_df["volume"] = pd.to_numeric(bids_df["volume"])
        asks_df["price"] = pd.to_numeric(asks_df["price"])
        asks_df["volume"] = pd.to_numeric(asks_df["volume"])
        
        return {
            "bids": bids_df,
            "asks": asks_df
        }
