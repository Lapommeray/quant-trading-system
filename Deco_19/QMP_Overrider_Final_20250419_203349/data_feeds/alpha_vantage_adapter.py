"""
Alpha Vantage Adapter

This module provides an adapter for the Alpha Vantage API to be used with the QMP Overrider system.
It handles fetching market data, technical indicators, and fundamental data.
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

class AlphaVantageAdapter:
    """
    Alpha Vantage Adapter
    
    Provides an adapter for the Alpha Vantage API to be used with the QMP Overrider system.
    It handles fetching market data, technical indicators, and fundamental data.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize Alpha Vantage Adapter
        
        Parameters:
        - api_key: Alpha Vantage API key
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        
        self.logger = self._setup_logger()
        self.logger.info("Initializing Alpha Vantage Adapter")
        
        self.cache = {}
        self.cache_expiry = {}
    
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AlphaVantageAdapter")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_data(self, params, cache_key=None, cache_duration=3600):
        """
        Get data from Alpha Vantage API
        
        Parameters:
        - params: API parameters
        - cache_key: Cache key
        - cache_duration: Cache duration in seconds
        
        Returns:
        - API response
        """
        if cache_key and cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
                return self.cache[cache_key]
        
        params["apikey"] = self.api_key
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if "Error Message" in data:
                self.logger.error(f"API Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                self.logger.warning(f"API Note: {data['Note']}")
            
            if cache_key:
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_duration)
            
            return data
        else:
            self.logger.error(f"Error getting data: {response.text}")
            return None
    
    def get_time_series(self, symbol, interval="1min", outputsize="compact"):
        """
        Get time series data
        
        Parameters:
        - symbol: Symbol to get data for
        - interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
        - outputsize: Output size (compact or full)
        
        Returns:
        - Time series data
        """
        function = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY",
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY"
        }.get(interval, "TIME_SERIES_INTRADAY")
        
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval
        
        cache_key = f"time_series_{symbol}_{interval}_{outputsize}"
        
        data = self._get_data(params, cache_key)
        
        if not data:
            return None
        
        time_series_key = {
            "1min": "Time Series (1min)",
            "5min": "Time Series (5min)",
            "15min": "Time Series (15min)",
            "30min": "Time Series (30min)",
            "60min": "Time Series (60min)",
            "daily": "Time Series (Daily)",
            "weekly": "Weekly Time Series",
            "monthly": "Monthly Time Series"
        }.get(interval)
        
        if time_series_key not in data:
            self.logger.error(f"Time series key not found: {time_series_key}")
            return None
        
        time_series = data[time_series_key]
        
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        df.columns = [col.split(". ")[1] if ". " in col else col for col in df.columns]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        
        return df
    
    def get_technical_indicator(self, symbol, indicator, interval="daily", time_period=20, series_type="close"):
        """
        Get technical indicator
        
        Parameters:
        - symbol: Symbol to get data for
        - indicator: Technical indicator (SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, T3, MACD, MACDEXT, STOCH, STOCHF, RSI, STOCHRSI, WILLR, ADX, ADXR, APO, PPO, MOM, BOP, CCI, CMO, ROC, ROCR, AROON, AROONOSC, MFI, TRIX, ULTOSC, DX, MINUS_DI, PLUS_DI, MINUS_DM, PLUS_DM, BBANDS, MIDPOINT, MIDPRICE, SAR, TRANGE, ATR, NATR, AD, ADOSC, OBV, HT_TRENDLINE, HT_SINE, HT_TRENDMODE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR)
        - interval: Time interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
        - time_period: Time period
        - series_type: Series type (close, open, high, low)
        
        Returns:
        - Technical indicator data
        """
        params = {
            "function": indicator,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        }
        
        cache_key = f"technical_{symbol}_{indicator}_{interval}_{time_period}_{series_type}"
        
        data = self._get_data(params, cache_key)
        
        if not data:
            return None
        
        technical_key = f"Technical Analysis: {indicator}"
        
        if technical_key not in data:
            self.logger.error(f"Technical key not found: {technical_key}")
            return None
        
        technical = data[technical_key]
        
        df = pd.DataFrame.from_dict(technical, orient="index")
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        df.index = pd.to_datetime(df.index)
        
        df = df.sort_index()
        
        return df
    
    def get_quote(self, symbol):
        """
        Get quote
        
        Parameters:
        - symbol: Symbol to get quote for
        
        Returns:
        - Quote data
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        }
        
        cache_key = f"quote_{symbol}"
        
        data = self._get_data(params, cache_key, cache_duration=60)
        
        if not data:
            return None
        
        if "Global Quote" not in data:
            self.logger.error("Global Quote not found")
            return None
        
        quote = data["Global Quote"]
        
        return quote
    
    def get_search(self, keywords):
        """
        Search for symbols
        
        Parameters:
        - keywords: Keywords to search for
        
        Returns:
        - Search results
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }
        
        cache_key = f"search_{keywords}"
        
        data = self._get_data(params, cache_key)
        
        if not data:
            return None
        
        if "bestMatches" not in data:
            self.logger.error("Best matches not found")
            return None
        
        matches = data["bestMatches"]
        
        return matches
    
    def get_fundamental(self, symbol, function="OVERVIEW"):
        """
        Get fundamental data
        
        Parameters:
        - symbol: Symbol to get data for
        - function: Function (OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, EARNINGS)
        
        Returns:
        - Fundamental data
        """
        params = {
            "function": function,
            "symbol": symbol
        }
        
        cache_key = f"fundamental_{symbol}_{function}"
        
        data = self._get_data(params, cache_key, cache_duration=86400)
        
        if not data:
            return None
        
        return data
    
    def get_forex_rate(self, from_currency, to_currency):
        """
        Get forex rate
        
        Parameters:
        - from_currency: From currency
        - to_currency: To currency
        
        Returns:
        - Forex rate
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency
        }
        
        cache_key = f"forex_{from_currency}_{to_currency}"
        
        data = self._get_data(params, cache_key, cache_duration=300)
        
        if not data:
            return None
        
        if "Realtime Currency Exchange Rate" not in data:
            self.logger.error("Realtime Currency Exchange Rate not found")
            return None
        
        rate = data["Realtime Currency Exchange Rate"]
        
        return rate
    
    def get_crypto_rate(self, symbol, market="USD"):
        """
        Get crypto rate
        
        Parameters:
        - symbol: Crypto symbol
        - market: Market
        
        Returns:
        - Crypto rate
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": symbol,
            "to_currency": market
        }
        
        cache_key = f"crypto_{symbol}_{market}"
        
        data = self._get_data(params, cache_key, cache_duration=60)
        
        if not data:
            return None
        
        if "Realtime Currency Exchange Rate" not in data:
            self.logger.error("Realtime Currency Exchange Rate not found")
            return None
        
        rate = data["Realtime Currency Exchange Rate"]
        
        return rate
    
    def get_sector_performance(self):
        """
        Get sector performance
        
        Returns:
        - Sector performance data
        """
        params = {
            "function": "SECTOR"
        }
        
        cache_key = "sector_performance"
        
        data = self._get_data(params, cache_key, cache_duration=3600)
        
        if not data:
            return None
        
        return data
    
    def get_economic_indicator(self, indicator):
        """
        Get economic indicator
        
        Parameters:
        - indicator: Economic indicator (REAL_GDP, REAL_GDP_PER_CAPITA, TREASURY_YIELD, FEDERAL_FUNDS_RATE, CPI, INFLATION, RETAIL_SALES, DURABLES, UNEMPLOYMENT, NONFARM_PAYROLL)
        
        Returns:
        - Economic indicator data
        """
        params = {
            "function": indicator
        }
        
        cache_key = f"economic_{indicator}"
        
        data = self._get_data(params, cache_key, cache_duration=86400)
        
        if not data:
            return None
        
        if "data" not in data:
            self.logger.error("Economic data not found")
            return None
        
        economic_data = data["data"]
        
        df = pd.DataFrame(economic_data)
        
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col])
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        df = df.sort_index()
        
        return df
