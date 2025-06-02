#!/usr/bin/env python3
"""
Mock AlgorithmImports for testing without QuantConnect environment
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt_datetime, timedelta as dt_timedelta
from typing import Dict, List, Any, Optional

class QCAlgorithm:
    """Mock QuantConnect Algorithm base class"""
    def __init__(self):
        self.Time = dt_datetime.now()
        self.debug_messages = []
        self.log_messages = []
        self.DataFolder = "/tmp/mock_data"  # Mock data folder for testing
        
    def Debug(self, message):
        self.debug_messages.append(f"[{self.Time}] {message}")
        print(f"DEBUG: {message}")
        
    def Log(self, message):
        self.log_messages.append(f"[{self.Time}] {message}")
        print(f"LOG: {message}")
        
    def SetStartDate(self, year, month, day):
        pass
        
    def SetEndDate(self, year, month, day):
        pass
        
    def SetCash(self, amount):
        pass
        
    def AddCrypto(self, symbol, resolution=None):
        pass
        
    def AddForex(self, symbol, resolution=None):
        pass
        
    def GetParameter(self, key, default_value=""):
        """Mock GetParameter method for QuantConnect parameters"""
        return default_value

class TradeBar:
    """Mock TradeBar class"""
    def __init__(self, open_price, high, low, close, volume, time):
        self._open = open_price
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._time = time
        self.EndTime = time
        
    @property
    def Open(self):
        return self._open
        
    @property
    def High(self):
        return self._high
        
    @property
    def Low(self):
        return self._low
        
    @property
    def Close(self):
        return self._close
        
    @property
    def Volume(self):
        return self._volume
        
    @property
    def Time(self):
        return self._time

class Resolution:
    """Mock Resolution class"""
    Minute = "1min"
    Hour = "1hour"
    Daily = "1day"

class OrderStatus:
    """Mock OrderStatus enum"""
    Submitted = "Submitted"
    PartiallyFilled = "PartiallyFilled"
    Filled = "Filled"
    Canceled = "Canceled"

class TradeBarConsolidator:
    """Mock TradeBarConsolidator class"""
    def __init__(self, period):
        self.period = period

class Market:
    """Mock Market class"""
    USA = "USA"
    Crypto = "Crypto"

def timedelta(*args, **kwargs):
    return dt_timedelta(*args, **kwargs)

def datetime(*args, **kwargs):
    return dt_datetime(*args, **kwargs)

__all__ = [
    'QCAlgorithm', 'TradeBar', 'Resolution', 'OrderStatus',
    'TradeBarConsolidator', 'Market', 'timedelta', 'datetime', 'pd', 'np'
]
