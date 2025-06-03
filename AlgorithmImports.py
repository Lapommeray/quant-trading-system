"""
Mock AlgorithmImports module for QuantConnect compatibility

This module provides stub implementations of QuantConnect classes
to prevent import errors during testing and CI runs.
"""

class QCAlgorithm:
    def __init__(self):
        pass
    
    def Debug(self, message):
        print(f"DEBUG: {message}")
    
    def Log(self, message):
        print(f"LOG: {message}")

class Symbol:
    def __init__(self, value=""):
        self.Value = value

class Resolution:
    Daily = "Daily"
    Hour = "Hour"
    Minute = "Minute"
    Second = "Second"

class OrderDirection:
    Buy = "Buy"
    Sell = "Sell"

class OrderType:
    Market = "Market"
    Limit = "Limit"

class SecurityType:
    Equity = "Equity"
    Forex = "Forex"
    Crypto = "Crypto"

class Market:
    USA = "USA"

def Debug(message):
    print(f"DEBUG: {message}")

def Log(message):
    print(f"LOG: {message}")

def Liquidate():
    pass

def SetCash(amount):
    pass

def AddEquity(symbol, resolution=None):
    return Symbol(symbol)

def AddForex(symbol, resolution=None):
    return Symbol(symbol)

def AddCrypto(symbol, resolution=None):
    return Symbol(symbol)

def History(symbol, periods, resolution=None):
    return []

def Schedule():
    pass

def DateRules():
    pass

def TimeRules():
    pass

__all__ = [
    'QCAlgorithm', 'Symbol', 'Resolution', 'OrderDirection', 'OrderType',
    'SecurityType', 'Market', 'Debug', 'Log', 'Liquidate', 'SetCash',
    'AddEquity', 'AddForex', 'AddCrypto', 'History', 'Schedule',
    'DateRules', 'TimeRules'
]
