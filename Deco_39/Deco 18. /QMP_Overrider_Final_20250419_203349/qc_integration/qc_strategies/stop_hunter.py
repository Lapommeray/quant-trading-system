"""
Stop Hunter QuantConnect Algorithm

This algorithm integrates the Stop Hunter component with QuantConnect
for live trading. It predicts where market makers will trigger stops
by analyzing stop clusters and market maker tactics.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append("/GitHub/QMP_Overrider_QuantConnect/")

from market_maker_slayer.stop_hunter import StopHunter, StopClusterDatabase, MarketMakerTactics

class StopHunterQC(QCAlgorithm):
    """
    Stop Hunter QuantConnect Algorithm
    
    This algorithm integrates the Stop Hunter component with QuantConnect
    for live trading. It predicts where market makers will trigger stops
    by analyzing stop clusters and market maker tactics.
    """
    
    def Initialize(self):
        """Initialize algorithm"""
        self.SetStartDate(2024, 1, 1)  # Set start date
        self.SetEndDate(2024, 4, 1)    # Set end date
        self.SetCash(100000)           # Set starting cash
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        
        self.symbols = {}
        self.symbols["SPY"] = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.symbols["QQQ"] = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.symbols["BTCUSD"] = self.AddCrypto("BTCUSD", Resolution.Minute).Symbol
        self.symbols["ETHUSD"] = self.AddCrypto("ETHUSD", Resolution.Minute).Symbol
        
        self.stop_hunter = StopHunter()
        
        self.trade_log = []
        
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(5)), self.GenerateSignals)
        
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        
        self.Debug("Stop Hunter initialized")
    
    def GenerateSignals(self):
        """Generate trading signals"""
        for symbol_name, symbol in self.symbols.items():
            price = self.Securities[symbol].Price
            
            result = self.stop_hunter.predict_hunt(symbol_name)
            
            if result:
                self.Debug(f"Signal for {symbol_name}: Fade {result['stop_type']} | Level: {result['stop_level']} | Confidence: {result['confidence']}")
                
                self.ExecuteTrade(symbol, result)
    
    def ExecuteTrade(self, symbol, signal):
        """
        Execute trade based on signal
        
        Parameters:
        - symbol: Symbol to trade
        - signal: Signal data
        """
        stop_type = signal["stop_type"]
        confidence = signal["confidence"]
        stop_level = signal["stop_level"]
        
        position_size = 0.1 * confidence  # 10% of portfolio * confidence
        
        if stop_type == "BUY":
            self.SetHoldings(symbol, -position_size)  # Sell to fade buy stops
        elif stop_type == "SELL":
            self.SetHoldings(symbol, position_size)   # Buy to fade sell stops
        
        trade = {
            "symbol": str(symbol),
            "direction": "SELL" if stop_type == "BUY" else "BUY",  # Fade the stop hunt
            "confidence": confidence,
            "stop_level": stop_level,
            "time": self.Time,
            "price": self.Securities[symbol].Price
        }
        
        self.trade_log.append(trade)
        
        self.metrics["total_trades"] += 1
    
    def OnOrderEvent(self, orderEvent):
        """
        Handle order events
        
        Parameters:
        - orderEvent: Order event data
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order {orderEvent.OrderId} filled: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
    
    def OnEndOfAlgorithm(self):
        """Handle end of algorithm"""
        self.Debug(f"Total trades: {self.metrics['total_trades']}")
        self.Debug(f"Winning trades: {self.metrics['winning_trades']}")
        self.Debug(f"Losing trades: {self.metrics['losing_trades']}")
