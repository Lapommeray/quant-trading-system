"""
Order Flow Hunter QuantConnect Algorithm

This algorithm integrates the Order Flow Hunter component with QuantConnect
for live trading. It detects and exploits order flow imbalances by analyzing
order book data and predicting HFT reactions to liquidity gaps.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append("/GitHub/QMP_Overrider_QuantConnect/")

from market_maker_slayer.order_flow_hunter import OrderFlowHunter, OrderBookImbalanceScanner, HTFBehaviorDatabase

class OrderFlowHunterQC(QCAlgorithm):
    """
    Order Flow Hunter QuantConnect Algorithm
    
    This algorithm integrates the Order Flow Hunter component with QuantConnect
    for live trading. It detects and exploits order flow imbalances by analyzing
    order book data and predicting HFT reactions to liquidity gaps.
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
        
        self.flow_hunter = OrderFlowHunter()
        
        self.trade_log = []
        
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(5)), self.GenerateSignals)
        
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        
        self.Debug("Order Flow Hunter initialized")
    
    def GenerateSignals(self):
        """Generate trading signals"""
        for symbol_name, symbol in self.symbols.items():
            price = self.Securities[symbol].Price
            
            result = self.flow_hunter.detect_imbalance(symbol_name)
            
            if result:
                self.Debug(f"Signal for {symbol_name}: {result['expected_hft_action']} | Imbalance: {result['imbalance_ratio']} | Confidence: {result['confidence']}")
                
                self.ExecuteTrade(symbol, result)
    
    def ExecuteTrade(self, symbol, signal):
        """
        Execute trade based on signal
        
        Parameters:
        - symbol: Symbol to trade
        - signal: Signal data
        """
        direction = signal["expected_hft_action"]
        confidence = signal["confidence"]
        imbalance = signal["imbalance_ratio"]
        
        position_size = 0.1 * confidence  # 10% of portfolio * confidence
        
        if direction == "BUY":
            self.SetHoldings(symbol, position_size)
        elif direction == "SELL":
            self.SetHoldings(symbol, -position_size)
        
        trade = {
            "symbol": str(symbol),
            "direction": direction,
            "confidence": confidence,
            "imbalance": imbalance,
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
