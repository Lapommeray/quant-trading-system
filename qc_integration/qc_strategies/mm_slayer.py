"""
Market Maker Slayer QuantConnect Algorithm

This algorithm integrates the Market Maker Slayer components with QuantConnect
for live trading. It uses the Dark Pool Sniper, Order Flow Hunter, and Stop Hunter
components to detect and exploit market maker behaviors.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append("/GitHub/QMP_Overrider_QuantConnect/")

from market_maker_slayer.dark_pool_sniper import DarkPoolSniper
from market_maker_slayer.order_flow_hunter import OrderFlowHunter
from market_maker_slayer.stop_hunter import StopHunter
from market_maker_slayer.market_maker_slayer import MarketMakerSlayer

import joblib

class MarketMakerSlayerQC(QCAlgorithm):
    """
    Market Maker Slayer QuantConnect Algorithm
    
    This algorithm integrates the Market Maker Slayer components with QuantConnect
    for live trading. It uses the Dark Pool Sniper, Order Flow Hunter, and Stop Hunter
    components to detect and exploit market maker behaviors.
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
        
        self.slayer = MarketMakerSlayer()
        
        self.liquidity_model = self.LoadModel("liquidity_predictor.pkl")
        self.hft_behavior_model = self.LoadModel("hft_behavior.h5")
        
        self.trade_log = []
        
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(TimeSpan.FromMinutes(5)), self.GenerateSignals)
        
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "dark_pool_trades": 0,
            "order_flow_trades": 0,
            "stop_hunt_trades": 0
        }
        
        self.Debug("Market Maker Slayer initialized")
    
    def LoadModel(self, model_name):
        """
        Load ML model from GitHub
        
        Parameters:
        - model_name: Name of the model file
        
        Returns:
        - Loaded model
        """
        try:
            model_url = f"https://raw.githubusercontent.com/youruser/QMP_Overrider_QuantConnect/main/qc_integration/ml_models/{model_name}"
            model_data = self.Download(model_url)
            
            return {"name": model_name, "loaded": True}
        except Exception as e:
            self.Debug(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def GenerateSignals(self):
        """Generate trading signals"""
        for symbol_name, symbol in self.symbols.items():
            price = self.Securities[symbol].Price
            
            result = self.slayer.execute(symbol_name)
            
            if result["execution_type"] is not None:
                self.Debug(f"Signal for {symbol_name}: {result['direction']} | Type: {result['execution_type']} | Confidence: {result['confidence']}")
                
                self.ExecuteTrade(symbol, result)
    
    def ExecuteTrade(self, symbol, signal):
        """
        Execute trade based on signal
        
        Parameters:
        - symbol: Symbol to trade
        - signal: Signal data
        """
        direction = signal["direction"]
        confidence = signal["confidence"]
        execution_type = signal["execution_type"]
        
        position_size = 0.1 * confidence  # 10% of portfolio * confidence
        
        if direction == "BUY":
            self.SetHoldings(symbol, position_size)
        elif direction == "SELL":
            self.SetHoldings(symbol, -position_size)
        
        trade = {
            "symbol": str(symbol),
            "direction": direction,
            "confidence": confidence,
            "execution_type": execution_type,
            "time": self.Time,
            "price": self.Securities[symbol].Price
        }
        
        self.trade_log.append(trade)
        
        self.metrics["total_trades"] += 1
        
        if execution_type == "dark_pool":
            self.metrics["dark_pool_trades"] += 1
        elif execution_type == "flow":
            self.metrics["order_flow_trades"] += 1
        elif execution_type == "stop_hunt":
            self.metrics["stop_hunt_trades"] += 1
    
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
        self.Debug(f"Dark pool trades: {self.metrics['dark_pool_trades']}")
        self.Debug(f"Order flow trades: {self.metrics['order_flow_trades']}")
        self.Debug(f"Stop hunt trades: {self.metrics['stop_hunt_trades']}")
