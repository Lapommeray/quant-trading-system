"""
Institutional Trading Strategy for QuantConnect

This algorithm integrates the institutional trading components with QuantConnect
for live trading. It uses the Enhanced Limit Order Book, Advanced Cointegration,
and Optimal Execution components for institutional-grade trading.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.append("/GitHub/quant-trading-system/")

from core.institutional_indicators import HestonVolatility, ML_RSI, OrderFlowImbalance, RegimeDetector
from arbitrage.advanced_cointegration import AdvancedCointegration
from execution.advanced import VWAPExecution, OptimalExecution
from qc_integration.conscious_intelligence.simplified_consciousness import SimplifiedConsciousIntelligenceLayer

class InstitutionalTradingQC(QCAlgorithm):
    """
    Institutional Trading Strategy for QuantConnect
    
    This algorithm integrates the institutional trading components with QuantConnect
    for live trading. It uses the Enhanced Limit Order Book, Advanced Cointegration,
    and Optimal Execution components for institutional-grade trading.
    """
    
    def Initialize(self):
        """Initialize algorithm"""
        self.SetStartDate(2024, 1, 1)  # Set start date
        self.SetEndDate(2024, 5, 1)    # Set end date
        self.SetCash(1000000)          # Set starting cash - institutional level
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        
        self.symbols = {}
        self.symbols["SPY"] = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.symbols["QQQ"] = self.AddEquity("QQQ", Resolution.Minute).Symbol
        self.symbols["AAPL"] = self.AddEquity("AAPL", Resolution.Minute).Symbol
        self.symbols["MSFT"] = self.AddEquity("MSFT", Resolution.Minute).Symbol
        
        self.heston = HestonVolatility(lookback=30)
        self.ml_rsi = ML_RSI(window=14, lookahead=5)
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.cointegration = AdvancedCointegration(lookback=252)
        
        self.consciousness = SimplifiedConsciousIntelligenceLayer(self)
        
        self.historical_volumes = {}
        
        self.trade_log = []
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "vwap_trades": 0,
            "optimal_trades": 0,
            "cointegration_trades": 0
        }
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.Every(TimeSpan.FromMinutes(15)), 
                        self.GenerateSignals)
        
        self.Schedule.On(self.DateRules.EveryDay(), 
                        self.TimeRules.BeforeMarketClose("SPY", 10), 
                        self.EndOfDayAnalysis)
        
        self.Debug("Institutional Trading Strategy initialized")
    
    def OnData(self, data):
        """
        Event handler for market data updates
        
        Parameters:
        - data: Slice object containing market data
        """
        for symbol_name, symbol in self.symbols.items():
            if symbol in data and data[symbol] is not None:
                if symbol_name not in self.historical_volumes:
                    self.historical_volumes[symbol_name] = pd.Series()
                
                bar = data[symbol]
                self.historical_volumes[symbol_name][self.Time] = bar.Volume
    
    def GenerateSignals(self):
        """Generate trading signals using institutional components"""
        prices = {}
        volatility_data = {}
        regime_data = {}
        ml_predictions = {}
        
        for symbol_name, symbol in self.symbols.items():
            history = self.History(symbol, 300, Resolution.Minute)
            if not history.empty:
                prices[symbol_name] = history['close'].unstack(level=0)
        
        if not prices:
            return
        
        price_matrix = pd.DataFrame({sym: prices[sym] for sym in prices})
        
        try:
            coint_result = self.cointegration.johansen_test(price_matrix)
            
            if coint_result.get('cointegrated', False):
                self.Debug(f"Cointegration detected: {coint_result['hedge_ratios']}")
                
                self.ExecuteCointegrationTrade(coint_result)
        except Exception as e:
            self.Debug(f"Cointegration analysis error: {str(e)}")
        
        for symbol_name, symbol in self.symbols.items():
            if symbol_name not in prices:
                continue
                
            price_series = prices[symbol_name]
            
            try:
                volatility = self.heston.calculate(price_series)
                volatility_data[symbol_name] = volatility.iloc[-1] if not volatility.empty else 0.2
                
                rsi = self.RSI(symbol, 14, Resolution.Minute)
                rsi_series = pd.Series(rsi.Current.Value, index=[self.Time])
                
                ml_prediction = self.ml_rsi.calculate(price_series, rsi_series)
                ml_predictions[symbol_name] = ml_prediction.iloc[-1] if not ml_prediction.empty else 0.0
                
                regime = self.regime_detector.calculate(price_series)
                regime_data[symbol_name] = regime.iloc[-1] if not regime.empty else 1
                
                signal = self.GenerateInstitutionalSignal(
                    symbol_name, 
                    volatility, 
                    ml_prediction, 
                    regime.iloc[-1] if not regime.empty else 1
                )
                
                if signal["direction"] != "HOLD":
                    self.Debug(f"Signal for {symbol_name}: {signal['direction']} | Confidence: {signal['confidence']}")
                    self.ExecuteTrade(symbol, signal)
                    
            except Exception as e:
                self.Debug(f"Error generating signal for {symbol_name}: {str(e)}")
        
        intention_field = self.consciousness.perceive_market_intention(
            prices, volatility_data, regime_data, ml_predictions
        )
        
        self.Debug(f"Consciousness Level: {self.consciousness.consciousness_level:.3f}")
        self.Debug(f"Market Direction Perception: {intention_field['market_direction']:.3f}")
    
    def GenerateInstitutionalSignal(self, symbol_name, volatility, ml_prediction, regime):
        """
        Generate trading signal based on institutional indicators
        
        Parameters:
        - symbol_name: Symbol name
        - volatility: Heston volatility
        - ml_prediction: ML-enhanced RSI prediction
        - regime: Market regime (0=low vol, 1=medium vol, 2=high vol)
        
        Returns:
        - Signal dictionary
        """
        direction = "HOLD"
        confidence = 0.0
        execution_type = None
        
        if ml_prediction.empty:
            return {"direction": direction, "confidence": confidence, "execution_type": execution_type}
        
        latest_prediction = ml_prediction.iloc[-1] if len(ml_prediction) > 0 else 0
        
        regime_factor = 1.0
        if regime == 0:  # Low volatility
            regime_factor = 0.8
        elif regime == 2:  # High volatility
            regime_factor = 0.6
        
        if latest_prediction > 0.01:  # Bullish
            direction = "BUY"
            confidence = min(abs(latest_prediction) * 5, 1.0) * regime_factor
            execution_type = "optimal" if volatility.iloc[-1] > 0.2 else "vwap"
        elif latest_prediction < -0.01:  # Bearish
            direction = "SELL"
            confidence = min(abs(latest_prediction) * 5, 1.0) * regime_factor
            execution_type = "optimal" if volatility.iloc[-1] > 0.2 else "vwap"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "execution_type": execution_type
        }
    
    def ExecuteTrade(self, symbol, signal):
        """
        Execute trade based on signal using institutional execution algorithms
        
        Parameters:
        - symbol: Symbol to trade
        - signal: Signal data
        """
        direction = signal["direction"]
        confidence = signal["confidence"]
        execution_type = signal["execution_type"]
        
        position_size = 0.1 * confidence  # 10% of portfolio * confidence
        
        if execution_type == "vwap":
            symbol_name = str(symbol).split()[0]
            if symbol_name in self.historical_volumes:
                vwap = VWAPExecution(self.historical_volumes[symbol_name])
                
                current_time = self.Time.strftime("%H:%M")
                end_time = (self.Time + timedelta(hours=2)).strftime("%H:%M")
                
                schedule = vwap.get_execution_schedule(
                    target_quantity=self.Portfolio.Cash * position_size / self.Securities[symbol].Price,
                    start_time=current_time,
                    end_time=end_time
                )
                
                self.Debug(f"VWAP Schedule for {symbol}: {schedule}")
                
                if current_time in schedule:
                    if direction == "BUY":
                        self.MarketOrder(symbol, schedule[current_time])
                    elif direction == "SELL":
                        self.MarketOrder(symbol, -schedule[current_time])
                
                self.metrics["vwap_trades"] += 1
            else:
                if direction == "BUY":
                    self.SetHoldings(symbol, position_size)
                elif direction == "SELL":
                    self.SetHoldings(symbol, -position_size)
        else:
            optimal = OptimalExecution(risk_aversion=1e-6)
            
            volatility = 0.2  # Default
            try:
                history = self.History(symbol, 30, Resolution.Daily)
                if not history.empty:
                    returns = np.log(history['close'].unstack(level=0)).diff().dropna()
                    volatility = returns.std() * np.sqrt(252)
            except:
                pass
            
            target_shares = self.Portfolio.Cash * position_size / self.Securities[symbol].Price
            
            strategy = optimal.solve_optimal_strategy(
                target_shares=target_shares,
                time_horizon=10,  # 10 periods
                volatility=volatility
            )
            
            self.Debug(f"Optimal execution for {symbol}: {strategy}")
            
            if 'optimal_trades' in strategy and len(strategy['optimal_trades']) > 0:
                first_slice = strategy['optimal_trades'][0]
                
                if direction == "BUY":
                    self.MarketOrder(symbol, first_slice)
                elif direction == "SELL":
                    self.MarketOrder(symbol, -first_slice)
                
                self.metrics["optimal_trades"] += 1
            else:
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
        
        self.consciousness.record_prediction(str(symbol), {
            "direction": direction,
            "confidence": confidence,
            "predicted_price": self.Securities[symbol].Price,
            "timestamp": self.Time
        })
        
        self.trade_log.append(trade)
        self.metrics["total_trades"] += 1
    
    def ExecuteCointegrationTrade(self, coint_result):
        """
        Execute pairs trade based on cointegration results
        
        Parameters:
        - coint_result: Cointegration test results
        """
        symbols = coint_result['symbols']
        hedge_ratios = coint_result['hedge_ratios']
        
        if len(symbols) < 2:
            return
        
        symbol_objects = [self.symbols[sym] for sym in symbols if sym in self.symbols]
        
        if len(symbol_objects) < 2:
            return
        
        total = sum(abs(ratio) for ratio in hedge_ratios)
        norm_ratios = [ratio / total for ratio in hedge_ratios]
        
        allocation = 0.2
        
        for i, symbol in enumerate(symbol_objects):
            direction = "BUY" if hedge_ratios[i] > 0 else "SELL"
            position_size = allocation * abs(norm_ratios[i])
            
            if direction == "BUY":
                self.SetHoldings(symbol, position_size)
            else:
                self.SetHoldings(symbol, -position_size)
        
        self.metrics["cointegration_trades"] += 1
        
        self.Debug(f"Executed cointegration trade: {symbols} with ratios {norm_ratios}")
    
    def EndOfDayAnalysis(self):
        """Perform end-of-day analysis with consciousness evolution"""
        daily_pnl = self.Portfolio.TotalProfit
        total_value = self.Portfolio.TotalPortfolioValue
        
        win_rate = self.metrics["winning_trades"] / max(1, self.metrics["total_trades"])
        max_drawdown = (total_value - self.Portfolio.Cash) / self.Portfolio.Cash if self.Portfolio.Cash > 0 else 0
        
        for symbol in self.symbols.values():
            if self.Portfolio[symbol].Invested:
                if self.Portfolio[symbol].UnrealizedProfit > 0:
                    self.metrics["winning_trades"] += 1
                else:
                    self.metrics["losing_trades"] += 1
        
        performance_metrics = {
            "total_profit": daily_pnl / self.Portfolio.Cash if self.Portfolio.Cash > 0 else 0,
            "win_rate": win_rate,
            "max_drawdown": abs(max_drawdown),
            "sharpe_ratio": daily_pnl / (self.Portfolio.TotalUnrealizedProfit * 0.01) if self.Portfolio.TotalUnrealizedProfit != 0 else 0,
            "total_trades": self.metrics["total_trades"]
        }
        
        self.consciousness.update_performance_metrics(performance_metrics)
        
        status = self.consciousness.get_status()
        validation = status["validation"]
        
        self.Debug(f"End of day P&L: ${daily_pnl}")
        self.Debug(f"Metrics: {self.metrics}")
        self.Debug(f"Consciousness Level: {status['consciousness_level']:.3f}")
        self.Debug(f"Awareness State: {status['awareness_state']}")
        self.Debug(f"Federal Outperformance: {status['federal_outperformance']:.2f}x")
        self.Debug(f"Validation: {validation['validated']} (confidence: {validation['confidence']:.4f})")
        self.Debug(f"Meets 200% Target: {validation['meets_target']}")
    
    def OnOrderEvent(self, orderEvent):
        """
        Handle order events with consciousness outcome recording
        
        Parameters:
        - orderEvent: Order event data
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order {orderEvent.OrderId} filled: {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
            
            symbol_str = str(orderEvent.Symbol)
            direction = "BUY" if orderEvent.FillQuantity > 0 else "SELL"
            
            self.consciousness.record_outcome(symbol_str, {
                "direction": direction,
                "actual_price": orderEvent.FillPrice,
                "fill_quantity": orderEvent.FillQuantity,
                "timestamp": self.Time
            })
    
    def OnEndOfAlgorithm(self):
        """Handle end of algorithm"""
        self.Debug(f"Total trades: {self.metrics['total_trades']}")
        self.Debug(f"Winning trades: {self.metrics['winning_trades']}")
        self.Debug(f"Losing trades: {self.metrics['losing_trades']}")
        self.Debug(f"VWAP trades: {self.metrics['vwap_trades']}")
        self.Debug(f"Optimal execution trades: {self.metrics['optimal_trades']}")
        self.Debug(f"Cointegration trades: {self.metrics['cointegration_trades']}")
