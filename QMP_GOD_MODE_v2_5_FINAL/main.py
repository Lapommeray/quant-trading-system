
from AlgorithmImports import *
from core.oversoul_integration import QMPOversoulEngine
from core.alignment_filter import is_fully_aligned
import pandas as pd
import os
import json
from datetime import timedelta
from QuantConnect import Resolution, Market
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Consolidators import TradeBarConsolidator
from QuantConnect.Orders import OrderStatus

class QMPOverriderUnified(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 4, 1)
        self.SetCash(100000)

        # Asset setup
        self.btc = self.AddCrypto("BTCUSD", Resolution.Minute, Market.Binance).Symbol
        self.eth = self.AddCrypto("ETHUSD", Resolution.Minute, Market.Binance).Symbol
        self.gold = self.AddForex("XAUUSD", Resolution.Minute, Market.Oanda).Symbol
        self.dow = self.AddEquity("DIA", Resolution.Minute).Symbol
        self.nasdaq = self.AddEquity("QQQ", Resolution.Minute).Symbol

        self.symbols = [self.btc, self.eth, self.gold, self.dow, self.nasdaq]
        
        self.symbol_data = {}
        for symbol in self.symbols:
            qmp_engine = QMPOversoulEngine(self)
            qmp_engine.ultra_engine.confidence_threshold = 0.65  # Minimum confidence to generate signal
            qmp_engine.ultra_engine.min_gate_score = 0.5  # Minimum score for each gate to pass
            
            self.symbol_data[symbol] = {
                "qmp": qmp_engine,  # Each symbol gets its own QMP engine with OverSoul intelligence
                "last_signal": None,
                "position_size": 0.0,
                "last_trade_time": None,
                "trades": [],
                "history_data": {
                    "1m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                    "5m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                    "10m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                    "15m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                    "20m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                    "25m": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
                }
            }

        self.alignment_df = self.LoadAlignmentCSV("alignment_blocks.csv")
        
        self.consolidators = {}
        for symbol in self.symbols:
            self.consolidators[symbol] = {
                "1m": None,
                "5m": TradeBarConsolidator(timedelta(minutes=5)),
                "10m": TradeBarConsolidator(timedelta(minutes=10)),
                "15m": TradeBarConsolidator(timedelta(minutes=15)),
                "20m": TradeBarConsolidator(timedelta(minutes=20)),
                "25m": TradeBarConsolidator(timedelta(minutes=25))
            }
            
            for timeframe, consolidator in self.consolidators[symbol].items():
                if timeframe != "1m":  # 1m is already the base timeframe
                    consolidator.DataConsolidated += self.OnDataConsolidated
                    self.SubscriptionManager.AddConsolidator(symbol, consolidator)

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EveryMinute, self.CheckSignals)
        
        self.trade_log_path = os.path.join(self.DataFolder, "data", "signal_feedback_log.csv")
        if not os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, "w") as f:
                f.write("signal,confidence,timestamp,reentry,vibration_alignment,symbol,result\n")

    def LoadAlignmentCSV(self, filename):
        path = os.path.join(self.DataFolder, "data", filename)
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["Time"])
        return pd.DataFrame(columns=["Time", "Base Direction", "Directions", "Label"])
    
    def OnDataConsolidated(self, sender, bar):
        """
        Handler for consolidated data for each timeframe
        Efficiently stores consolidated bars in the appropriate history DataFrame
        """
        for symbol in self.symbols:
            for timeframe, consolidator in self.consolidators[symbol].items():
                if sender == consolidator:
                    bar_data = pd.DataFrame({
                        "Open": [bar.Open],
                        "High": [bar.High],
                        "Low": [bar.Low],
                        "Close": [bar.Close],
                        "Volume": [bar.Volume]
                    }, index=[bar.EndTime])
                    
                    history_df = self.symbol_data[symbol]["history_data"][timeframe]
                    self.symbol_data[symbol]["history_data"][timeframe] = pd.concat([
                        history_df, bar_data
                    ]).tail(200)
                    
                    if timeframe == "1m" and len(history_df) % 50 == 0:
                        self.Debug(f"Collected {len(history_df)} bars for {symbol} on {timeframe}")
                    
                    return

    def OnData(self, data):
        """
        Event handler for market data updates
        Stores 1-minute bars directly from data feed
        """
        for symbol in self.symbols:
            if symbol in data and data[symbol] is not None:
                bar = data[symbol]
                
                bar_data = pd.DataFrame({
                    "Open": [bar.Open],
                    "High": [bar.High],
                    "Low": [bar.Low],
                    "Close": [bar.Close],
                    "Volume": [bar.Volume]
                }, index=[self.Time])
                
                self.symbol_data[symbol]["history_data"]["1m"] = pd.concat([
                    self.symbol_data[symbol]["history_data"]["1m"], 
                    bar_data
                ]).tail(200)
    
    def CheckSignals(self):
        """Check for trading signals across all symbols"""
        now = self.Time.replace(second=0, microsecond=0)

        if now.minute % 5 != 0:
            return
            
        for symbol in self.symbols:
            if not all(not df.empty for df in self.symbol_data[symbol]["history_data"].values()):
                continue
                
            is_aligned = is_fully_aligned(
                now, 
                self.alignment_df, 
                self.symbol_data[symbol]["history_data"]
            )
            
            if not is_aligned:
                continue

            direction, confidence, gate_details, diagnostics = self.symbol_data[symbol]["qmp"].generate_signal(
                symbol, 
                self.symbol_data[symbol]["history_data"]
            )
            
            if diagnostics:
                self.Debug(f"OverSoul diagnostics for {symbol}:")
                for msg in diagnostics:
                    self.Debug(f"  - {msg}")
            
            if direction and direction != self.symbol_data[symbol]["last_signal"]:
                self.symbol_data[symbol]["last_signal"] = direction
                self.symbol_data[symbol]["last_trade_time"] = now
                
                self.Plot("QMP Signal", str(symbol), 1 if direction == "BUY" else -1)
                self.Debug(f"{symbol} Signal at {now}: {direction} | Confidence: {confidence:.2f}")
                
                position_size = 1.0 * confidence
                self.SetHoldings(symbol, position_size if direction == "BUY" else -position_size)
    
    def OnOrderEvent(self, orderEvent):
        """Event handler for order status updates"""
        if orderEvent.Status != OrderStatus.Filled:
            return
            
        symbol = None
        for sym in self.symbols:
            if orderEvent.Symbol == sym:
                symbol = sym
                break
                
        if symbol is None:
            return
            
        trade = {
            "time": self.Time,
            "symbol": str(symbol),
            "direction": "BUY" if orderEvent.FillQuantity > 0 else "SELL",
            "price": orderEvent.FillPrice,
            "quantity": abs(orderEvent.FillQuantity),
            "order_id": orderEvent.OrderId
        }
        
        self.symbol_data[symbol]["trades"].append(trade)
        
        trades = self.symbol_data[symbol]["trades"]
        if len(trades) < 2:
            return
            
        current_trade = trades[-1]
        previous_trade = trades[-2]
        
        if current_trade["direction"] == previous_trade["direction"]:
            return  # Not a closing trade
            
        if previous_trade["direction"] == "BUY":
            pnl = (current_trade["price"] - previous_trade["price"]) / previous_trade["price"]
        else:
            pnl = (previous_trade["price"] - current_trade["price"]) / previous_trade["price"]
        
        result = 1 if pnl > 0 else 0
        
        gate_scores = self.symbol_data[symbol]["qmp"].gate_scores
        if gate_scores:
            self.symbol_data[symbol]["qmp"].record_feedback(gate_scores, result)
            
            self.Debug(f"Trade result for {symbol}: {'PROFIT' if result == 1 else 'LOSS'}, PnL: {pnl:.2%}")
            self.Debug(f"Gate scores: {gate_scores}")
            
            self.LogTradeResult(symbol, result)
    
    def LogTradeResult(self, symbol, result):
        """
        Logs trade results to CSV file for future analysis
        
        Parameters:
        - symbol: Trading symbol
        - result: 1 for profit, 0 for loss
        """
        try:
            detailed_log_path = os.path.join(self.DataFolder, "data", "detailed_signal_log.json")
            
            with open(self.trade_log_path, "a") as f:
                data = {
                    "signal": self.symbol_data[symbol]["last_signal"],
                    "confidence": self.symbol_data[symbol]["qmp"].ultra_engine.last_confidence,
                    "timestamp": self.Time,
                    "reentry": 0,  # Not implemented yet
                    "vibration_alignment": 1,  # Always 1 since we check alignment
                    "symbol": str(symbol),
                    "result": result
                }
                
                line = (f"{data['signal']},{data['confidence']:.4f},{data['timestamp']},"
                       f"{data['reentry']},{data['vibration_alignment']},{data['symbol']},{data['result']}\n")
                
                f.write(line)
                
            detailed_data = {
                "timestamp": str(self.Time),
                "symbol": str(symbol),
                "signal": self.symbol_data[symbol]["last_signal"],
                "confidence": self.symbol_data[symbol]["qmp"].ultra_engine.last_confidence,
                "result": result,
                "gate_scores": self.symbol_data[symbol]["qmp"].ultra_engine.gate_scores,
                "environment_state": self.symbol_data[symbol]["qmp"].environment_state,
                "oversoul_enabled_modules": self.symbol_data[symbol]["qmp"].oversoul.enabled_modules
            }
            
            try:
                if os.path.exists(detailed_log_path):
                    with open(detailed_log_path, "r") as f:
                        log_data = json.load(f)
                else:
                    log_data = []
                    
                log_data.append(detailed_data)
                
                with open(detailed_log_path, "w") as f:
                    json.dump(log_data, f, indent=2)
            except Exception as e:
                self.Debug(f"Error writing detailed log: {e}")
                
        except Exception as e:
            self.Debug(f"Error logging trade result: {e}")
