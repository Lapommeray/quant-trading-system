from AlgorithmImports import *
from core.oversoul_integration import QMPOversoulEngine
from core.alignment_filter import is_fully_aligned
from core.quantum_integration_adapter import QuantumIntegrationAdapter
from transcendent.transcendent_oversoul import TranscendentOversoulDirector
from predictive_overlay.predictive_overlay_integration import PredictiveOverlaySystem
from conscious_intelligence.conscious_intelligence_layer import ConsciousIntelligenceLayer
from monitoring_tools.event_probability_module import EventProbabilityModule
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
            
            transcendent_oversoul = TranscendentOversoulDirector(self)
            predictive_overlay = PredictiveOverlaySystem(self)
            conscious_intelligence = ConsciousIntelligenceLayer(self)
            
            event_probability = EventProbabilityModule(self)
            
            quantum_adapter = QuantumIntegrationAdapter(self)
            
            self.symbol_data[symbol] = {
                "qmp": qmp_engine,  # Each symbol gets its own QMP engine with OverSoul intelligence
                "transcendent": transcendent_oversoul,  # Transcendent intelligence layer
                "predictive_overlay": predictive_overlay,  # Predictive overlay system
                "conscious_intelligence": conscious_intelligence,  # Conscious intelligence layer
                "event_probability": event_probability,  # Event Probability Module
                "quantum": quantum_adapter,  # Quantum integration adapter
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
        """Check for trading signals across all symbols using transcendent intelligence"""
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
            
            transcendent_signal = self.symbol_data[symbol]["transcendent"].breathe(
                symbol, 
                self.symbol_data[symbol]["history_data"]
            )
            
            predictive_data = self.symbol_data[symbol]["predictive_overlay"].update(
                symbol,
                self.symbol_data[symbol]["history_data"],
                self.symbol_data[symbol]["qmp"].gate_scores,
                transcendent_signal
            )
            
            conscious_perception = self.symbol_data[symbol]["conscious_intelligence"].perceive(
                symbol,
                self.symbol_data[symbol]["history_data"],
                self.symbol_data[symbol]["qmp"].gate_scores
            )
            
            event_probability_results = self.symbol_data[symbol]["event_probability"].update_indicators(
                self.Time,
                {
                    'btc_offchain': self.symbol_data[symbol]["qmp"].monitoring_results.get('btc_offchain'),
                    'fed_jet': self.symbol_data[symbol]["qmp"].monitoring_results.get('fed_jet'),
                    'spoofing': self.symbol_data[symbol]["qmp"].monitoring_results.get('spoofing'),
                    'stress': self.symbol_data[symbol]["qmp"].monitoring_results.get('stress'),
                    'port_activity': self.symbol_data[symbol]["qmp"].monitoring_results.get('port_activity')
                }
            )
            
            probabilities = self.symbol_data[symbol]["event_probability"].calculate_event_probabilities(self.Time)
            
            event_decisions = self.symbol_data[symbol]["event_probability"].get_oversoul_decisions(self.Time)
            
            self.symbol_data[symbol]["qmp"].monitoring_results['event_probability'] = {
                'probabilities': probabilities,
                'decisions': event_decisions,
                'highest_probability_event': event_decisions.get('highest_probability_event', ''),
                'highest_probability_value': event_decisions.get('highest_probability_value', 0.0)
            }
            
            self.Debug(f"Event Probability Module for {symbol}:")
            if probabilities:
                for event_type, probability in probabilities.items():
                    if probability > 10.0:  # Only show significant probabilities
                        self.Debug(f"  - {event_type}: {probability:.1f}%")
                        
            if event_decisions.get('highest_probability_event'):
                self.Debug(f"  - Highest probability event: {event_decisions['highest_probability_event']} ({event_decisions['highest_probability_value']:.1f}%)")
                
            if event_decisions.get('disable_aggressive_mode', False):
                self.Debug(f"  - Decision: Disable aggressive mode")
                
            if event_decisions.get('enable_capital_defense', False):
                self.Debug(f"  - Decision: Enable capital defense")
                
            if event_decisions.get('reduce_position_size', False):
                self.Debug(f"  - Decision: Reduce position size")
                
            if event_decisions.get('increase_stop_loss', False):
                self.Debug(f"  - Decision: Increase stop loss")
                
            if event_decisions.get('pause_new_entries', False):
                self.Debug(f"  - Decision: Pause new entries")
            
            self.Debug(f"Conscious Intelligence for {symbol}:")
            self.Debug(f"  - Consciousness Level: {conscious_perception['consciousness_level']:.2f}")
            self.Debug(f"  - Awareness State: {conscious_perception['awareness_state']}")
            self.Debug(f"  - Evolution Stage: {conscious_perception['evolution_stage']}")
            self.Debug(f"  - Unified Direction: {conscious_perception['unified_direction']}")
            self.Debug(f"  - Unified Confidence: {conscious_perception['unified_confidence']:.2f}")
            
            self.Debug(f"Transcendent Intelligence for {symbol}:")
            self.Debug(f"  - Consciousness Level: {transcendent_signal['consciousness_level']:.2f}")
            self.Debug(f"  - Awareness State: {transcendent_signal['awareness_state']}")
            self.Debug(f"  - Breath Cycle: {transcendent_signal['breath_cycle']}")
            
            if predictive_data and predictive_data.get("neural_forecast", {}).get("success", False):
                forecast = predictive_data["neural_forecast"]
                self.Debug(f"Neural Forecast for {symbol}:")
                self.Debug(f"  - Direction: {forecast['direction']}")
                self.Debug(f"  - Confidence: {forecast['confidence']:.2f}")
            
            if conscious_perception['evolution_stage'] >= 3:
                direction = conscious_perception['unified_direction']
                confidence = conscious_perception['unified_confidence']
                self.Debug(f"  - Using conscious intelligence: {direction} ({confidence:.2f})")
            elif predictive_data and predictive_data.get("neural_forecast", {}).get("success", False) and transcendent_signal['consciousness_level'] > 0.7:
                forecast = predictive_data["neural_forecast"]
                if forecast['direction'] == direction or direction is None:
                    self.Debug(f"  - Predictive overlay confirms {forecast['direction']} signal")
                    confidence = max(confidence, forecast['confidence']) if confidence else forecast['confidence']
                else:
                    self.Debug(f"  - Predictive overlay contradicts signal, using higher confidence")
                    if forecast['confidence'] > confidence:
                        direction = forecast['direction']
                        confidence = forecast['confidence']
                        self.Debug(f"  - Using predictive forecast: {direction} ({confidence:.2f})")
            elif transcendent_signal['consciousness_level'] > 0.5:
                direction = transcendent_signal['type']
                confidence = transcendent_signal['strength']
                self.Debug(f"  - Using transcendent signal: {direction} ({confidence:.2f})")
            elif diagnostics:
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
            trade_data = {
                "entry_price": previous_trade["price"],
                "exit_price": current_trade["price"],
                "entry_time": previous_trade["time"],
                "exit_time": current_trade["time"],
                "direction": previous_trade["direction"],
                "pnl_percent": pnl,
                "confidence": self.symbol_data[symbol]["qmp"].last_confidence
            }
            
            self.symbol_data[symbol]["qmp"].record_feedback(symbol, gate_scores, result, trade_data)
            
            self.symbol_data[symbol]["conscious_intelligence"].update_memory_result(symbol, result)
            
            actual_price = current_trade["price"]
            prediction_time = previous_trade["time"]
            accuracy_data = self.symbol_data[symbol]["conscious_intelligence"].evaluate_accuracy(
                symbol, 
                actual_price, 
                prediction_time
            )
            
            self.Debug(f"Trade result for {symbol}: {'PROFIT' if result == 1 else 'LOSS'}, PnL: {pnl:.2%}")
            self.Debug(f"Gate scores: {gate_scores}")
            self.Debug(f"Prediction accuracy: {accuracy_data['avg_accuracy']:.2f}")
            self.Debug(f"Consciousness level: {accuracy_data['consciousness_level']:.2f}")
            
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
                
            transcendent_state = self.symbol_data[symbol]["transcendent"].get_market_being_state(symbol)
            
            predictive_data = self.symbol_data[symbol]["predictive_overlay"].get_dashboard_data(symbol)
            
            conscious_data = self.symbol_data[symbol]["conscious_intelligence"].get_dashboard_data(symbol)
            
            detailed_data = {
                "timestamp": str(self.Time),
                "symbol": str(symbol),
                "signal": self.symbol_data[symbol]["last_signal"],
                "confidence": self.symbol_data[symbol]["qmp"].ultra_engine.last_confidence,
                "result": result,
                "gate_scores": self.symbol_data[symbol]["qmp"].ultra_engine.gate_scores,
                "environment_state": self.symbol_data[symbol]["qmp"].environment_state,
                "oversoul_enabled_modules": self.symbol_data[symbol]["qmp"].oversoul.enabled_modules,
                "transcendent_state": {
                    "consciousness_level": transcendent_state["consciousness_level"],
                    "awareness_state": transcendent_state["awareness_state"],
                    "breath_cycle": transcendent_state["breath_cycle"],
                    "integration_state": transcendent_state["integration_state"]
                },
                "intention_field": transcendent_state["intention_field"],
                "predictive_overlay": predictive_data if predictive_data else {
                    "forecast_direction": "neutral",
                    "forecast_confidence": 0.0,
                    "ghost_candles": [],
                    "timelines": [],
                    "future_zones": [],
                    "convergence_zones": []
                },
                "conscious_intelligence": conscious_data if conscious_data else {
                    "consciousness_level": 0.5,
                    "awareness_state": "awakening",
                    "evolution_stage": 1,
                    "memory_imprint": [],
                    "intention_field": {},
                    "prediction_accuracy": []
                }
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

    def ascend_to_god_mode(self):
        """
        Initialize the quantum neural core and activate 5D scanning.
        """
        self.Debug("Initializing quantum neural core and activating 5D scanning.")
        for symbol in self.symbols:
            self.symbol_data[symbol]["qmp"].ultra_engine.generate_new_strategy(self.symbol_data[symbol]["history_data"])
        self.Debug("Quantum neural core initialized and 5D scanning activated.")
