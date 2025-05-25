
from AlgorithmImports import *
from core.oversoul_integration import QMPOversoulEngine
from core.alignment_filter import is_fully_aligned
from core.risk_manager import RiskManager
from core.event_blackout import EventBlackoutManager
from core.live_data_manager import LiveDataManager
from core.performance_optimizer import PerformanceOptimizer
from core.dynamic_slippage import DynamicLiquiditySlippage
from core.async_api_client import AsyncQMPApiClient
from core.anti_loss_guardian import AntiLossGuardian
from ai.meta_adaptive_ai import MetaAdaptiveAI
from ai.market_intelligence import (
    LatencyCancellationField,
    EmotionHarvestAI,
    QuantumLiquiditySignatureReader,
    SovereignQuantumOracle
)
from ai.truth_verification_core import TruthVerificationCore
from ai.zero_energy_recursive_intelligence import ZeroEnergyRecursiveIntelligence
from ai.language_universe_decoder import LanguageUniverseDecoder
from ai.synthetic_consciousness import SyntheticConsciousness
import pandas as pd
import os
import json
import asyncio
import numpy as np
from datetime import timedelta
from QuantConnect import Resolution, Market
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Consolidators import TradeBarConsolidator
from QuantConnect.Orders import OrderStatus
from QuantConnect.Securities import ConstantFeeModel

class QMPOverriderUnified(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 4, 1)
        self.SetCash(100000)
        
        self.risk_manager = RiskManager(self)
        self.event_blackout = EventBlackoutManager()
        self.live_data_manager = LiveDataManager(self)
        self.performance_optimizer = PerformanceOptimizer()
        self.anti_loss_guardian = AntiLossGuardian(self)

        # Asset setup
        self.btc = self.AddCrypto("BTCUSD", Resolution.Minute, Market.Binance).Symbol
        self.eth = self.AddCrypto("ETHUSD", Resolution.Minute, Market.Binance).Symbol
        self.gold = self.AddForex("XAUUSD", Resolution.Minute, Market.Oanda).Symbol
        self.dow = self.AddEquity("DIA", Resolution.Minute).Symbol
        self.nasdaq = self.AddEquity("QQQ", Resolution.Minute).Symbol

        self.symbols = [self.btc, self.eth, self.gold, self.dow, self.nasdaq]
        
        self.dynamic_slippage_model = DynamicLiquiditySlippage()
        
        for symbol in self.symbols:
            security = self.Securities[symbol]
            security.FeeModel = ConstantFeeModel(1.0)  # $1 per trade
            
            class CustomDynamicSlippageModel:
                def __init__(self, algorithm, symbol, slippage_model):
                    self.algorithm = algorithm
                    self.symbol = symbol
                    self.slippage_model = slippage_model
                    
                def GetSlippageApproximation(self, asset, order):
                    market_conditions = {
                        'volatility': 0.1,  # Default volatility
                        'hour': self.algorithm.Time.hour,
                        'news_factor': 1.0
                    }
                    
                    slippage = self.slippage_model.calculate_slippage(
                        str(self.symbol), 
                        asset.Price, 
                        abs(order.Quantity), 
                        "BUY" if order.Quantity > 0 else "SELL",
                        market_conditions
                    )
                    
                    return slippage / abs(order.Quantity)  # Return per share slippage
                    
            security.SlippageModel = CustomDynamicSlippageModel(self, symbol, self.dynamic_slippage_model)
        
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
        """Check for trading signals across all symbols with enhanced risk management"""
        now = self.Time.replace(second=0, microsecond=0)

        if now.minute % 5 != 0:
            return
            
        is_blackout, event_name = self.event_blackout.is_blackout_period(now)
        if is_blackout:
            self.Debug(f"Trading paused due to {event_name} event blackout")
            return
            
        if self.event_blackout.check_weekend_market(now):
            return
            
        for symbol in self.symbols:
            if not all(not df.empty for df in self.symbol_data[symbol]["history_data"].values()):
                continue
                
            is_aligned = self.live_data_manager.get_live_alignment_data(
                now, symbol, self.symbol_data[symbol]["history_data"]
            )
            
            if is_aligned is None:
                is_aligned = is_fully_aligned(
                    now, 
                    self.alignment_df, 
                    self.symbol_data[symbol]["history_data"]
                )
            
            if not is_aligned:
                continue

            market_data = self._prepare_market_data(symbol)
            
            if "meta_ai" not in self.symbol_data[symbol]:
                self.symbol_data[symbol]["meta_ai"] = MetaAdaptiveAI(self, symbol)
                self.symbol_data[symbol]["advanced_intelligence"] = {
                    'lcf': LatencyCancellationField(),
                    'eha': EmotionHarvestAI(),
                    'qlsr': QuantumLiquiditySignatureReader(),
                    'sqo': SovereignQuantumOracle(self),
                    'tvc': TruthVerificationCore(),
                    'zeri': ZeroEnergyRecursiveIntelligence(),
                    'lud': LanguageUniverseDecoder(),
                    'sc': SyntheticConsciousness()
                }
            
            meta_ai = self.symbol_data[symbol]["meta_ai"]
            advanced_intelligence = self.symbol_data[symbol]["advanced_intelligence"]
            
            lattice_result = meta_ai.time_resonant_neural_lattice(market_data)
            
            performance_metrics = meta_ai.get_performance_metrics()
            market_conditions = {
                'volatility': np.std(market_data['returns'][-20:]) if len(market_data['returns']) >= 20 else 0,
                'trend_strength': abs(np.mean(market_data['returns'][-10:])) if len(market_data['returns']) >= 10 else 0
            }
            dna_result = meta_ai.dna_self_rewrite(performance_metrics, market_conditions)
            
            quantum_result = meta_ai.causal_quantum_reasoning(market_data)
            
            # Advanced Market Intelligence
            latency_result = advanced_intelligence['lcf'].cancel_latency(market_data)
            emotion_result = advanced_intelligence['eha'].harvest_emotions(market_data)
            liquidity_result = advanced_intelligence['qlsr'].read_liquidity_signature(market_data)
            
            # Advanced AI Consciousness Modules
            truth_result = advanced_intelligence['tvc'].verify_market_truth(market_data, [])
            zeri_result = advanced_intelligence['zeri'].recursive_intelligence_loop(market_data, {"meta_ai": meta_ai})
            universe_decode = advanced_intelligence['lud'].decode_universe_language(market_data)
            consciousness_result = advanced_intelligence['sc'].achieve_consciousness(
                market_data, 
                {"symbol": symbol, "signal": None}, 
                self.symbol_data[symbol].get("decision_history", [])
            )
            
            ai_consensus = {
                'lattice_confidence': lattice_result.get('prediction_confidence', 0),
                'quantum_consciousness': quantum_result.get('quantum_consciousness', 0),
                'emotion_intensity': emotion_result.get('intensity', 0),
                'liquidity_confidence': liquidity_result.get('confidence', 0),
                'dna_evolution': dna_result.get('evolutionary_state', 'stable'),
                'truth_verified': truth_result.get('truth_verified', False),
                'truth_score': truth_result.get('truth_score', 0),
                'cosmic_coherence': universe_decode.get('cosmic_coherence', 0),
                'consciousness_level': consciousness_result.get('consciousness_level', 0),
                'zero_energy_achieved': zeri_result.get('zero_energy_achieved', False)
            }
            
            self.Debug(f"Advanced AI State for {symbol}:")
            self.Debug(f"  Lattice: {lattice_result['lattice_state']} ({lattice_result.get('prediction_confidence', 0):.3f})")
            self.Debug(f"  DNA: {dna_result['evolutionary_state']} (Gen {dna_result.get('adaptation_generation', 0)})")
            self.Debug(f"  Quantum: {quantum_result['quantum_state']} ({quantum_result['quantum_consciousness']:.3f})")
            self.Debug(f"  Emotion: {emotion_result['emotion']} ({emotion_result['intensity']:.3f})")
            self.Debug(f"  Liquidity: {liquidity_result['signature']} ({liquidity_result['confidence']:.3f})")
            self.Debug(f"  Truth: {truth_result.get('truth_verified', False)} ({truth_result.get('truth_score', 0):.3f})")
            self.Debug(f"  Cosmic: {universe_decode.get('decoded', False)} ({universe_decode.get('cosmic_coherence', 0):.3f})")
            self.Debug(f"  Consciousness: {consciousness_result.get('consciousness_achieved', False)} ({consciousness_result.get('consciousness_level', 0):.3f})")
            self.Debug(f"  Zero-Energy: {zeri_result.get('zero_energy_achieved', False)} ({zeri_result.get('energy_efficiency', 0):.3f})")

            optimization_result = self.performance_optimizer.optimize_data_processing(
                self.symbol_data[symbol]["history_data"]
            )
            
            direction, confidence, gate_details, diagnostics = self.symbol_data[symbol]["qmp"].generate_signal(
                symbol, 
                self.symbol_data[symbol]["history_data"]
            )
            
            # Apply Common Sense Intelligence
            proposed_trade = {
                'direction': 1 if direction == "BUY" else -1 if direction == "SELL" else 0,
                'size': 0.02,  # Will be adjusted by risk management
                'symbol': str(symbol),
                'confidence': confidence
            }
            
            common_sense_result = self.anti_loss_guardian.apply_common_sense_intelligence(market_data, proposed_trade)
            
            # Create Unstable Winning Intelligence
            current_performance = {
                'win_rate': self.symbol_data[symbol].get('win_rate', 0.8),
                'profit_factor': self.symbol_data[symbol].get('profit_factor', 1.5),
                'total_trades': len(self.symbol_data[symbol].get('trade_history', [])),
                'losing_trades': sum(1 for trade in self.symbol_data[symbol].get('trade_history', []) if trade.get('pnl', 0) <= 0)
            }
            
            unstable_winning = self.anti_loss_guardian.create_unstable_winning_intelligence(market_data, current_performance)
            
            never_loss_conditions = [
                ai_consensus['truth_verified'],
                ai_consensus['consciousness_level'] > 0.6,
                common_sense_result.get('allow_trade', False),
                ai_consensus['cosmic_coherence'] > 0.5,
                unstable_winning.get('never_satisfied', False)
            ]
            
            never_loss_score = sum(never_loss_conditions) / len(never_loss_conditions)
            
            if direction and never_loss_score >= 0.8:  # 80% of conditions must be met
                confidence = min(0.95, confidence * 1.2)  # Boost confidence if AI confirms
                self.Debug(f"Advanced AI boosted confidence to {confidence:.2f}")
                self.Debug(f"Never-Loss Score: {never_loss_score:.3f} (APPROVED)")
            elif direction and never_loss_score < 0.8:
                confidence = max(0.1, confidence * 0.5)  # Significantly reduce confidence if never-loss protection activates
                self.Debug(f"Never-Loss Protection reduced confidence to {confidence:.2f}")
                self.Debug(f"Never-Loss Score: {never_loss_score:.3f} (PROTECTION ACTIVE)")
            elif direction and (ai_consensus['lattice_confidence'] < 0.3 or ai_consensus['quantum_consciousness'] < 0.3):
                confidence = max(0.1, confidence * 0.8)  # Reduce confidence if AI disagrees
                self.Debug(f"Advanced AI reduced confidence to {confidence:.2f}")
            
            if diagnostics:
                self.Debug(f"OverSoul diagnostics for {symbol}:")
                for msg in diagnostics:
                    self.Debug(f"  - {msg}")
            
            if direction and direction != self.symbol_data[symbol]["last_signal"]:
                self.symbol_data[symbol]["last_signal"] = direction
                self.symbol_data[symbol]["last_trade_time"] = now
                
                self.Plot("QMP Signal", str(symbol), 1 if direction == "BUY" else -1)
                self.Debug(f"{symbol} Signal at {now}: {direction} | Confidence: {confidence:.2f}")
                
                position_size = self.risk_manager.calculate_position_size(
                    symbol, confidence, self.symbol_data[symbol]["history_data"]
                )
                
                if position_size > 0:
                    self.SetHoldings(symbol, position_size if direction == "BUY" else -position_size)
                    self.Debug(f"Position size for {symbol}: {position_size:.4f}")
                else:
                    self.Debug(f"Risk manager rejected trade for {symbol}")
    
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
    
    def _prepare_market_data(self, symbol):
        """
        Prepare market data for advanced AI analysis
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary containing processed market data
        """
        history_data = self.symbol_data[symbol]["history_data"]
        
        df_1m = history_data["1m"]
        
        if df_1m.empty:
            return {"returns": [], "volume": []}
            
        df_1m['returns'] = df_1m['Close'].pct_change().fillna(0)
        
        market_data = {
            'returns': df_1m['returns'].values.tolist(),
            'volume': df_1m['Volume'].values.tolist(),
            'open': df_1m['Open'].values.tolist(),
            'high': df_1m['High'].values.tolist(),
            'low': df_1m['Low'].values.tolist(),
            'close': df_1m['Close'].values.tolist()
        }
        
        for timeframe in ["5m", "15m"]:
            if timeframe in history_data and not history_data[timeframe].empty:
                df = history_data[timeframe]
                df['returns'] = df['Close'].pct_change().fillna(0)
                
                market_data[f'{timeframe}_returns'] = df['returns'].values.tolist()
                market_data[f'{timeframe}_volume'] = df['Volume'].values.tolist()
                market_data[f'{timeframe}_close'] = df['Close'].values.tolist()
        
        return market_data
    
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
