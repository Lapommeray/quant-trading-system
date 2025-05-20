"""
qc_bridge.py

QuantConnect Bridge for QMP Overrider

Provides integration with QuantConnect's Lean Engine for backtesting and live trading.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta

class QCBridge:
    """
    QuantConnect Bridge for QMP Overrider
    
    Provides integration with QuantConnect's Lean Engine for backtesting and live trading.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the QuantConnect Bridge
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.config = self._load_config()
        self.symbols = {}
        self.history_data = {}
        self.consolidators = {}
        self.last_signals = {}
        self.trades = []
        self.initialized = False
    
    def _load_config(self):
        """
        Load configuration
        
        Returns:
        - Dictionary with configuration
        """
        config = {
            "symbols": ["BTCUSD", "ETHUSD", "XAUUSD", "DIA", "QQQ"],
            "markets": {
                "BTCUSD": "crypto",
                "ETHUSD": "crypto",
                "XAUUSD": "forex",
                "DIA": "equity",
                "QQQ": "equity"
            },
            "resolutions": ["1m", "5m", "10m", "15m", "20m", "25m"],
            "position_size": 0.1,
            "max_positions": 5,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "log_signals": True,
            "log_trades": True
        }
        
        config_path = os.path.join(os.path.dirname(__file__), "qc_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    
                    for key, value in loaded_config.items():
                        config[key] = value
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return config
    
    def initialize(self, algorithm=None):
        """
        Initialize the QuantConnect Bridge
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        if algorithm:
            self.algorithm = algorithm
        
        if not self.algorithm:
            print("No algorithm provided")
            return False
        
        self._initialize_symbols()
        
        self._initialize_consolidators()
        
        self._schedule_signal_checking()
        
        self.initialized = True
        
        if self.algorithm:
            self.algorithm.Debug("QuantConnect Bridge: Initialized")
        
        return True
    
    def _initialize_symbols(self):
        """Initialize symbols"""
        if not self.algorithm:
            return
        
        for symbol_str in self.config["symbols"]:
            market_type = self.config["markets"].get(symbol_str, "equity")
            
            if market_type == "crypto":
                symbol = self.algorithm.AddCrypto(symbol_str, resolution=self.algorithm.Resolution.Minute).Symbol
            elif market_type == "forex":
                symbol = self.algorithm.AddForex(symbol_str, resolution=self.algorithm.Resolution.Minute).Symbol
            else:
                symbol = self.algorithm.AddEquity(symbol_str, resolution=self.algorithm.Resolution.Minute).Symbol
            
            self.symbols[symbol_str] = symbol
            self.history_data[symbol_str] = {}
            self.last_signals[symbol_str] = None
            
            if self.algorithm:
                self.algorithm.Debug(f"Added symbol: {symbol_str}")
    
    def _initialize_consolidators(self):
        """Initialize consolidators"""
        if not self.algorithm:
            return
        
        for symbol_str, symbol in self.symbols.items():
            self.consolidators[symbol_str] = {}
            
            for resolution in self.config["resolutions"]:
                if resolution == "1m":
                    continue
                
                minutes = int(resolution[:-1])
                
                consolidator = self.algorithm.TradeBarConsolidator(timedelta(minutes=minutes))
                consolidator.DataConsolidated += self._on_data_consolidated
                
                self.algorithm.SubscriptionManager.AddConsolidator(symbol, consolidator)
                
                self.consolidators[symbol_str][resolution] = consolidator
                
                self.history_data[symbol_str][resolution] = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
                
                if self.algorithm:
                    self.algorithm.Debug(f"Added {resolution} consolidator for {symbol_str}")
    
    def _on_data_consolidated(self, sender, bar):
        """
        Handler for consolidated data
        
        Parameters:
        - sender: Consolidator that triggered the event
        - bar: Consolidated bar
        """
        if not self.algorithm:
            return
        
        for symbol_str, consolidators in self.consolidators.items():
            for resolution, consolidator in consolidators.items():
                if sender == consolidator:
                    bar_data = pd.DataFrame({
                        "Open": [bar.Open],
                        "High": [bar.High],
                        "Low": [bar.Low],
                        "Close": [bar.Close]
                    }, index=[bar.EndTime])
                    
                    self.history_data[symbol_str][resolution] = pd.concat([
                        self.history_data[symbol_str][resolution],
                        bar_data
                    ])
                    
                    if len(self.history_data[symbol_str][resolution]) > 100:
                        self.history_data[symbol_str][resolution] = self.history_data[symbol_str][resolution].iloc[-100:]
                    
                    return
    
    def _schedule_signal_checking(self):
        """Schedule signal checking"""
        if not self.algorithm:
            return
        
        self.algorithm.Schedule.On(
            self.algorithm.DateRules.EveryDay(),
            self.algorithm.TimeRules.EveryMinute,
            self._check_signals
        )
    
    def _check_signals(self):
        """Check signals"""
        if not self.algorithm:
            return
        
        now = self.algorithm.Time.replace(second=0, microsecond=0)
        
        if now.minute % 25 != 0:
            return
        
        for symbol_str, symbol in self.symbols.items():
            if not self._has_enough_data(symbol_str):
                continue
            
            market_state = self._get_market_state(symbol_str)
            
            signal = self._get_signal(symbol_str, market_state)
            
            if signal:
                self._process_signal(symbol_str, symbol, signal)
    
    def _has_enough_data(self, symbol_str):
        """
        Check if we have enough data for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        
        Returns:
        - True if we have enough data, False otherwise
        """
        for resolution in self.config["resolutions"]:
            if resolution == "1m":
                continue
            
            if resolution not in self.history_data[symbol_str]:
                return False
            
            if self.history_data[symbol_str][resolution].empty:
                return False
        
        return True
    
    def _get_market_state(self, symbol_str):
        """
        Get market state for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        
        Returns:
        - Dictionary with market state
        """
        if not self.algorithm:
            return {}
        
        symbol = self.symbols[symbol_str]
        
        price = self.algorithm.Securities[symbol].Price
        
        history = {}
        
        for resolution in self.config["resolutions"]:
            if resolution == "1m":
                continue
            
            if resolution in self.history_data[symbol_str]:
                history[resolution] = self.history_data[symbol_str][resolution]
        
        market_state = {
            "symbol": symbol_str,
            "price": price,
            "timestamp": self.algorithm.Time,
            "history": history
        }
        
        if symbol_str in ["BTCUSD", "ETHUSD"]:
            market_state["volatility"] = self._calculate_volatility(symbol_str)
            market_state["trend"] = self._calculate_trend(symbol_str)
            market_state["volume"] = self._calculate_relative_volume(symbol_str)
        
        elif symbol_str in ["XAUUSD"]:
            market_state["volatility"] = self._calculate_volatility(symbol_str)
            market_state["trend"] = self._calculate_trend(symbol_str)
            market_state["correlation"] = self._calculate_correlation(symbol_str, "BTCUSD")
        
        elif symbol_str in ["DIA", "QQQ"]:
            market_state["volatility"] = self._calculate_volatility(symbol_str)
            market_state["trend"] = self._calculate_trend(symbol_str)
            market_state["volume"] = self._calculate_relative_volume(symbol_str)
            market_state["correlation"] = self._calculate_correlation(symbol_str, "BTCUSD")
        
        return market_state
    
    def _calculate_volatility(self, symbol_str):
        """
        Calculate volatility for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        
        Returns:
        - Volatility value
        """
        if "15m" not in self.history_data[symbol_str]:
            return 0.0
        
        history = self.history_data[symbol_str]["15m"]
        
        if history.empty:
            return 0.0
        
        returns = history["Close"].pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        volatility = returns.std() * 100.0
        
        return volatility
    
    def _calculate_trend(self, symbol_str):
        """
        Calculate trend for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        
        Returns:
        - Trend value (-1.0 to 1.0)
        """
        if "15m" not in self.history_data[symbol_str]:
            return 0.0
        
        history = self.history_data[symbol_str]["15m"]
        
        if history.empty or len(history) < 2:
            return 0.0
        
        first_price = history["Close"].iloc[0]
        last_price = history["Close"].iloc[-1]
        
        trend = (last_price - first_price) / first_price
        
        trend = max(-1.0, min(1.0, trend * 10.0))
        
        return trend
    
    def _calculate_relative_volume(self, symbol_str):
        """
        Calculate relative volume for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        
        Returns:
        - Relative volume value
        """
        if not self.algorithm:
            return 1.0
        
        symbol = self.symbols[symbol_str]
        
        volume = self.algorithm.Securities[symbol].Volume
        
        avg_volume = self.algorithm.Securities[symbol].AverageVolume
        
        if avg_volume == 0:
            return 1.0
        
        relative_volume = volume / avg_volume
        
        return relative_volume
    
    def _calculate_correlation(self, symbol_str1, symbol_str2):
        """
        Calculate correlation between two symbols
        
        Parameters:
        - symbol_str1: First symbol string
        - symbol_str2: Second symbol string
        
        Returns:
        - Correlation value (-1.0 to 1.0)
        """
        if "15m" not in self.history_data[symbol_str1] or "15m" not in self.history_data[symbol_str2]:
            return 0.0
        
        history1 = self.history_data[symbol_str1]["15m"]
        history2 = self.history_data[symbol_str2]["15m"]
        
        if history1.empty or history2.empty or len(history1) < 2 or len(history2) < 2:
            return 0.0
        
        returns1 = history1["Close"].pct_change().dropna()
        returns2 = history2["Close"].pct_change().dropna()
        
        common_index = returns1.index.intersection(returns2.index)
        
        if len(common_index) < 2:
            return 0.0
        
        returns1 = returns1.loc[common_index]
        returns2 = returns2.loc[common_index]
        
        correlation = returns1.corr(returns2)
        
        return correlation
    
    def _get_signal(self, symbol_str, market_state):
        """
        Get signal for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        - market_state: Dictionary with market state
        
        Returns:
        - Dictionary with signal information
        """
        if not self.algorithm:
            return None
        
        oversoul = self._get_oversoul_director()
        
        if not oversoul:
            return None
        
        signal = oversoul.get_signal(market_state)
        
        if self.config["log_signals"] and signal:
            self._log_signal(symbol_str, signal)
        
        return signal
    
    def _get_oversoul_director(self):
        """
        Get OversoulDirector instance
        
        Returns:
        - OversoulDirector instance
        """
        if not self.algorithm:
            return None
        
        if hasattr(self.algorithm, "oversoul"):
            return self.algorithm.oversoul
        
        try:
            from Core.OversoulDirector.main import OversoulDirector
            
            oversoul = OversoulDirector(self.algorithm)
            
            self.algorithm.oversoul = oversoul
            
            return oversoul
        except ImportError:
            self.algorithm.Debug("Error importing OversoulDirector")
            return None
    
    def _process_signal(self, symbol_str, symbol, signal):
        """
        Process signal for a symbol
        
        Parameters:
        - symbol_str: Symbol string
        - symbol: Symbol object
        - signal: Dictionary with signal information
        """
        if not self.algorithm:
            return
        
        if not signal or "direction" not in signal:
            return
        
        direction = signal["direction"]
        confidence = signal.get("confidence", 0.5)
        
        if direction == self.last_signals.get(symbol_str):
            return
        
        self.last_signals[symbol_str] = direction
        
        position_size = self.config["position_size"] * confidence
        
        if direction == "BUY":
            self.algorithm.SetHoldings(symbol, position_size)
            
            self._log_trade(symbol_str, "BUY", position_size, signal)
            
            if self.algorithm:
                self.algorithm.Debug(f"BUY signal for {symbol_str} | Confidence: {confidence:.2f}")
        
        elif direction == "SELL":
            self.algorithm.SetHoldings(symbol, -position_size)
            
            self._log_trade(symbol_str, "SELL", -position_size, signal)
            
            if self.algorithm:
                self.algorithm.Debug(f"SELL signal for {symbol_str} | Confidence: {confidence:.2f}")
        
        elif direction == "NEUTRAL":
            self.algorithm.Liquidate(symbol)
            
            self._log_trade(symbol_str, "NEUTRAL", 0.0, signal)
            
            if self.algorithm:
                self.algorithm.Debug(f"NEUTRAL signal for {symbol_str}")
    
    def _log_signal(self, symbol_str, signal):
        """
        Log signal
        
        Parameters:
        - symbol_str: Symbol string
        - signal: Dictionary with signal information
        """
        if not self.algorithm:
            return
        
        signal_log_path = os.path.join(self.algorithm.DataFolder, "data", "signal_log.csv")
        
        os.makedirs(os.path.dirname(signal_log_path), exist_ok=True)
        
        file_exists = os.path.exists(signal_log_path)
        
        with open(signal_log_path, "a") as f:
            if not file_exists:
                f.write("timestamp,symbol,direction,confidence,modules\n")
            
            timestamp = self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S")
            direction = signal.get("direction", "NEUTRAL")
            confidence = signal.get("confidence", 0.0)
            modules = ",".join(signal.get("modules", {}).keys())
            
            f.write(f"{timestamp},{symbol_str},{direction},{confidence:.4f},{modules}\n")
    
    def _log_trade(self, symbol_str, direction, position_size, signal):
        """
        Log trade
        
        Parameters:
        - symbol_str: Symbol string
        - direction: Trade direction
        - position_size: Position size
        - signal: Dictionary with signal information
        """
        if not self.algorithm:
            return
        
        if not self.config["log_trades"]:
            return
        
        trade_log_path = os.path.join(self.algorithm.DataFolder, "data", "trade_log.csv")
        
        os.makedirs(os.path.dirname(trade_log_path), exist_ok=True)
        
        file_exists = os.path.exists(trade_log_path)
        
        with open(trade_log_path, "a") as f:
            if not file_exists:
                f.write("timestamp,symbol,direction,position_size,confidence,modules\n")
            
            timestamp = self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S")
            confidence = signal.get("confidence", 0.0)
            modules = ",".join(signal.get("modules", {}).keys())
            
            f.write(f"{timestamp},{symbol_str},{direction},{position_size:.4f},{confidence:.4f},{modules}\n")
        
        trade = {
            "timestamp": self.algorithm.Time,
            "symbol": symbol_str,
            "direction": direction,
            "position_size": position_size,
            "confidence": confidence,
            "signal": signal
        }
        
        self.trades.append(trade)
    
    def get_status(self):
        """
        Get QuantConnect Bridge status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "symbols": list(self.symbols.keys()),
            "last_signals": self.last_signals,
            "trade_count": len(self.trades)
        }
