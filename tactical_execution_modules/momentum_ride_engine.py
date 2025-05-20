"""
Momentum Ride Engine

Detects parabolic price events (e.g., meme rallies) for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class MomentumRideEngine:
    """
    Detects parabolic price events and rides momentum.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Momentum Ride Engine.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("MomentumRideEngine")
        self.logger.setLevel(logging.INFO)
        
        self.lookback_periods = {
            "short": 5,    # 5 bars for short-term momentum
            "medium": 20,  # 20 bars for medium-term momentum
            "long": 50     # 50 bars for long-term momentum
        }
        
        self.volatility_window = 20  # Window for volatility calculation
        self.std_dev_threshold = 2.0  # Standard deviation threshold for parabolic moves
        self.volume_ratio_threshold = 3.0  # Volume ratio threshold for unusual volume
        
        self.price_data = {}  # Symbol -> price data
        self.volume_data = {}  # Symbol -> volume data
        self.momentum_scores = {}  # Symbol -> momentum scores
        self.parabolic_events = {}  # Symbol -> parabolic event data
        
        self.active_rides = {}  # Symbol -> active ride data
        self.completed_rides = []  # List of completed rides
        
    def update(self, symbol, current_price, current_volume, current_time):
        """
        Update the engine with latest price and volume data.
        
        Parameters:
        - symbol: The trading symbol
        - current_price: Current price
        - current_volume: Current volume
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing momentum analysis results
        """
        if symbol not in self.price_data:
            self.price_data[symbol] = []
            self.volume_data[symbol] = []
            self.momentum_scores[symbol] = {
                "short": 0.0,
                "medium": 0.0,
                "long": 0.0,
                "combined": 0.0
            }
            self.parabolic_events[symbol] = {
                "is_parabolic": False,
                "duration": 0,
                "magnitude": 0.0,
                "volume_surge": 0.0
            }
        
        self.price_data[symbol].append(current_price)
        self.volume_data[symbol].append(current_volume)
        
        max_lookback = max(self.lookback_periods.values()) + self.volatility_window
        if len(self.price_data[symbol]) > max_lookback:
            self.price_data[symbol] = self.price_data[symbol][-max_lookback:]
            self.volume_data[symbol] = self.volume_data[symbol][-max_lookback:]
        
        self._calculate_momentum_scores(symbol)
        
        self._detect_parabolic_events(symbol)
        
        self._manage_active_rides(symbol, current_price, current_time)
        
        signal = self._generate_signal(symbol)
        
        return {
            "symbol": symbol,
            "momentum_scores": self.momentum_scores[symbol],
            "parabolic_event": self.parabolic_events[symbol],
            "active_ride": self.active_rides.get(symbol, None),
            "signal": signal
        }
        
    def _calculate_momentum_scores(self, symbol):
        """
        Calculate momentum scores for different timeframes.
        
        Parameters:
        - symbol: The trading symbol
        """
        price_data = self.price_data[symbol]
        
        if len(price_data) < max(self.lookback_periods.values()):
            return
        
        current_price = price_data[-1]
        
        short_lookback = self.lookback_periods["short"]
        if len(price_data) >= short_lookback:
            short_price = price_data[-short_lookback]
            short_return = (current_price / short_price) - 1.0
            short_annualized = ((1.0 + short_return) ** (252 / short_lookback)) - 1.0
            self.momentum_scores[symbol]["short"] = short_annualized
        
        medium_lookback = self.lookback_periods["medium"]
        if len(price_data) >= medium_lookback:
            medium_price = price_data[-medium_lookback]
            medium_return = (current_price / medium_price) - 1.0
            medium_annualized = ((1.0 + medium_return) ** (252 / medium_lookback)) - 1.0
            self.momentum_scores[symbol]["medium"] = medium_annualized
        
        long_lookback = self.lookback_periods["long"]
        if len(price_data) >= long_lookback:
            long_price = price_data[-long_lookback]
            long_return = (current_price / long_price) - 1.0
            long_annualized = ((1.0 + long_return) ** (252 / long_lookback)) - 1.0
            self.momentum_scores[symbol]["long"] = long_annualized
        
        if all(score != 0.0 for score in self.momentum_scores[symbol].values()):
            self.momentum_scores[symbol]["combined"] = (
                self.momentum_scores[symbol]["short"] * 0.5 +
                self.momentum_scores[symbol]["medium"] * 0.3 +
                self.momentum_scores[symbol]["long"] * 0.2
            )
        
    def _detect_parabolic_events(self, symbol):
        """
        Detect parabolic price events.
        
        Parameters:
        - symbol: The trading symbol
        """
        price_data = self.price_data[symbol]
        volume_data = self.volume_data[symbol]
        
        if len(price_data) < self.volatility_window + 1:
            return
        
        returns = np.diff(price_data) / price_data[:-1]
        
        if len(returns) >= self.volatility_window:
            volatility = np.std(returns[-self.volatility_window:])
            
            recent_return = returns[-1]
            
            z_score = recent_return / volatility if volatility > 0 else 0.0
            
            avg_volume = np.mean(volume_data[-self.volatility_window:-1])
            volume_ratio = volume_data[-1] / avg_volume if avg_volume > 0 else 1.0
            
            is_parabolic = (z_score > self.std_dev_threshold and volume_ratio > self.volume_ratio_threshold)
            
            if is_parabolic:
                if not self.parabolic_events[symbol]["is_parabolic"]:
                    self.parabolic_events[symbol] = {
                        "is_parabolic": True,
                        "duration": 1,
                        "magnitude": z_score,
                        "volume_surge": volume_ratio
                    }
                else:
                    self.parabolic_events[symbol]["duration"] += 1
                    self.parabolic_events[symbol]["magnitude"] = max(self.parabolic_events[symbol]["magnitude"], z_score)
                    self.parabolic_events[symbol]["volume_surge"] = max(self.parabolic_events[symbol]["volume_surge"], volume_ratio)
            else:
                self.parabolic_events[symbol] = {
                    "is_parabolic": False,
                    "duration": 0,
                    "magnitude": 0.0,
                    "volume_surge": 0.0
                }
        
    def _manage_active_rides(self, symbol, current_price, current_time):
        """
        Manage active momentum rides.
        
        Parameters:
        - symbol: The trading symbol
        - current_price: Current price
        - current_time: Current datetime
        """
        if symbol in self.active_rides:
            ride = self.active_rides[symbol]
            
            ride["current_price"] = current_price
            ride["current_time"] = current_time
            ride["duration"] = (current_time - ride["start_time"]).total_seconds() / 60.0  # Duration in minutes
            ride["return"] = (current_price / ride["start_price"]) - 1.0
            
            should_exit = False
            
            if self.momentum_scores[symbol]["short"] < 0:
                should_exit = True
                ride["exit_reason"] = "Momentum reversal"
            
            if not self.parabolic_events[symbol]["is_parabolic"] and ride["return"] > 0.1:
                should_exit = True
                ride["exit_reason"] = "Parabolic event ended with profit"
            
            if ride["duration"] > 1440:  # 24 hours in minutes
                should_exit = True
                ride["exit_reason"] = "Time-based exit"
            
            if should_exit:
                ride["exit_price"] = current_price
                ride["exit_time"] = current_time
                
                self.completed_rides.append(ride.copy())
                
                del self.active_rides[symbol]
        
        elif self.parabolic_events[symbol]["is_parabolic"] and self.momentum_scores[symbol]["combined"] > 0:
            self.active_rides[symbol] = {
                "symbol": symbol,
                "start_price": current_price,
                "start_time": current_time,
                "current_price": current_price,
                "current_time": current_time,
                "duration": 0.0,
                "return": 0.0,
                "parabolic_magnitude": self.parabolic_events[symbol]["magnitude"],
                "volume_surge": self.parabolic_events[symbol]["volume_surge"],
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None
            }
        
    def _generate_signal(self, symbol):
        """
        Generate trading signal based on momentum analysis.
        
        Parameters:
        - symbol: The trading symbol
        
        Returns:
        - Dictionary containing signal information
        """
        if symbol not in self.momentum_scores or self.momentum_scores[symbol]["combined"] == 0.0:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reason": "Insufficient data for momentum analysis"
            }
        
        combined_score = self.momentum_scores[symbol]["combined"]
        short_score = self.momentum_scores[symbol]["short"]
        
        has_active_ride = symbol in self.active_rides
        
        is_parabolic = self.parabolic_events[symbol]["is_parabolic"]
        
        if is_parabolic and short_score > 0.5:
            return {
                "direction": "STRONG_BUY",
                "confidence": min(1.0, self.parabolic_events[symbol]["magnitude"] / 5.0),
                "reason": "Parabolic price event with strong momentum"
            }
        elif has_active_ride:
            return {
                "direction": "HOLD",
                "confidence": min(1.0, self.active_rides[symbol]["return"] * 5.0),
                "reason": "Active momentum ride in progress"
            }
        elif combined_score > 0.5:
            return {
                "direction": "BUY",
                "confidence": min(1.0, combined_score),
                "reason": "Strong positive momentum across timeframes"
            }
        elif combined_score > 0.2:
            return {
                "direction": "WEAK_BUY",
                "confidence": min(1.0, combined_score * 2.0),
                "reason": "Moderate positive momentum"
            }
        elif combined_score < -0.5:
            return {
                "direction": "STRONG_SELL",
                "confidence": min(1.0, abs(combined_score)),
                "reason": "Strong negative momentum across timeframes"
            }
        elif combined_score < -0.2:
            return {
                "direction": "WEAK_SELL",
                "confidence": min(1.0, abs(combined_score) * 2.0),
                "reason": "Moderate negative momentum"
            }
        else:
            return {
                "direction": "NEUTRAL",
                "confidence": 1.0 - min(1.0, abs(combined_score) * 5.0),
                "reason": "Weak or mixed momentum signals"
            }
    
    def get_completed_rides(self):
        """
        Get list of completed momentum rides.
        
        Returns:
        - List of completed rides
        """
        return self.completed_rides
    
    def get_active_rides(self):
        """
        Get dictionary of active momentum rides.
        
        Returns:
        - Dictionary of active rides
        """
        return self.active_rides
