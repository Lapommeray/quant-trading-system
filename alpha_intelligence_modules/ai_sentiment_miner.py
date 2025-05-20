"""
AI Sentiment Miner

Trades off Reddit/X sentiment spikes for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import re

class AISentimentMiner:
    """
    Trades off Reddit/X sentiment spikes.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the AI Sentiment Miner.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("AISentimentMiner")
        self.logger.setLevel(logging.INFO)
        
        self.sentiment_sources = ["reddit", "twitter", "stocktwits", "youtube", "news"]
        
        self.sentiment_thresholds = {
            "extreme_negative": -0.8,
            "negative": -0.3,
            "neutral": 0.3,
            "positive": 0.7,
            "extreme_positive": 0.9
        }
        
        self.volume_thresholds = {
            "low": 0.5,
            "normal": 1.0,
            "high": 2.0,
            "extreme": 5.0
        }
        
        self.sentiment_data = {}
        
        self.sentiment_signals = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(minutes=15)
        
        self.tracked_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BTC", "ETH"]
        
        self.sentiment_history = {}
        
    def update(self, current_time, custom_data=None):
        """
        Update the AI sentiment miner with latest data.
        
        Parameters:
        - current_time: Current datetime
        - custom_data: Custom sentiment data (optional)
        
        Returns:
        - Dictionary containing sentiment results
        """
        if current_time - self.last_update < self.update_frequency and custom_data is None:
            return {
                "sentiment_data": self.sentiment_data,
                "sentiment_signals": self.sentiment_signals
            }
            
        if custom_data is not None:
            self._update_sentiment_data(custom_data)
        else:
            self._update_sentiment_data_internal()
        
        self._analyze_sentiment()
        
        self._generate_signals()
        
        self.last_update = current_time
        
        return {
            "sentiment_data": self.sentiment_data,
            "sentiment_signals": self.sentiment_signals
        }
        
    def _update_sentiment_data(self, custom_data):
        """
        Update sentiment data.
        
        Parameters:
        - custom_data: Custom sentiment data
        """
        for symbol, data in custom_data.items():
            if symbol not in self.sentiment_data:
                self.sentiment_data[symbol] = {}
            
            for source, sentiment in data.items():
                if source in self.sentiment_sources:
                    self.sentiment_data[symbol][source] = sentiment
        
    def _update_sentiment_data_internal(self):
        """
        Update sentiment data internally.
        """
        
        for symbol in self.tracked_symbols:
            if symbol not in self.sentiment_data:
                self.sentiment_data[symbol] = {}
                
            for source in self.sentiment_sources:
                base_sentiment = np.random.normal(0.1, 0.3)
                
                if symbol in ["AAPL", "MSFT", "AMZN"]:
                    base_sentiment += 0.2
                elif symbol in ["BTC", "ETH"]:
                    base_sentiment += np.random.normal(0, 0.5)
                
                sentiment_score = max(-1.0, min(1.0, base_sentiment))
                
                self.sentiment_data[symbol][source] = {
                    "score": sentiment_score,
                    "volume": np.random.uniform(0.5, 3.0),  # Relative to normal volume
                    "change_1h": np.random.normal(0, 0.1),
                    "change_24h": np.random.normal(0, 0.2)
                }
            
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            avg_sentiment = np.mean([data["score"] for source, data in self.sentiment_data[symbol].items()])
            avg_volume = np.mean([data["volume"] for source, data in self.sentiment_data[symbol].items()])
            
            self.sentiment_history[symbol].append({
                "timestamp": datetime.now(),
                "sentiment": avg_sentiment,
                "volume": avg_volume
            })
            
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]
        
    def _analyze_sentiment(self):
        """
        Analyze sentiment data.
        """
        for symbol, sources in self.sentiment_data.items():
            weighted_sentiment = 0.0
            total_weight = 0.0
            
            for source, data in sources.items():
                if source == "reddit":
                    weight = 0.25
                elif source == "twitter":
                    weight = 0.3
                elif source == "stocktwits":
                    weight = 0.2
                elif source == "youtube":
                    weight = 0.1
                elif source == "news":
                    weight = 0.15
                else:
                    weight = 0.0
                
                volume_multiplier = min(2.0, data["volume"])
                adjusted_weight = weight * volume_multiplier
                
                weighted_sentiment += data["score"] * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                final_sentiment = weighted_sentiment / total_weight
            else:
                final_sentiment = 0.0
            
            sentiment_volatility = 0.0
            if symbol in self.sentiment_history and len(self.sentiment_history[symbol]) > 5:
                recent_sentiments = [point["sentiment"] for point in self.sentiment_history[symbol][-5:]]
                sentiment_volatility = np.std(recent_sentiments)
            
            sentiment_momentum = 0.0
            if symbol in self.sentiment_history and len(self.sentiment_history[symbol]) > 5:
                recent_sentiments = [point["sentiment"] for point in self.sentiment_history[symbol][-5:]]
                sentiment_momentum = recent_sentiments[-1] - recent_sentiments[0]
            
            if symbol not in self.sentiment_signals:
                self.sentiment_signals[symbol] = {}
            
            self.sentiment_signals[symbol]["analysis"] = {
                "weighted_sentiment": final_sentiment,
                "sentiment_volatility": sentiment_volatility,
                "sentiment_momentum": sentiment_momentum
            }
        
    def _generate_signals(self):
        """
        Generate sentiment signals.
        """
        for symbol, data in self.sentiment_signals.items():
            if "analysis" not in data:
                continue
                
            analysis = data["analysis"]
            
            sentiment = analysis["weighted_sentiment"]
            volatility = analysis["sentiment_volatility"]
            momentum = analysis["sentiment_momentum"]
            
            if sentiment <= self.sentiment_thresholds["extreme_negative"]:
                sentiment_level = "extreme_negative"
            elif sentiment <= self.sentiment_thresholds["negative"]:
                sentiment_level = "negative"
            elif sentiment <= self.sentiment_thresholds["neutral"]:
                sentiment_level = "neutral"
            elif sentiment <= self.sentiment_thresholds["positive"]:
                sentiment_level = "positive"
            else:
                sentiment_level = "extreme_positive"
            
            signal_type = "NEUTRAL"
            signal_strength = 0.0
            
            if sentiment_level == "extreme_positive" and momentum > 0.3:
                signal_type = "STRONG_BUY"
                signal_strength = 0.9
            elif sentiment_level == "positive" and momentum > 0.2:
                signal_type = "BUY"
                signal_strength = 0.7
            elif sentiment_level == "extreme_negative" and momentum < -0.3:
                signal_type = "STRONG_SELL"
                signal_strength = 0.9
            elif sentiment_level == "negative" and momentum < -0.2:
                signal_type = "SELL"
                signal_strength = 0.7
            elif sentiment_level == "extreme_positive":
                signal_type = "BUY"
                signal_strength = 0.6
            elif sentiment_level == "extreme_negative":
                signal_type = "SELL"
                signal_strength = 0.6
            elif sentiment_level == "positive" and momentum > 0:
                signal_type = "WEAK_BUY"
                signal_strength = 0.4
            elif sentiment_level == "negative" and momentum < 0:
                signal_type = "WEAK_SELL"
                signal_strength = 0.4
            
            if volatility > 0.3:
                signal_strength *= 0.8
            
            self.sentiment_signals[symbol]["signal"] = {
                "type": signal_type,
                "strength": signal_strength,
                "sentiment_level": sentiment_level,
                "momentum": momentum,
                "volatility": volatility
            }
        
    def get_sentiment_data(self, symbol=None):
        """
        Get sentiment data.
        
        Parameters:
        - symbol: Symbol to get data for (optional)
        
        Returns:
        - Sentiment data
        """
        if symbol is not None:
            return self.sentiment_data.get(symbol, {})
        else:
            return self.sentiment_data
        
    def get_sentiment_signals(self, symbol=None):
        """
        Get sentiment signals.
        
        Parameters:
        - symbol: Symbol to get signals for (optional)
        
        Returns:
        - Sentiment signals
        """
        if symbol is not None:
            return self.sentiment_signals.get(symbol, {})
        else:
            return self.sentiment_signals
        
    def get_sentiment_history(self, symbol=None):
        """
        Get sentiment history.
        
        Parameters:
        - symbol: Symbol to get history for (optional)
        
        Returns:
        - Sentiment history
        """
        if symbol is not None:
            return self.sentiment_history.get(symbol, [])
        else:
            return self.sentiment_history
        
    def get_trading_signal(self, symbol):
        """
        Get trading signal for a symbol.
        
        Parameters:
        - symbol: Symbol to get signal for
        
        Returns:
        - Trading signal
        """
        if symbol not in self.sentiment_signals or "signal" not in self.sentiment_signals[symbol]:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0
            }
        
        signal = self.sentiment_signals[symbol]["signal"]
        
        if signal["type"] == "STRONG_BUY":
            action = "BUY"
            confidence = signal["strength"]
        elif signal["type"] == "BUY":
            action = "BUY"
            confidence = signal["strength"]
        elif signal["type"] == "WEAK_BUY":
            action = "BUY"
            confidence = signal["strength"]
        elif signal["type"] == "STRONG_SELL":
            action = "SELL"
            confidence = signal["strength"]
        elif signal["type"] == "SELL":
            action = "SELL"
            confidence = signal["strength"]
        elif signal["type"] == "WEAK_SELL":
            action = "SELL"
            confidence = signal["strength"]
        else:
            action = "NEUTRAL"
            confidence = 0.0
        
        return {
            "action": action,
            "confidence": confidence
        }
