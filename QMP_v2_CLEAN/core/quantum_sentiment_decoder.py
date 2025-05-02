"""
Quantum Sentiment Decoder Module

This module decodes quantum-level sentiment patterns in market data, identifying
emotional resonance and intention fields that precede price movements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

class QuantumSentimentDecoder:
    """
    Decodes quantum-level sentiment patterns in market data.
    """
    
    def __init__(self, algorithm):
        """Initialize the Quantum Sentiment Decoder module."""
        self.algorithm = algorithm
        self.logger = logging.getLogger("QuantumSentimentDecoder")
        self.logger.setLevel(logging.INFO)
        
        self.fear_greed_field = {}
        self.emotional_resonance = {}
        self.intention_field = {}
        self.quantum_probability = {}
        
        self.sentiment_decay = 0.95
        self.resonance_threshold = 0.7
        self.intention_clarity_threshold = 0.65
        
        self.emotional_patterns = {
            "fear": {"price": "sharp_decline", "volume": "spike"},
            "greed": {"price": "accelerating_uptrend", "volume": "increasing"},
            "capitulation": {"price": "waterfall_decline", "volume": "climax"},
            "euphoria": {"price": "parabolic_rise", "volume": "climax"},
            "anxiety": {"price": "choppy_range", "volume": "declining"},
            "relief": {"price": "bounce_after_decline", "volume": "average"},
            "disbelief": {"price": "early_uptrend", "volume": "below_average"},
            "complacency": {"price": "slow_uptrend", "volume": "declining"}
        }
        
        algorithm.Debug("Quantum Sentiment Decoder module initialized")
    
    def decode(self, symbol, history_data, news_data=None):
        """
        Decode quantum sentiment patterns in market data.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        - news_data: Optional dictionary containing news sentiment data
        
        Returns:
        - Dictionary containing decoded sentiment information
        """
        symbol_str = str(symbol)
        
        if "1m" not in history_data or history_data["1m"].empty:
            return self._create_neutral_result(symbol_str)
        
        price_sentiment = self._analyze_price_sentiment(symbol_str, history_data)
        
        news_sentiment = self._analyze_news_sentiment(symbol_str, news_data)
        
        emotional_state = self._detect_emotional_state(symbol_str, history_data)
        
        intention = self._detect_intention_field(symbol_str, history_data)
        
        quantum_field = self._calculate_quantum_field(
            symbol_str, price_sentiment, news_sentiment, 
            emotional_state, intention
        )
        
        self._update_sentiment_fields(
            symbol_str, price_sentiment, news_sentiment, 
            emotional_state, intention, quantum_field
        )
        
        result = {
            "symbol": symbol_str,
            "timestamp": self.algorithm.Time,
            "price_sentiment": price_sentiment,
            "news_sentiment": news_sentiment,
            "emotional_state": emotional_state,
            "intention_field": intention,
            "quantum_field": quantum_field,
            "overall_sentiment": self._calculate_overall_sentiment(
                price_sentiment, news_sentiment, emotional_state, 
                intention, quantum_field
            )
        }
        
        return result
    
    def _create_neutral_result(self, symbol_str):
        """Create a neutral sentiment result when data is insufficient."""
        return {
            "symbol": symbol_str,
            "timestamp": self.algorithm.Time,
            "price_sentiment": {"score": 0.5, "direction": "NEUTRAL", "strength": 0.0},
            "news_sentiment": {"score": 0.5, "direction": "NEUTRAL", "strength": 0.0},
            "emotional_state": {
                "primary": "neutral",
                "secondary": None,
                "intensity": 0.0,
                "transition": None
            },
            "intention_field": {
                "primary": "neutral",
                "clarity": 0.0,
                "strength": 0.0,
                "direction": "NEUTRAL"
            },
            "quantum_field": {
                "probability_bias": 0.5,
                "timeline_convergence": 0.0,
                "emotional_resonance": 0.0,
                "field_strength": 0.0
            },
            "overall_sentiment": {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "strength": 0.0,
                "clarity": 0.0
            }
        }
    
    def _analyze_price_sentiment(self, symbol_str, history_data):
        """Analyze price action to determine sentiment."""
        sentiment = {
            "score": 0.5,
            "direction": "NEUTRAL",
            "strength": 0.0,
            "timeframes": {}
        }
        
        for timeframe, df in history_data.items():
            if df.empty or len(df) < 20:
                continue
                
            df_copy = df.copy()
            
            delta = df_copy["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df_copy["RSI"] = 100 - (100 / (1 + rs))
            
            df_copy["EMA12"] = df_copy["Close"].ewm(span=12, adjust=False).mean()
            df_copy["EMA26"] = df_copy["Close"].ewm(span=26, adjust=False).mean()
            df_copy["MACD"] = df_copy["EMA12"] - df_copy["EMA26"]
            df_copy["Signal"] = df_copy["MACD"].ewm(span=9, adjust=False).mean()
            
            recent_rsi = df_copy["RSI"].iloc[-1]
            recent_macd = df_copy["MACD"].iloc[-1]
            recent_signal = df_copy["Signal"].iloc[-1]
            
            timeframe_sentiment = 0.5  # Neutral starting point
            
            if recent_rsi > 70:
                rsi_sentiment = 0.7  # Overbought
            elif recent_rsi < 30:
                rsi_sentiment = 0.3  # Oversold
            else:
                rsi_sentiment = 0.5 + (recent_rsi - 50) / 100
            
            if recent_macd > 0 and recent_macd > recent_signal:
                macd_sentiment = 0.7  # Strong bullish
            elif recent_macd > 0 and recent_macd < recent_signal:
                macd_sentiment = 0.6  # Weakening bullish
            elif recent_macd < 0 and recent_macd < recent_signal:
                macd_sentiment = 0.3  # Strong bearish
            elif recent_macd < 0 and recent_macd > recent_signal:
                macd_sentiment = 0.4  # Weakening bearish
            else:
                macd_sentiment = 0.5
            
            price_change = df_copy["Close"].pct_change(5).iloc[-1]
            if price_change > 0:
                price_sentiment = 0.5 + min(0.3, price_change * 10)
            else:
                price_sentiment = 0.5 - min(0.3, abs(price_change) * 10)
            
            timeframe_sentiment = price_sentiment * 0.4 + rsi_sentiment * 0.3 + macd_sentiment * 0.3
            
            if timeframe_sentiment > 0.55:
                direction = "BUY"
                strength = (timeframe_sentiment - 0.5) * 2  # Scale to 0-1
            elif timeframe_sentiment < 0.45:
                direction = "SELL"
                strength = (0.5 - timeframe_sentiment) * 2  # Scale to 0-1
            else:
                direction = "NEUTRAL"
                strength = 0.0
            
            sentiment["timeframes"][timeframe] = {
                "score": timeframe_sentiment,
                "direction": direction,
                "strength": strength
            }
        
        if sentiment["timeframes"]:
            weights = {
                "1m": 0.05, "5m": 0.10, "10m": 0.15,
                "15m": 0.20, "20m": 0.25, "25m": 0.25
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for timeframe, data in sentiment["timeframes"].items():
                weight = weights.get(timeframe, 0.1)
                weighted_sum += data["score"] * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_score = weighted_sum / total_weight
                
                if overall_score > 0.55:
                    sentiment["direction"] = "BUY"
                    sentiment["strength"] = (overall_score - 0.5) * 2
                elif overall_score < 0.45:
                    sentiment["direction"] = "SELL"
                    sentiment["strength"] = (0.5 - overall_score) * 2
                
                sentiment["score"] = overall_score
        
        return sentiment
    
    def _analyze_news_sentiment(self, symbol_str, news_data):
        """Analyze news data to determine sentiment."""
        sentiment = {
            "score": 0.5,
            "direction": "NEUTRAL",
            "strength": 0.0,
            "sources": {}
        }
        
        if not news_data:
            return sentiment
        
        total_score = 0
        total_weight = 0
        
        for source, data in news_data.items():
            if "sentiment_score" in data:
                source_score = data["sentiment_score"]
                source_weight = data.get("weight", 1.0)
            else:
                continue
            
            sentiment["sources"][source] = {
                "score": source_score,
                "weight": source_weight
            }
            
            total_score += source_score * source_weight
            total_weight += source_weight
        
        if total_weight > 0:
            overall_score = total_score / total_weight
            
            if overall_score > 0.55:
                sentiment["direction"] = "BUY"
                sentiment["strength"] = (overall_score - 0.5) * 2
            elif overall_score < 0.45:
                sentiment["direction"] = "SELL"
                sentiment["strength"] = (0.5 - overall_score) * 2
            
            sentiment["score"] = overall_score
        
        return sentiment
    
    def _detect_emotional_state(self, symbol_str, history_data):
        """Detect emotional state from price patterns."""
        emotional_state = {
            "primary": "neutral",
            "secondary": None,
            "intensity": 0.0,
            "transition": None
        }
        
        if "1m" not in history_data or history_data["1m"].empty:
            return emotional_state
        
        df_1m = history_data["1m"].copy()
        
        if len(df_1m) < 30:
            return emotional_state
        
        df_1m["returns"] = df_1m["Close"].pct_change()
        df_1m["volatility"] = df_1m["returns"].rolling(window=20).std()
        
        if "Volume" in df_1m.columns:
            df_1m["volume_ma"] = df_1m["Volume"].rolling(window=20).mean()
            df_1m["volume_ratio"] = df_1m["Volume"] / df_1m["volume_ma"]
        else:
            df_1m["volume_ratio"] = 1.0
        
        price_change_5m = df_1m["Close"].pct_change(5).iloc[-1] * 100
        price_change_15m = df_1m["Close"].pct_change(15).iloc[-1] * 100
        recent_volatility = df_1m["volatility"].iloc[-1]
        recent_volume_ratio = df_1m["volume_ratio"].iloc[-1]
        
        emotion_scores = {}
        
        if price_change_5m < -1.0 and recent_volatility > df_1m["volatility"].mean() * 1.5:
            fear_score = min(1.0, abs(price_change_5m) / 3.0)
            emotion_scores["fear"] = fear_score
        
        if price_change_5m > 1.0 and price_change_15m > 2.0:
            greed_score = min(1.0, price_change_5m / 3.0)
            emotion_scores["greed"] = greed_score
        
        if price_change_15m < -3.0 and recent_volume_ratio > 2.0:
            capitulation_score = min(1.0, abs(price_change_15m) / 5.0)
            emotion_scores["capitulation"] = capitulation_score
        
        if price_change_15m > 3.0 and recent_volume_ratio > 2.0:
            euphoria_score = min(1.0, price_change_15m / 5.0)
            emotion_scores["euphoria"] = euphoria_score
        
        if emotion_scores:
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            emotional_state["primary"] = sorted_emotions[0][0]
            emotional_state["intensity"] = sorted_emotions[0][1]
            
            if len(sorted_emotions) > 1:
                emotional_state["secondary"] = sorted_emotions[1][0]
            
            if symbol_str in self.emotional_resonance:
                previous_emotion = self.emotional_resonance[symbol_str]["primary"]
                
                if previous_emotion != emotional_state["primary"]:
                    emotional_state["transition"] = f"{previous_emotion}_to_{emotional_state['primary']}"
        
        return emotional_state
    
    def _detect_intention_field(self, symbol_str, history_data):
        """Detect intention field from price patterns."""
        intention = {
            "primary": "neutral",
            "clarity": 0.0,
            "strength": 0.0,
            "direction": "NEUTRAL"
        }
        
        if "15m" not in history_data or history_data["15m"].empty:
            return intention
        
        df_15m = history_data["15m"].copy()
        
        if len(df_15m) < 20:
            return intention
        
        df_15m["higher_high"] = (df_15m["High"] > df_15m["High"].shift(1)) & (df_15m["High"].shift(1) > df_15m["High"].shift(2))
        df_15m["lower_low"] = (df_15m["Low"] < df_15m["Low"].shift(1)) & (df_15m["Low"].shift(1) < df_15m["Low"].shift(2))
        df_15m["higher_low"] = (df_15m["Low"] > df_15m["Low"].shift(1))
        df_15m["lower_high"] = (df_15m["High"] < df_15m["High"].shift(1))
        
        intention_scores = {}
        
        recent_higher_lows = df_15m["higher_low"].iloc[-10:].sum()
        if recent_higher_lows >= 7:
            accumulation_score = min(1.0, recent_higher_lows / 10.0)
            intention_scores["accumulation"] = accumulation_score
            
        recent_lower_highs = df_15m["lower_high"].iloc[-10:].sum()
        if recent_lower_highs >= 7:
            distribution_score = min(1.0, recent_lower_highs / 10.0)
            intention_scores["distribution"] = distribution_score
        
        recent_higher_highs = df_15m["higher_high"].iloc[-10:].sum()
        if recent_higher_highs >= 5 and recent_higher_lows >= 5:
            markup_score = min(1.0, (recent_higher_highs + recent_higher_lows) / 20.0)
            intention_scores["markup"] = markup_score
        
        recent_lower_lows = df_15m["lower_low"].iloc[-10:].sum()
        if recent_lower_lows >= 5 and recent_lower_highs >= 5:
            markdown_score = min(1.0, (recent_lower_lows + recent_lower_highs) / 20.0)
            intention_scores["markdown"] = markdown_score
        
        if intention_scores:
            sorted_intentions = sorted(intention_scores.items(), key=lambda x: x[1], reverse=True)
            
            intention["primary"] = sorted_intentions[0][0]
            intention["clarity"] = sorted_intentions[0][1]
            
            if intention["primary"] in ["accumulation", "markup"]:
                intention["direction"] = "BUY"
                intention["strength"] = intention["clarity"]
            elif intention["primary"] in ["distribution", "markdown"]:
                intention["direction"] = "SELL"
                intention["strength"] = intention["clarity"]
        
        return intention
    
    def _calculate_quantum_field(self, symbol_str, price_sentiment, news_sentiment, emotional_state, intention):
        """Calculate quantum probability field."""
        quantum_field = {
            "probability_bias": 0.5,
            "timeline_convergence": 0.0,
            "emotional_resonance": 0.0,
            "field_strength": 0.0
        }
        
        if price_sentiment["direction"] == "BUY":
            prob_bias = 0.5 + price_sentiment["strength"] * 0.3
        elif price_sentiment["direction"] == "SELL":
            prob_bias = 0.5 - price_sentiment["strength"] * 0.3
        else:
            prob_bias = 0.5
            
        if news_sentiment["direction"] == price_sentiment["direction"] and news_sentiment["direction"] != "NEUTRAL":
            prob_bias += (news_sentiment["strength"] * 0.1) * (1 if price_sentiment["direction"] == "BUY" else -1)
        
        if intention["direction"] == price_sentiment["direction"] and intention["direction"] != "NEUTRAL":
            timeline_convergence = intention["clarity"] * price_sentiment["strength"]
        else:
            timeline_convergence = 0.0
        
        if emotional_state["primary"] in ["greed", "euphoria"] and price_sentiment["direction"] == "BUY":
            emotional_resonance = emotional_state["intensity"] * 0.8  # Reduced due to potential reversal
        elif emotional_state["primary"] in ["fear", "capitulation"] and price_sentiment["direction"] == "SELL":
            emotional_resonance = emotional_state["intensity"] * 0.8  # Reduced due to potential reversal
        elif emotional_state["primary"] in ["disbelief", "complacency"] and price_sentiment["direction"] == "BUY":
            emotional_resonance = emotional_state["intensity"] * 1.2  # Enhanced due to potential continuation
        elif emotional_state["primary"] in ["anxiety", "relief"] and price_sentiment["direction"] == "SELL":
            emotional_resonance = emotional_state["intensity"] * 1.2  # Enhanced due to potential continuation
        else:
            emotional_resonance = emotional_state["intensity"] * 0.5
        
        field_strength = (timeline_convergence + emotional_resonance) / 2
        
        quantum_field["probability_bias"] = max(0.0, min(1.0, prob_bias))
        quantum_field["timeline_convergence"] = timeline_convergence
        quantum_field["emotional_resonance"] = emotional_resonance
        quantum_field["field_strength"] = field_strength
        
        return quantum_field
    
    def _update_sentiment_fields(self, symbol_str, price_sentiment, news_sentiment, emotional_state, intention, quantum_field):
        """Update sentiment fields for future reference."""
        self.emotional_resonance[symbol_str] = emotional_state
        
        self.intention_field[symbol_str] = intention
        
        self.quantum_probability[symbol_str] = quantum_field
        
        fear_greed_score = 0.5
        
        if emotional_state["primary"] == "fear":
            fear_greed_score = 0.3 - emotional_state["intensity"] * 0.2
        elif emotional_state["primary"] == "capitulation":
            fear_greed_score = 0.1
        elif emotional_state["primary"] == "greed":
            fear_greed_score = 0.7 + emotional_state["intensity"] * 0.2
        elif emotional_state["primary"] == "euphoria":
            fear_greed_score = 0.9
        
        self.fear_greed_field[symbol_str] = fear_greed_score
    
    def _calculate_overall_sentiment(self, price_sentiment, news_sentiment, emotional_state, intention, quantum_field):
        """Calculate overall sentiment from all components."""
        overall_sentiment = {
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "strength": 0.0,
            "clarity": 0.0
        }
        
        directions = []
        
        if price_sentiment["direction"] != "NEUTRAL":
            directions.append(price_sentiment["direction"])
        
        if news_sentiment["direction"] != "NEUTRAL":
            directions.append(news_sentiment["direction"])
        
        if intention["direction"] != "NEUTRAL":
            directions.append(intention["direction"])
        
        if directions:
            buy_count = directions.count("BUY")
            sell_count = directions.count("SELL")
            
            if buy_count > sell_count:
                overall_sentiment["direction"] = "BUY"
            elif sell_count > buy_count:
                overall_sentiment["direction"] = "SELL"
        
        if overall_sentiment["direction"] != "NEUTRAL":
            confidence_factors = []
            
            if price_sentiment["direction"] == overall_sentiment["direction"]:
                confidence_factors.append(price_sentiment["strength"] * 0.4)
            
            if news_sentiment["direction"] == overall_sentiment["direction"]:
                confidence_factors.append(news_sentiment["strength"] * 0.2)
            
            if intention["direction"] == overall_sentiment["direction"]:
                confidence_factors.append(intention["clarity"] * 0.3)
            
            if overall_sentiment["direction"] == "BUY" and quantum_field["probability_bias"] > 0.5:
                confidence_factors.append((quantum_field["probability_bias"] - 0.5) * 2 * 0.1)
            elif overall_sentiment["direction"] == "SELL" and quantum_field["probability_bias"] < 0.5:
                confidence_factors.append((0.5 - quantum_field["probability_bias"]) * 2 * 0.1)
            
            if confidence_factors:
                overall_sentiment["confidence"] = sum(confidence_factors)
        
        if overall_sentiment["direction"] == "BUY":
            strength_factors = []
            
            if price_sentiment["direction"] == "BUY":
                strength_factors.append(price_sentiment["strength"])
            
            if intention["direction"] == "BUY":
                strength_factors.append(intention["strength"])
            
            if quantum_field["probability_bias"] > 0.5:
                strength_factors.append((quantum_field["probability_bias"] - 0.5) * 2)
            
            if strength_factors:
                overall_sentiment["strength"] = sum(strength_factors) / len(strength_factors)
        
        elif overall_sentiment["direction"] == "SELL":
            strength_factors = []
            
            if price_sentiment["direction"] == "SELL":
                strength_factors.append(price_sentiment["strength"])
            
            if intention["direction"] == "SELL":
                strength_factors.append(intention["strength"])
            
            if quantum_field["probability_bias"] < 0.5:
                strength_factors.append((0.5 - quantum_field["probability_bias"]) * 2)
            
            if strength_factors:
                overall_sentiment["strength"] = sum(strength_factors) / len(strength_factors)
        
        clarity_factors = [
            quantum_field["timeline_convergence"],
            intention["clarity"],
            emotional_state["intensity"]
        ]
        
        overall_sentiment["clarity"] = sum(clarity_factors) / len(clarity_factors)
        
        return overall_sentiment
    
    def get_sentiment_state(self, symbol):
        """Get the current sentiment state for a symbol."""
        symbol_str = str(symbol)
        
        return {
            "fear_greed": self.fear_greed_field.get(symbol_str, 0.5),
            "emotional_resonance": self.emotional_resonance.get(symbol_str, {"primary": "neutral", "intensity": 0.0}),
            "intention_field": self.intention_field.get(symbol_str, {"primary": "neutral", "clarity": 0.0}),
            "quantum_probability": self.quantum_probability.get(symbol_str, {"probability_bias": 0.5})
        }
