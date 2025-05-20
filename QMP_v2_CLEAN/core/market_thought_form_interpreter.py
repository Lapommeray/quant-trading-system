# Reads collective trader intent from volume + structure over time.

import numpy as np
from scipy import stats
import math

class MarketThoughtFormInterpreter:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 60  # Number of candles to analyze
        self.thought_threshold = 0.65
        self.volume_weight = 0.6
        self.price_weight = 0.4
        
    def decode(self, symbol, history_window):
        """
        Analyzes market structure and volume to detect collective trader intent.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - Dictionary with collective intent detection results
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"ThoughtForm: Insufficient history for {symbol}")
            return {"collective_intent": "NEUTRAL", "confidence": 0.0, "thought_patterns": []}
            
        closes = np.array([bar.Close for bar in history_window])
        opens = np.array([bar.Open for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        volumes = np.array([bar.Volume for bar in history_window])
        
        volume_intent = self._analyze_volume_patterns(volumes, closes)
        
        price_intent = self._analyze_price_structure(opens, highs, lows, closes)
        
        time_intent = self._analyze_time_patterns(closes, volumes)
        
        intent_score = (
            self.volume_weight * volume_intent["score"] + 
            self.price_weight * price_intent["score"]
        )
        
        if intent_score > self.thought_threshold:
            direction_votes = {
                "BUY": (volume_intent["direction"] == "BUY") * self.volume_weight + 
                       (price_intent["direction"] == "BUY") * self.price_weight,
                       
                "SELL": (volume_intent["direction"] == "SELL") * self.volume_weight + 
                        (price_intent["direction"] == "SELL") * self.price_weight,
                        
                "NEUTRAL": (volume_intent["direction"] == "NEUTRAL") * self.volume_weight + 
                          (price_intent["direction"] == "NEUTRAL") * self.price_weight
            }
            
            collective_intent = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            
            thought_patterns = []
            if volume_intent["pattern"]:
                thought_patterns.append(volume_intent["pattern"])
            if price_intent["pattern"]:
                thought_patterns.append(price_intent["pattern"])
            if time_intent["pattern"]:
                thought_patterns.append(time_intent["pattern"])
                
            self.algo.Debug(f"ThoughtForm: {symbol} - Collective intent detected! Direction: {collective_intent}, Score: {intent_score:.2f}")
            self.algo.Debug(f"ThoughtForm: {symbol} - Patterns: {', '.join(thought_patterns)}")
        else:
            collective_intent = "NEUTRAL"
            thought_patterns = []
            self.algo.Debug(f"ThoughtForm: {symbol} - No clear collective intent. Score: {intent_score:.2f}")
        
        return {
            "collective_intent": collective_intent,
            "confidence": intent_score,
            "thought_patterns": thought_patterns,
            "volume_intent": volume_intent["direction"],
            "price_intent": price_intent["direction"],
            "time_intent": time_intent["direction"]
        }
    
    def _analyze_volume_patterns(self, volumes, prices):
        """Analyze volume patterns for collective intent"""
        if len(volumes) < 30:
            return {"direction": "NEUTRAL", "score": 0.0, "pattern": None}
        
        recent_vols = volumes[-10:]
        earlier_vols = volumes[-30:-10]
        
        recent_vol_avg = np.mean(recent_vols)
        earlier_vol_avg = np.mean(earlier_vols)
        
        vol_change = recent_vol_avg / earlier_vol_avg if earlier_vol_avg > 0 else 1.0
        
        recent_prices = prices[-10:]
        price_changes = np.diff(recent_prices)
        vol_changes = np.diff(recent_vols)
        
        if len(price_changes) > 1 and len(vol_changes) > 1:
            pv_correlation = np.corrcoef(price_changes, vol_changes)[0, 1]
        else:
            pv_correlation = 0
            
        vol_std = np.std(volumes[-30:])
        vol_mean = np.mean(volumes[-30:])
        recent_vol_z = (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0
        
        direction = "NEUTRAL"
        pattern = None
        score = 0.5  # Base score
        
        if recent_vol_z > 2.0:
            if prices[-1] > prices[-2]:
                direction = "BUY"
                pattern = "BUYING_CLIMAX"
                score = 0.7 + min(recent_vol_z / 10, 0.3)
            else:
                direction = "SELL"
                pattern = "SELLING_CLIMAX"
                score = 0.7 + min(recent_vol_z / 10, 0.3)
                
        elif vol_change > 1.3:  # Significant volume increase
            if pv_correlation > 0.5:
                direction = "BUY"
                pattern = "ACCUMULATION"
                score = 0.6 + min(pv_correlation / 2, 0.3)
            elif pv_correlation < -0.5:
                direction = "SELL"
                pattern = "DISTRIBUTION"
                score = 0.6 + min(abs(pv_correlation) / 2, 0.3)
                
        elif vol_change < 0.7:  # Significant volume decrease
            recent_price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            if recent_price_trend > 0:
                direction = "BUY"
                pattern = "STEALTH_BUYING"
                score = 0.6
            elif recent_price_trend < 0:
                direction = "SELL"
                pattern = "STEALTH_SELLING"
                score = 0.6
                
        return {"direction": direction, "score": score, "pattern": pattern}
    
    def _analyze_price_structure(self, opens, highs, lows, closes):
        """Analyze price structure for collective intent"""
        if len(closes) < 20:
            return {"direction": "NEUTRAL", "score": 0.0, "pattern": None}
        
        bodies = np.abs(closes - opens)
        ranges = highs - lows
        body_ratios = bodies / ranges
        
        close_positions = (closes - lows) / ranges
        
        recent_bodies = body_ratios[-5:]
        recent_positions = close_positions[-5:]
        
        avg_body_ratio = np.mean(recent_bodies)
        avg_position = np.mean(recent_positions)
        
        direction = "NEUTRAL"
        pattern = None
        score = 0.5  # Base score
        
        if avg_body_ratio > 0.6 and avg_position > 0.7:
            direction = "BUY"
            pattern = "STRONG_BULLISH_STRUCTURE"
            score = 0.7 + min(avg_body_ratio / 5, 0.3)
            
        elif avg_body_ratio > 0.6 and avg_position < 0.3:
            direction = "SELL"
            pattern = "STRONG_BEARISH_STRUCTURE"
            score = 0.7 + min(avg_body_ratio / 5, 0.3)
            
        elif avg_body_ratio < 0.4 and avg_position > 0.6:
            direction = "SELL"
            pattern = "UPPER_WICK_REJECTION"
            score = 0.6
            
        elif avg_body_ratio < 0.4 and avg_position < 0.4:
            direction = "BUY"
            pattern = "LOWER_WICK_SUPPORT"
            score = 0.6
            
        for i in range(len(closes) - 1, max(len(closes) - 5, 0), -1):
            if (closes[i] > opens[i] and  # Current candle is bullish
                closes[i-1] < opens[i-1] and  # Previous candle is bearish
                closes[i] > opens[i-1] and  # Current close above previous open
                opens[i] < closes[i-1]):  # Current open below previous close
                
                direction = "BUY"
                pattern = "BULLISH_ENGULFING"
                score = 0.8
                break
                
            elif (closes[i] < opens[i] and  # Current candle is bearish
                  closes[i-1] > opens[i-1] and  # Previous candle is bullish
                  closes[i] < opens[i-1] and  # Current close below previous open
                  opens[i] > closes[i-1]):  # Current open above previous close
                
                direction = "SELL"
                pattern = "BEARISH_ENGULFING"
                score = 0.8
                break
                
        return {"direction": direction, "score": score, "pattern": pattern}
    
    def _analyze_time_patterns(self, closes, volumes):
        """Analyze time-based patterns for collective intent"""
        if len(closes) < 40:
            return {"direction": "NEUTRAL", "score": 0.0, "pattern": None}
        
        ma_short = np.mean(closes[-10:])
        ma_medium = np.mean(closes[-20:])
        ma_long = np.mean(closes[-40:])
        
        vwap = np.sum(closes[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
        
        direction = "NEUTRAL"
        pattern = None
        score = 0.5  # Base score
        
        if ma_short > ma_medium > ma_long:
            direction = "BUY"
            pattern = "BULLISH_MA_ALIGNMENT"
            score = 0.6
            
        elif ma_short < ma_medium < ma_long:
            direction = "SELL"
            pattern = "BEARISH_MA_ALIGNMENT"
            score = 0.6
            
        if closes[-1] > vwap and closes[-2] < vwap:
            direction = "BUY"
            pattern = "VWAP_BREAKOUT"
            score = 0.7
            
        elif closes[-1] < vwap and closes[-2] > vwap:
            direction = "SELL"
            pattern = "VWAP_BREAKDOWN"
            score = 0.7
            
        price_momentum = closes[-1] - closes[-10]
        vol_momentum = volumes[-1] - np.mean(volumes[-10:])
        
        if price_momentum > 0 and vol_momentum < 0:
            direction = "SELL"
            pattern = "BEARISH_DIVERGENCE"
            score = 0.65
            
        elif price_momentum < 0 and vol_momentum > 0:
            direction = "BUY"
            pattern = "BULLISH_DIVERGENCE"
            score = 0.65
            
        return {"direction": direction, "score": score, "pattern": pattern}
