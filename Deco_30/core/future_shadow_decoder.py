# Detects unmanifested future intentions from market behavior.

import numpy as np
from scipy import stats

class FutureShadowDecoder:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 50  # Number of candles to analyze
        self.shadow_threshold = 0.7
        self.future_window = 10  # Future window to project into
        
    def decode(self, symbol, history_window):
        """
        Analyzes market behavior to detect unmanifested future intentions.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars
        
        Returns:
        - Dictionary with future shadow detection results
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"FutureShadow: Insufficient history for {symbol}")
            return {"future_direction": "NEUTRAL", "confidence": 0.0, "timeframe": 0}
            
        closes = np.array([bar.Close for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        volumes = np.array([bar.Volume for bar in history_window])
        
        accumulation = self._detect_accumulation_distribution(closes, volumes)
        momentum_divergence = self._detect_momentum_divergence(closes)
        order_flow_imbalance = self._detect_order_flow_imbalance(highs, lows, closes, volumes)
        
        shadow_score = (
            0.4 * accumulation["score"] + 
            0.3 * momentum_divergence["score"] + 
            0.3 * order_flow_imbalance["score"]
        )
        
        if shadow_score > self.shadow_threshold:
            direction_votes = {
                "BUY": (accumulation["direction"] == "BUY") * 0.4 + 
                       (momentum_divergence["direction"] == "BUY") * 0.3 + 
                       (order_flow_imbalance["direction"] == "BUY") * 0.3,
                       
                "SELL": (accumulation["direction"] == "SELL") * 0.4 + 
                        (momentum_divergence["direction"] == "SELL") * 0.3 + 
                        (order_flow_imbalance["direction"] == "SELL") * 0.3,
                        
                "NEUTRAL": (accumulation["direction"] == "NEUTRAL") * 0.4 + 
                          (momentum_divergence["direction"] == "NEUTRAL") * 0.3 + 
                          (order_flow_imbalance["direction"] == "NEUTRAL") * 0.3
            }
            
            future_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            
            if accumulation["timeframe"] > 0:
                timeframe = accumulation["timeframe"]
            elif momentum_divergence["timeframe"] > 0:
                timeframe = momentum_divergence["timeframe"]
            else:
                timeframe = order_flow_imbalance["timeframe"]
                
            self.algo.Debug(f"FutureShadow: {symbol} - Future intention detected! Direction: {future_direction}, Score: {shadow_score:.2f}, Timeframe: {timeframe} bars")
        else:
            future_direction = "NEUTRAL"
            timeframe = 0
            self.algo.Debug(f"FutureShadow: {symbol} - No clear future intention. Score: {shadow_score:.2f}")
        
        return {
            "future_direction": future_direction,
            "confidence": shadow_score,
            "timeframe": timeframe,
            "accumulation_score": accumulation["score"],
            "momentum_score": momentum_divergence["score"],
            "order_flow_score": order_flow_imbalance["score"]
        }
    
    def _detect_accumulation_distribution(self, closes, volumes):
        """Detect accumulation or distribution patterns"""
        if len(closes) < 20 or len(volumes) < 20:
            return {"score": 0.0, "direction": "NEUTRAL", "timeframe": 0}
        
        price_changes = np.diff(closes)
        vol_price_changes = price_changes[-20:] * volumes[-21:-1]  # Align arrays
        
        recent_vol_price = np.sum(vol_price_changes[-10:])
        earlier_vol_price = np.sum(vol_price_changes[-20:-10])
        
        recent_price_trend = np.polyfit(range(10), closes[-10:], 1)[0]
        
        if recent_vol_price > earlier_vol_price * 1.2 and recent_price_trend <= 0:
            direction = "BUY"
            score = min(recent_vol_price / (earlier_vol_price + 1e-9), 2.0) / 2.0  # Normalize to [0,1]
            timeframe = self.future_window
        elif recent_vol_price < earlier_vol_price * 0.8 and recent_price_trend >= 0:
            direction = "SELL"
            score = min(earlier_vol_price / (recent_vol_price + 1e-9), 2.0) / 2.0  # Normalize to [0,1]
            timeframe = self.future_window
        else:
            direction = "NEUTRAL"
            score = 0.3  # Base score
            timeframe = 0
            
        return {"score": score, "direction": direction, "timeframe": timeframe}
    
    def _detect_momentum_divergence(self, closes):
        """Detect hidden momentum divergences"""
        if len(closes) < 30:
            return {"score": 0.0, "direction": "NEUTRAL", "timeframe": 0}
        
        momentum_periods = [5, 10, 20]
        momentum_values = []
        
        for period in momentum_periods:
            momentum = closes[-1] - closes[-period-1]
            momentum_values.append(momentum)
        
        short_momentum = momentum_values[0]
        medium_momentum = momentum_values[1]
        long_momentum = momentum_values[2]
        
        if long_momentum > 0 and medium_momentum < 0 and short_momentum < 0:
            direction = "BUY"
            score = 0.7 + min(abs(long_momentum / (short_momentum - 1e-9)), 0.3)
            timeframe = 15  # Longer timeframe for manifestation
        elif long_momentum < 0 and medium_momentum > 0 and short_momentum > 0:
            direction = "SELL"
            score = 0.7 + min(abs(long_momentum / (short_momentum + 1e-9)), 0.3)
            timeframe = 15
        else:
            if all(m > 0 for m in momentum_values):
                direction = "BUY"
                score = 0.6
                timeframe = 5
            elif all(m < 0 for m in momentum_values):
                direction = "SELL"
                score = 0.6
                timeframe = 5
            else:
                direction = "NEUTRAL"
                score = 0.3
                timeframe = 0
                
        return {"score": score, "direction": direction, "timeframe": timeframe}
    
    def _detect_order_flow_imbalance(self, highs, lows, closes, volumes):
        """Detect order flow imbalances that suggest future price movement"""
        if len(closes) < 20:
            return {"score": 0.0, "direction": "NEUTRAL", "timeframe": 0}
        
        ranges = highs - lows
        avg_range = np.mean(ranges[-20:])
        
        vol_ranges = ranges[-20:] * volumes[-20:]
        avg_vol_range = np.mean(vol_ranges)
        
        close_positions = (closes[-20:] - lows[-20:]) / (ranges[-20:] + 1e-9)  # Avoid div by zero
        avg_close_position = np.mean(close_positions)
        
        recent_close_pos = np.mean(close_positions[-5:])
        earlier_close_pos = np.mean(close_positions[-20:-5])
        
        if recent_close_pos > 0.7 and earlier_close_pos < 0.5:
            direction = "BUY"
            score = 0.5 + min(recent_close_pos - earlier_close_pos, 0.5)
            timeframe = 8
        elif recent_close_pos < 0.3 and earlier_close_pos > 0.5:
            direction = "SELL"
            score = 0.5 + min(earlier_close_pos - recent_close_pos, 0.5)
            timeframe = 8
        else:
            if avg_close_position > 0.8:
                direction = "BUY"
                score = 0.6
                timeframe = 5
            elif avg_close_position < 0.2:
                direction = "SELL"
                score = 0.6
                timeframe = 5
            else:
                direction = "NEUTRAL"
                score = 0.4
                timeframe = 0
                
        return {"score": score, "direction": direction, "timeframe": timeframe}
