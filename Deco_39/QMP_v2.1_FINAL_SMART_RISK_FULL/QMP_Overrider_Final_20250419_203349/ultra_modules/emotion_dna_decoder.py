# Decodes emotional DNA embedded in market candle structure.

import numpy as np

class EmotionDNADecoder:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.lookback_period = 25  # Number of candles to analyze
        self.fear_threshold = 0.7
        self.greed_threshold = 0.8
        
    def decode(self, symbol, history_window):
        """
        Analyzes candle patterns to detect emotional signatures in the market.
        
        Parameters:
        - symbol: The trading symbol
        - history_window: List of TradeBars (e.g., 25 1-minute bars)
        
        Returns:
        - True if emotional state is favorable for trading, False otherwise
        """
        if len(history_window) < self.lookback_period:
            self.algo.Debug(f"EmotionDNA: Insufficient history for {symbol}")
            return False
            
        opens = np.array([bar.Open for bar in history_window])
        highs = np.array([bar.High for bar in history_window])
        lows = np.array([bar.Low for bar in history_window])
        closes = np.array([bar.Close for bar in history_window])
        
        body_sizes = np.abs(closes - opens)
        ranges = highs - lows
        body_to_range_ratios = body_sizes / (ranges + 1e-9)  # Avoid division by zero
        
        upper_wicks = np.zeros_like(body_sizes)
        lower_wicks = np.zeros_like(body_sizes)
        
        for i in range(len(history_window)):
            if closes[i] >= opens[i]:  # Bullish candle
                upper_wicks[i] = highs[i] - closes[i]
                lower_wicks[i] = opens[i] - lows[i]
            else:  # Bearish candle
                upper_wicks[i] = highs[i] - opens[i]
                lower_wicks[i] = closes[i] - lows[i]
        
        fear_score = self._calculate_fear_score(body_to_range_ratios, upper_wicks, lower_wicks)
        greed_score = self._calculate_greed_score(body_to_range_ratios, upper_wicks, lower_wicks)
        
        is_favorable = (fear_score < self.fear_threshold) and (greed_score < self.greed_threshold)
        
        self.algo.Debug(f"EmotionDNA: {symbol} - Fear: {fear_score:.2f}, Greed: {greed_score:.2f}, Favorable: {is_favorable}")
        return is_favorable
    
    def _calculate_fear_score(self, body_ratios, upper_wicks, lower_wicks):
        """Calculate fear score based on candle patterns"""
        
        small_body_count = np.sum(body_ratios < 0.3)
        long_lower_wick_count = np.sum(lower_wicks > np.mean(lower_wicks) * 1.5)
        
        volatility = np.std(body_ratios)
        recent_volatility = np.std(body_ratios[-5:])
        volatility_increasing = recent_volatility > volatility
        
        fear_score = (
            (small_body_count / len(body_ratios)) * 0.4 +
            (long_lower_wick_count / len(lower_wicks)) * 0.4 +
            (0.2 if volatility_increasing else 0)
        )
        
        return fear_score
    
    def _calculate_greed_score(self, body_ratios, upper_wicks, lower_wicks):
        """Calculate greed score based on candle patterns"""
        
        large_body_count = np.sum(body_ratios > 0.7)
        small_upper_wick_count = np.sum(upper_wicks < np.mean(upper_wicks) * 0.5)
        
        consecutive_count = 0
        max_consecutive = 0
        
        for i in range(1, len(body_ratios)):
            if (body_ratios[i] > 0.5 and body_ratios[i-1] > 0.5) or \
               (body_ratios[i] < 0.5 and body_ratios[i-1] < 0.5):
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        greed_score = (
            (large_body_count / len(body_ratios)) * 0.4 +
            (small_upper_wick_count / len(upper_wicks)) * 0.3 +
            (min(max_consecutive / 5, 1.0) * 0.3)  # Cap at 1.0
        )
        
        return greed_score
