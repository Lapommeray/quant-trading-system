
import numpy as np
from datetime import datetime
import pandas as pd

class LegbaCrossroads:
    """
    Legba Crossroads Algorithm - Sacred Breakout Detector
    
    Features:
    - EMA 21 (Legba's Time Gate)
    - Volume surge (Spirit confirmation)
    - Baron Samedi Chaos Filter (volatility rejection)
    - Dynamic EMA Windows (session-aware)
    """
    
    def __init__(self, ema_period=21, volume_mult=1.5, chaos_threshold=2.0):
        """
        Initialize the Legba Crossroads Algorithm
        
        Parameters:
        - ema_period: EMA period for NY session (default: 21)
        - volume_mult: Volume surge multiplier (default: 1.5)
        - chaos_threshold: Baron Samedi chaos threshold (default: 2.0)
        """
        self.ema_period = ema_period
        self.volume_mult = volume_mult
        self.chaos_threshold = chaos_threshold
        self.asia_ema_period = 13  # Shorter EMA for Asia session
        self.asia_volume_mult = 2.0  # Higher volume requirement for Asia session
        
    def _get_session(self, timestamp=None):
        """
        Determine the current trading session (NY or Asia)
        
        Parameters:
        - timestamp: Optional timestamp to check (default: current time)
        
        Returns:
        - Session name ("NY" or "Asia")
        """
        if timestamp is None:
            now = datetime.now()
        else:
            now = timestamp
            
        hour = (now.hour - 5) % 24  # Adjust for EST
        
        if 9 <= hour < 16:
            return "NY"
        else:
            return "Asia"
    
    def detect_breakout(self, close_prices, volume, atr, timestamp=None):
        """
        Detects sacred breakouts using:
        - Dynamic EMA (session-aware)
        - Volume surge (Spirit confirmation)
        - Baron Samedi Chaos Filter (rejects high volatility)
        
        Parameters:
        - close_prices: Array of closing prices
        - volume: Array of volume data
        - atr: Array of Average True Range values
        - timestamp: Optional timestamp for session detection
        
        Returns:
        - "⚡GATE OPEN⚡" if breakout is sacred
        - None if no signal
        """
        if len(close_prices) < 30 or len(volume) < 30 or len(atr) < 30:
            return None  # Not enough data
            
        session = self._get_session(timestamp)
        
        ema_period = self.ema_period if session == "NY" else self.asia_ema_period
        
        ema = np.mean(close_prices[-ema_period:])
        
        median_atr = np.median(atr[-14:])
        if atr[-1] > self.chaos_threshold * median_atr:
            return None  # CHAOS DETECTED - NO TRADE
        
        vol_mult = self.volume_mult if session == "NY" else self.asia_volume_mult
        volume_ok = volume[-1] > vol_mult * np.mean(volume[-14:])
        
        if (close_prices[-1] > ema) and volume_ok:
            return "⚡GATE OPEN⚡"
        
        return None
        
    def detect_breakout_df(self, df, timestamp=None):
        """
        Detect breakouts using a pandas DataFrame
        
        Parameters:
        - df: DataFrame with 'close', 'volume', and 'atr' columns
        - timestamp: Optional timestamp for session detection
        
        Returns:
        - "⚡GATE OPEN⚡" if breakout is sacred
        - None if no signal
        """
        if 'close' not in df.columns or 'volume' not in df.columns or 'atr' not in df.columns:
            raise ValueError("DataFrame must contain 'close', 'volume', and 'atr' columns")
            
        close_prices = df['close'].values
        volume = df['volume'].values
        atr = df['atr'].values
        
        return self.detect_breakout(close_prices, volume, atr, timestamp)
        
    def calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range
        
        Parameters:
        - high: Array of high prices
        - low: Array of low prices
        - close: Array of closing prices
        - period: ATR period (default: 14)
        
        Returns:
        - Array of ATR values
        """
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("Input arrays must have the same length")
            
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = np.zeros(len(high))
        atr[period-1] = np.mean(tr[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr
