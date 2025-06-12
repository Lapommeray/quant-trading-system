
import numpy as np
import pandas as pd
from datetime import datetime
import math

class VeveTriggers:
    """
    Vèvè Market Triggers - Sacred Geometry Integration
    
    Features:
    - Papa Legba's Crossroads Signal (breakout detection)
    - Erzulie Freda's Love Cycle (mean-reversion algo)
    - Baron Samedi's Death Zone (volatility collapse alert)
    """
    
    def __init__(self):
        """
        Initialize the Vèvè Market Triggers system
        """
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.sacred_ratios = {
            'legba': 1.618,      # Golden ratio (phi)
            'erzulie': 0.618,    # 1/phi
            'samedi': 2.618,     # phi^2
            'crossroads': 0.382  # 1 - 0.618
        }
        
    def papa_legba_crossroads(self, prices, volumes, period=21):
        """
        Papa Legba's Crossroads Signal (breakout detection)
        
        Parameters:
        - prices: Array of price data
        - volumes: Array of volume data
        - period: EMA period (default: 21)
        
        Returns:
        - Dictionary with signal information
        """
        if len(prices) < period or len(volumes) < period:
            return {'signal': None, 'strength': 0, 'message': 'Insufficient data'}
            
        ema = np.mean(prices[-period:])
        
        crossover = prices[-1] > ema and prices[-2] <= ema
        
        avg_volume = np.mean(volumes[-period:])
        volume_surge = volumes[-1] > 1.5 * avg_volume
        
        distance = (prices[-1] - ema) / ema
        strength = min(1.0, distance * 100)
        
        if crossover and volume_surge:
            return {
                'signal': 'GATE OPEN',
                'strength': strength,
                'message': 'Papa Legba opens the gate. Crossroads aligned with volume surge.',
                'ema': ema,
                'price': prices[-1],
                'volume_ratio': volumes[-1] / avg_volume
            }
        elif crossover:
            return {
                'signal': 'GATE PARTIAL',
                'strength': strength * 0.7,
                'message': 'Papa Legba partially opens the gate. Crossroads aligned but volume weak.',
                'ema': ema,
                'price': prices[-1],
                'volume_ratio': volumes[-1] / avg_volume
            }
        else:
            return {
                'signal': None,
                'strength': 0,
                'message': 'No crossroads alignment detected.',
                'ema': ema,
                'price': prices[-1],
                'volume_ratio': volumes[-1] / avg_volume
            }
    
    def erzulie_freda_cycle(self, prices, period=14):
        """
        Erzulie Freda's Love Cycle (mean-reversion algo)
        
        Parameters:
        - prices: Array of price data
        - period: RSI period (default: 14)
        
        Returns:
        - Dictionary with signal information
        """
        if len(prices) < period + 1:
            return {'signal': None, 'strength': 0, 'message': 'Insufficient data'}
            
        changes = np.diff(prices)
        
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        love_level = self.sacred_ratios['erzulie'] * 100  # ~61.8
        rejection_level = 100 - love_level  # ~38.2
        
        if rsi < love_level:
            strength = min(1.0, float((love_level - rsi) / love_level))
            signal = 'LOVE EMBRACE'
            message = 'Erzulie Freda embraces with love. Mean reversion upward likely.'
        elif rsi > rejection_level:
            strength = min(1.0, float((rsi - rejection_level) / (100 - rejection_level)))
            signal = 'LOVE REJECTION'
            message = 'Erzulie Freda rejects with passion. Mean reversion downward likely.'
        else:
            strength = 0
            signal = None
            message = 'Erzulie Freda is neutral. No strong mean reversion signal.'
        
        return {
            'signal': signal,
            'strength': strength,
            'message': message,
            'rsi': rsi,
            'love_level': love_level,
            'rejection_level': rejection_level
        }
    
    def baron_samedi_death_zone(self, high_prices, low_prices, close_prices, period=14):
        """
        Baron Samedi's Death Zone (volatility collapse alert)
        
        Parameters:
        - high_prices: Array of high prices
        - low_prices: Array of low prices
        - close_prices: Array of closing prices
        - period: ATR period (default: 14)
        
        Returns:
        - Dictionary with signal information
        """
        if len(high_prices) < period + 10 or len(low_prices) < period + 10 or len(close_prices) < period + 10:
            return {'signal': None, 'strength': 0, 'message': 'Insufficient data'}
            
        tr = np.zeros(len(high_prices))
        for i in range(1, len(high_prices)):
            tr[i] = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i-1]),
                abs(low_prices[i] - close_prices[i-1])
            )
        
        atr = np.zeros(len(high_prices))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high_prices)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        recent_atr = atr[-1]
        historical_atr = np.mean(atr[-10:-1])
        
        collapse_ratio = recent_atr / historical_atr
        
        death_threshold = 1 / self.sacred_ratios['samedi']  # ~0.382
        
        if collapse_ratio < death_threshold:
            strength = min(1.0, (death_threshold - collapse_ratio) / death_threshold)
            signal = 'DEATH ZONE'
            message = 'Baron Samedi marks the death zone. Volatility collapse detected, explosive move imminent.'
        else:
            strength = 0
            signal = None
            message = 'No death zone detected. Volatility is normal.'
        
        return {
            'signal': signal,
            'strength': strength,
            'message': message,
            'collapse_ratio': collapse_ratio,
            'death_threshold': death_threshold,
            'recent_atr': recent_atr,
            'historical_atr': historical_atr
        }
    
    def analyze_market(self, df):
        """
        Analyze market data using all Vèvè triggers
        
        Parameters:
        - df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        
        Returns:
        - Dictionary with all signal information
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close', 'volume' columns")
            
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        legba_signal = self.papa_legba_crossroads(closes, volumes)
        erzulie_signal = self.erzulie_freda_cycle(closes)
        samedi_signal = self.baron_samedi_death_zone(highs, lows, closes)
        
        combined_strength = 0
        active_signals = []
        
        if legba_signal['signal']:
            combined_strength += legba_signal['strength']
            active_signals.append(legba_signal['signal'])
            
        if erzulie_signal['signal']:
            combined_strength += erzulie_signal['strength']
            active_signals.append(erzulie_signal['signal'])
            
        if samedi_signal['signal']:
            combined_strength += samedi_signal['strength'] * 1.5  # Baron Samedi has higher weight
            active_signals.append(samedi_signal['signal'])
        
        if active_signals:
            combined_strength = min(1.0, combined_strength / (len(active_signals) + 0.5))
        
        return {
            'legba': legba_signal,
            'erzulie': erzulie_signal,
            'samedi': samedi_signal,
            'combined_strength': combined_strength,
            'active_signals': active_signals,
            'timestamp': datetime.now().isoformat()
        }
