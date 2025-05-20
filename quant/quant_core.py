
import numpy as np
import pandas as pd
from datetime import datetime
import json

from quant.liquidity_mirror import LiquidityMirror
from quant.time_fractal import TimeFractal
from quant.entropy_shield import EntropyShield

class QuantCore:
    """
    Quant Foundation - Institutional-Grade Strategies
    
    Integrates multiple quantitative modules:
    - Liquidity Mirror Scanner (detects hidden order blocks)
    - Time Fractal Predictor (Fibonacci + machine learning)
    - Entropy Shield (adaptive risk management)
    """
    
    def __init__(self, min_imbalance=2.0, min_cycle=14, max_risk=0.02):
        """
        Initialize the Quant Core
        
        Parameters:
        - min_imbalance: Minimum liquidity imbalance to detect (default: 2.0)
        - min_cycle: Minimum cycle length in candles (default: 14)
        - max_risk: Maximum risk per trade as decimal (default: 0.02 = 2%)
        """
        self.liquidity_mirror = LiquidityMirror(min_imbalance=min_imbalance)
        self.time_fractal = TimeFractal(min_cycle=min_cycle)
        self.entropy_shield = EntropyShield(max_risk=max_risk)
        
    def analyze_market(self, price_data, order_book=None, account_size=None):
        """
        Comprehensive market analysis using all quant modules
        
        Parameters:
        - price_data: DataFrame with OHLCV data
        - order_book: Optional order book data for liquidity analysis
        - account_size: Optional account size for position sizing
        
        Returns:
        - Dictionary with comprehensive analysis results
        """
        if not isinstance(price_data, pd.DataFrame):
            raise ValueError("price_data must be a pandas DataFrame")
            
        if not all(col in price_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("price_data must contain 'open', 'high', 'low', 'close', 'volume' columns")
            
        closes = price_data['close'].values
        highs = price_data['high'].values
        lows = price_data['low'].values
        volumes = price_data['volume'].values
        
        atr = self._calculate_atr(highs, lows, closes)
        
        fractal_analysis = self.time_fractal.detect_fractals(closes)
        
        entropy = self.entropy_shield.calc_entropy(closes)
        market_state = self.entropy_shield.analyze_market_state(closes, volumes, highs, lows)
        
        liquidity_analysis = None
        if order_book is not None:
            liquidity_analysis = self.liquidity_mirror.analyze_order_book(order_book)
        
        position_sizing = None
        if account_size is not None and len(closes) > 0:
            position_sizing = self.entropy_shield.position_size(
                entropy, 
                account_size, 
                closes[-1]
            )
        
        future_prediction = self.time_fractal.predict_future(closes, periods_ahead=10)
        
        similar_patterns = self.time_fractal.find_similar_patterns(closes)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(closes[-1]) if len(closes) > 0 else None,
            'fractal_analysis': fractal_analysis,
            'market_state': market_state,
            'liquidity_analysis': liquidity_analysis,
            'position_sizing': position_sizing,
            'future_prediction': future_prediction.tolist() if future_prediction is not None else None,
            'similar_patterns': similar_patterns,
            'entropy': float(entropy)
        }
    
    def _calculate_atr(self, highs, lows, closes, period=14):
        """
        Calculate Average True Range
        
        Parameters:
        - highs: Array of high prices
        - lows: Array of low prices
        - closes: Array of closing prices
        - period: ATR period (default: 14)
        
        Returns:
        - Array of ATR values
        """
        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("Input arrays must have the same length")
            
        tr = np.zeros(len(highs))
        
        for i in range(1, len(highs)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        atr = np.zeros(len(highs))
        atr[period-1] = np.mean(tr[1:period])
        
        for i in range(period, len(highs)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        return atr
        
    def generate_trading_signal(self, price_data, order_book=None, account_size=None):
        """
        Generate a comprehensive trading signal
        
        Parameters:
        - price_data: DataFrame with OHLCV data
        - order_book: Optional order book data for liquidity analysis
        - account_size: Optional account size for position sizing
        
        Returns:
        - Dictionary with trading signal information
        """
        analysis = self.analyze_market(price_data, order_book, account_size)
        
        current_price = analysis['current_price']
        market_state = analysis['market_state']
        fractal_analysis = analysis['fractal_analysis']
        liquidity_analysis = analysis['liquidity_analysis']
        
        signal = None
        direction = None
        strength = 0
        confidence = 0
        
        if fractal_analysis and 'primary_cycle' in fractal_analysis and fractal_analysis['primary_cycle']:
            cycle_length = fractal_analysis['primary_cycle']
            
            if len(price_data) % cycle_length < 2:
                direction = "BUY"
                strength += 0.3
            elif len(price_data) % cycle_length > cycle_length - 2:
                direction = "SELL"
                strength += 0.3
        
        if liquidity_analysis and 'signal' in liquidity_analysis:
            if liquidity_analysis['signal'] == "HIDDEN BIDS DETECTED" and direction == "BUY":
                strength += 0.3
                confidence += 0.2
            elif liquidity_analysis['signal'] == "HIDDEN ASKS DETECTED" and direction == "SELL":
                strength += 0.3
                confidence += 0.2
        
        if market_state and 'market_state' in market_state:
            if market_state['market_state'] == "HIGH CHAOS":
                strength *= 0.5  # Reduce strength in high chaos
                confidence *= 0.7  # Reduce confidence in high chaos
            elif market_state['market_state'] == "LOW CHAOS":
                confidence += 0.1  # Increase confidence in low chaos
        
        if strength > 0.5 and direction:
            signal = direction
            
        position_size = None
        if account_size is not None and signal and current_price:
            position_size = analysis['position_sizing']['position_size'] if analysis['position_sizing'] else None
        
        return {
            'signal': signal,
            'direction': direction,
            'strength': min(1.0, strength),
            'confidence': min(1.0, confidence),
            'price': current_price,
            'position_size': position_size,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis
        }
        
    def save_analysis(self, analysis, filename='quant_analysis.json'):
        """
        Save quant analysis to a JSON file
        
        Parameters:
        - analysis: Analysis results dictionary
        - filename: Output filename (default: 'quant_analysis.json')
        """
        if 'future_prediction' in analysis and analysis['future_prediction'] is not None:
            if isinstance(analysis['future_prediction'], np.ndarray):
                analysis['future_prediction'] = analysis['future_prediction'].tolist()
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
            
    def load_analysis(self, filename='quant_analysis.json'):
        """
        Load quant analysis from a JSON file
        
        Parameters:
        - filename: Input filename (default: 'quant_analysis.json')
        
        Returns:
        - Analysis results dictionary
        """
        with open(filename, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'high': 100 + np.cumsum(np.random.normal(0, 1, 100)) + np.random.normal(0, 0.5, 100),
        'low': 100 + np.cumsum(np.random.normal(0, 1, 100)) - np.random.normal(0, 0.5, 100),
        'close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)
    
    order_book = {
        'bids': [[99.5, 500], [99.0, 300], [98.5, 200]],
        'asks': [[100.5, 200], [101.0, 300], [101.5, 100]]
    }
    
    quant_core = QuantCore()
    
    signal = quant_core.generate_trading_signal(price_data, order_book, account_size=10000)
    print(f"Signal: {signal['signal']}")
    print(f"Strength: {signal['strength']:.2f}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Position Size: {signal['position_size']:.2f} units")
