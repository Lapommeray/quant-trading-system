
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.fft import fft
import json

class TimeFractal:
    """
    Time Fractal Predictor
    
    Uses Fast Fourier Transform (FFT) to identify dominant cycles in price data
    and predict future price movements based on fractal patterns.
    """
    
    def __init__(self, min_cycle=14, max_cycle=120):
        """
        Initialize the Time Fractal Predictor
        
        Parameters:
        - min_cycle: Minimum cycle length in candles (default: 14)
        - max_cycle: Maximum cycle length in candles (default: 120)
        """
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.detected_cycles = []
        
    def detect_fractals(self, prices):
        """
        Uses FFT to identify dominant cycles in price data
        
        Parameters:
        - prices: Array of price data
        
        Returns:
        - Dictionary with cycle information
        """
        if len(prices) < self.max_cycle:
            return {'primary_cycle': None, 'message': 'Insufficient data'}
            
        detrended = self._detrend(prices)
        
        spectrum = fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        
        power = np.abs(spectrum)**2
        
        dominant_indices = np.argsort(power[1:len(power)//2])[-3:][::-1] + 1
        
        cycles = []
        for idx in dominant_indices:
            cycle_length = int(1.0 / abs(freqs[idx]))
            
            if self.min_cycle <= cycle_length <= self.max_cycle:
                phase = np.angle(spectrum[idx])
                
                confidence = power[idx] / np.sum(power[1:len(power)//2])
                
                cycles.append({
                    'length': cycle_length,
                    'power': float(power[idx]),
                    'phase': float(phase),
                    'confidence': float(confidence)
                })
        
        cycles.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.detected_cycles = cycles
        
        if cycles:
            primary_cycle = cycles[0]['length']
            return {
                'primary_cycle': primary_cycle,
                'all_cycles': cycles,
                'message': f'Primary cycle of {primary_cycle} candles detected with {cycles[0]["confidence"]:.2%} confidence'
            }
        else:
            return {
                'primary_cycle': None,
                'all_cycles': [],
                'message': 'No significant cycles detected'
            }
    
    def _detrend(self, prices):
        """
        Remove linear trend from price data
        
        Parameters:
        - prices: Array of price data
        
        Returns:
        - Detrended price data
        """
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend = slope * x + intercept
        return prices - trend
        
    def predict_future(self, prices, periods_ahead=10):
        """
        Predict future prices based on detected cycles
        
        Parameters:
        - prices: Array of price data
        - periods_ahead: Number of periods to predict (default: 10)
        
        Returns:
        - Array of predicted prices
        """
        if not self.detected_cycles:
            self.detect_fractals(prices)
            
        if not self.detected_cycles:
            return None
            
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        
        predictions = np.zeros(periods_ahead)
        
        future_x = np.arange(len(prices), len(prices) + periods_ahead)
        trend_component = slope * future_x + intercept
        
        for cycle in self.detected_cycles:
            cycle_length = cycle['length']
            phase = cycle['phase']
            power = np.sqrt(cycle['power'])
            
            for i in range(periods_ahead):
                position = (len(prices) + i) % cycle_length
                cycle_position = 2 * np.pi * position / cycle_length + phase
                predictions[i] += power * np.sin(cycle_position)
        
        predictions += trend_component
        
        return predictions
        
    def find_similar_patterns(self, prices, lookback_window=30, num_patterns=3):
        """
        Find historical patterns similar to the current market structure
        
        Parameters:
        - prices: Array of price data
        - lookback_window: Window size to compare (default: 30)
        - num_patterns: Number of similar patterns to return (default: 3)
        
        Returns:
        - List of dictionaries with similar pattern information
        """
        if len(prices) < lookback_window * 2:
            return []
            
        current_pattern = prices[-lookback_window:]
        
        current_normalized = (current_pattern - np.mean(current_pattern)) / np.std(current_pattern)
        
        similarities = []
        
        for i in range(len(prices) - 2 * lookback_window):
            historical_pattern = prices[i:i+lookback_window]
            
            historical_normalized = (historical_pattern - np.mean(historical_pattern)) / np.std(historical_pattern)
            
            correlation = np.corrcoef(current_normalized, historical_normalized)[0, 1]
            
            distance = np.sqrt(np.sum((current_normalized - historical_normalized) ** 2))
            
            similarity = correlation / (1 + distance)
            
            similarities.append({
                'start_idx': i,
                'end_idx': i + lookback_window,
                'correlation': float(correlation),
                'distance': float(distance),
                'similarity': float(similarity),
                'future_returns': float(prices[i+lookback_window] / prices[i+lookback_window-1] - 1)
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:num_patterns]
        
    def plot_cycles(self, prices, filename=None):
        """
        Plot price data with detected cycles
        
        Parameters:
        - prices: Array of price data
        - filename: Optional filename to save the plot
        """
        if not self.detected_cycles:
            self.detect_fractals(prices)
            
        if not self.detected_cycles:
            return
            
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(prices, label='Price')
        plt.title('Price Data with Detected Cycles')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        
        x = np.arange(len(prices))
        
        for i, cycle in enumerate(self.detected_cycles[:3]):
            cycle_length = cycle['length']
            phase = cycle['phase']
            power = np.sqrt(cycle['power'])
            
            cycle_component = np.zeros(len(prices))
            for j in range(len(prices)):
                cycle_position = 2 * np.pi * j / cycle_length + phase
                cycle_component[j] = power * np.sin(cycle_position)
            
            plt.plot(x, cycle_component, label=f'Cycle {cycle_length} candles')
        
        plt.title('Detected Cycles')
        plt.legend()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def save_analysis(self, analysis, filename='fractal_analysis.json'):
        """
        Save fractal analysis to a JSON file
        
        Parameters:
        - analysis: Analysis results dictionary
        - filename: Output filename (default: 'fractal_analysis.json')
        """
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
            
    def load_analysis(self, filename='fractal_analysis.json'):
        """
        Load fractal analysis from a JSON file
        
        Parameters:
        - filename: Input filename (default: 'fractal_analysis.json')
        
        Returns:
        - Analysis results dictionary
        """
        with open(filename, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    np.random.seed(42)
    t = np.arange(200)
    
    cycle1 = 10 * np.sin(2 * np.pi * t / 21)
    cycle2 = 5 * np.sin(2 * np.pi * t / 55)
    
    trend = 0.1 * t
    noise = np.random.normal(0, 2, len(t))
    
    prices = 100 + trend + cycle1 + cycle2 + noise
    
    fractal = TimeFractal()
    
    cycles = fractal.detect_fractals(prices)
    print(f"Detected cycles: {cycles}")
    
    future = fractal.predict_future(prices, 20)
    print(f"Future predictions: {future}")
    
    patterns = fractal.find_similar_patterns(prices)
    print(f"Similar patterns: {patterns}")
