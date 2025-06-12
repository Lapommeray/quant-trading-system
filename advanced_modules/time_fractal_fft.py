import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks, hilbert
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime

@dataclass
class TimeFractalConfig:
    fft_window: int = 256
    min_cycle_length: int = 8
    max_cycle_length: int = 64
    significance_threshold: float = 0.1
    overlap_ratio: float = 0.5
    dominant_cycles_count: int = 5

class TimeFractalFFT:
    """
    Time Fractal - Market cycle analysis using Fast Fourier Transform
    
    Analyzes market cycles and fractal patterns using FFT to identify
    dominant frequencies and predict cycle-based price movements.
    """
    
    def __init__(self, config: Optional[TimeFractalConfig] = None):
        self.config = config or TimeFractalConfig()
        self.cycle_history = []
        self.dominant_frequencies = []
        self.phase_analysis = {}
        
    def analyze_cycles(self, prices: np.ndarray, sample_rate: float = 1.0) -> Dict[str, Any]:
        """
        Analyze market cycles using FFT
        
        Parameters:
        - prices: Array of price data
        - sample_rate: Sampling rate (default: 1.0 for daily data)
        
        Returns:
        - Dictionary with cycle analysis results
        """
        if len(prices) < self.config.fft_window:
            return {
                'dominant_cycles': [],
                'cycle_strength': 0.0,
                'next_turning_point': None,
                'confidence': 0.0
            }
        
        detrended_prices = self._detrend_prices(prices)
        
        windowed_data = self._apply_window(detrended_prices[-self.config.fft_window:])
        
        fft_result = fft(windowed_data)
        frequencies = fftfreq(len(windowed_data), d=1/sample_rate)
        
        fft_magnitude = np.abs(fft_result)
        power_spectrum = fft_magnitude * fft_magnitude
        
        dominant_cycles = self._find_dominant_cycles(frequencies, power_spectrum)
        
        cycle_strength = self._calculate_cycle_strength(power_spectrum)
        
        phase_info = self._analyze_phases(fft_result, frequencies, dominant_cycles)
        
        next_turning_point = self._predict_turning_point(dominant_cycles, phase_info)
        
        confidence = self._calculate_confidence(dominant_cycles, cycle_strength)
        
        result = {
            'dominant_cycles': dominant_cycles,
            'cycle_strength': cycle_strength,
            'next_turning_point': next_turning_point,
            'confidence': confidence,
            'phase_analysis': phase_info,
            'timestamp': datetime.now().isoformat()
        }
        
        self.cycle_history.append(result)
        if len(self.cycle_history) > 100:
            self.cycle_history = self.cycle_history[-100:]
        
        return result
    
    def _detrend_prices(self, prices: np.ndarray) -> np.ndarray:
        """Remove linear trend from price data"""
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend = np.polyval(coeffs, x)
        return prices - trend
    
    def _apply_window(self, data: np.ndarray) -> np.ndarray:
        """Apply Hanning window to reduce spectral leakage"""
        window = np.hanning(len(data))
        return data * window
    
    def _find_dominant_cycles(self, frequencies: np.ndarray, power_spectrum: np.ndarray) -> List[Dict[str, float]]:
        """Find dominant cycles in the frequency domain"""
        positive_freqs = frequencies[frequencies > 0]
        positive_power = power_spectrum[frequencies > 0]
        
        cycle_lengths = 1.0 / positive_freqs
        
        valid_mask = (cycle_lengths >= self.config.min_cycle_length) & (cycle_lengths <= self.config.max_cycle_length)
        valid_cycles = cycle_lengths[valid_mask]
        valid_power = positive_power[valid_mask]
        
        if len(valid_power) == 0:
            return []
        
        peaks, properties = find_peaks(valid_power, height=np.max(valid_power) * self.config.significance_threshold)
        
        dominant_cycles = []
        for i in peaks[:self.config.dominant_cycles_count]:
            cycle_length = valid_cycles[i]
            power = valid_power[i]
            frequency = 1.0 / cycle_length
            
            dominant_cycles.append({
                'cycle_length': float(cycle_length),
                'frequency': float(frequency),
                'power': float(power),
                'relative_strength': float(power / np.max(valid_power))
            })
        
        return sorted(dominant_cycles, key=lambda x: x['power'], reverse=True)
    
    def _calculate_cycle_strength(self, power_spectrum: np.ndarray) -> float:
        """Calculate overall cycle strength"""
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        
        dc_component = power_spectrum[0]
        ac_power = total_power - dc_component
        
        cycle_strength = ac_power / total_power
        return min(1.0, max(0.0, float(cycle_strength)))
    
    def _analyze_phases(self, fft_result, frequencies: np.ndarray, dominant_cycles: List[Dict]) -> Dict[str, Any]:
        """Analyze phase information for dominant cycles"""
        phase_info = {}
        
        for cycle in dominant_cycles:
            freq = cycle['frequency']
            
            freq_idx = np.argmin(np.abs(frequencies - freq))
            
            complex_amplitude = fft_result[freq_idx]
            phase = np.angle(complex_amplitude)
            amplitude = np.abs(complex_amplitude)
            
            phase_info[f"cycle_{cycle['cycle_length']:.1f}"] = {
                'phase': float(phase),
                'amplitude': float(amplitude),
                'phase_degrees': float(np.degrees(phase))
            }
        
        return phase_info
    
    def _predict_turning_point(self, dominant_cycles: List[Dict], phase_info: Dict) -> Optional[Dict[str, Any]]:
        """Predict next turning point based on cycle analysis"""
        if not dominant_cycles:
            return None
        
        strongest_cycle = dominant_cycles[0]
        cycle_length = strongest_cycle['cycle_length']
        
        phase_key = f"cycle_{cycle_length:.1f}"
        if phase_key not in phase_info:
            return None
        
        current_phase = phase_info[phase_key]['phase']
        
        phase_to_peak = np.pi/2 - current_phase
        if phase_to_peak < 0:
            phase_to_peak += 2 * np.pi
        
        phase_to_trough = -np.pi/2 - current_phase
        if phase_to_trough < 0:
            phase_to_trough += 2 * np.pi
        
        bars_to_peak = (phase_to_peak / (2 * np.pi)) * cycle_length
        bars_to_trough = (phase_to_trough / (2 * np.pi)) * cycle_length
        
        if bars_to_peak < bars_to_trough:
            return {
                'type': 'peak',
                'bars_ahead': float(bars_to_peak),
                'cycle_length': cycle_length,
                'confidence': strongest_cycle['relative_strength']
            }
        else:
            return {
                'type': 'trough',
                'bars_ahead': float(bars_to_trough),
                'cycle_length': cycle_length,
                'confidence': strongest_cycle['relative_strength']
            }
    
    def _calculate_confidence(self, dominant_cycles: List[Dict], cycle_strength: float) -> float:
        """Calculate confidence in cycle analysis"""
        if not dominant_cycles:
            return 0.0
        
        strongest_cycle_power = dominant_cycles[0]['relative_strength']
        
        cycle_consistency = len([c for c in dominant_cycles if c['relative_strength'] > 0.3])
        consistency_factor = min(1.0, cycle_consistency / 3.0)
        
        confidence = (strongest_cycle_power * 0.6 + cycle_strength * 0.3 + consistency_factor * 0.1)
        
        return min(1.0, max(0.0, float(confidence)))
    
    def get_cycle_summary(self) -> Dict[str, Any]:
        """Get summary of recent cycle analysis"""
        if not self.cycle_history:
            return {'status': 'no_data'}
        
        recent_analysis = self.cycle_history[-1]
        
        avg_confidence = np.mean([h['confidence'] for h in self.cycle_history[-10:]])
        
        dominant_cycle_lengths = []
        for history in self.cycle_history[-5:]:
            if history['dominant_cycles']:
                dominant_cycle_lengths.append(history['dominant_cycles'][0]['cycle_length'])
        
        stable_cycles = len(set(np.round(dominant_cycle_lengths))) <= 2 if dominant_cycle_lengths else False
        
        return {
            'status': 'active',
            'current_confidence': recent_analysis['confidence'],
            'average_confidence': float(avg_confidence),
            'stable_cycles': stable_cycles,
            'dominant_cycle_length': dominant_cycle_lengths[-1] if dominant_cycle_lengths else None,
            'next_turning_point': recent_analysis['next_turning_point']
        }
    
    def predict_price_direction(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Predict price direction based on fractal cycle analysis
        
        Parameters:
        - prices: Array of recent price data
        
        Returns:
        - Dictionary with direction prediction
        """
        cycle_analysis = self.analyze_cycles(prices)
        
        if cycle_analysis['confidence'] < 0.3:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'Insufficient cycle clarity'
            }
        
        turning_point = cycle_analysis['next_turning_point']
        if not turning_point:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'reason': 'No clear turning point detected'
            }
        
        bars_ahead = turning_point['bars_ahead']
        turning_type = turning_point['type']
        
        if bars_ahead <= 3:
            if turning_type == 'peak':
                direction = 'SELL'
                reason = f'Peak expected in {bars_ahead:.1f} bars'
            else:
                direction = 'BUY'
                reason = f'Trough expected in {bars_ahead:.1f} bars'
        elif bars_ahead <= 10:
            if turning_type == 'peak':
                direction = 'BUY'
                reason = f'Approaching peak in {bars_ahead:.1f} bars'
            else:
                direction = 'SELL'
                reason = f'Approaching trough in {bars_ahead:.1f} bars'
        else:
            direction = 'NEUTRAL'
            reason = f'Turning point too far ahead ({bars_ahead:.1f} bars)'
        
        confidence = min(0.9, cycle_analysis['confidence'] * turning_point['confidence'])
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'reason': reason,
            'turning_point': turning_point,
            'cycle_analysis': cycle_analysis
        }
