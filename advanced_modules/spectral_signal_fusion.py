import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SpectralComponents:
    """Container for spectral signal components"""
    emotion: float = 0.0
    volatility: float = 0.0
    entropy: float = 0.0
    quantum: float = 0.0
    trend: float = 0.0
    void: float = 0.0

class SpectralSignalFusion:
    """
    Spectral Signal Fusion - Advanced Multi-Dimensional Signal Processing
    
    Fuses emotional, volatility, entropy, quantum, trend, and void signals
    using advanced spectral analysis and Hilbert transforms.
    """
    
    def __init__(self, signal_inputs=None):
        self.signals = signal_inputs or []
        self.spectral_weights = {
            'quantum': 0.25,
            'emotional': 0.20,
            'volatility': 0.20,
            'trend': 0.15,
            'entropy': 0.10,
            'void': 0.10
        }
        self.asset_params = {
            'crypto': {'emotion_weight': 0.4, 'volatility_weight': 0.35, 'entropy_weight': 0.25},
            'forex': {'emotion_weight': 0.3, 'volatility_weight': 0.45, 'entropy_weight': 0.25},
            'commodities': {'emotion_weight': 0.25, 'volatility_weight': 0.4, 'entropy_weight': 0.35},
            'indices': {'emotion_weight': 0.2, 'volatility_weight': 0.5, 'entropy_weight': 0.3}
        }
        self.fusion_history = []
        self.spectral_cache = {}
        self.hilbert_window = 14
        
    def fuse_signals(self, asset_class: str = 'crypto', components: Optional[SpectralComponents] = None) -> float:
        """
        Fuse multi-dimensional signals using advanced spectral analysis
        
        Args:
            asset_class: Asset class for context-aware fusion
            components: SpectralComponents object with signal values
            
        Returns:
            Fused signal strength (-1 to 1)
        """
        if components is None:
            components = self._extract_components_from_signals()
            
        quantum_amplitude = getattr(components, 'quantum', 0.5)
        emotion_phase = getattr(components, 'emotion', 0.0)
        volatility_energy = getattr(components, 'volatility', 0.3)
        trend_momentum = getattr(components, 'trend', 0.5)
        entropy_chaos = getattr(components, 'entropy', 0.2)
        void_silence = getattr(components, 'void', 0.1)
        
        complex_signal = self._create_complex_signal(
            quantum_amplitude, emotion_phase, volatility_energy
        )
        
        spectral_power = self._calculate_spectral_power(complex_signal)
        
        class_multiplier = self._get_asset_class_multiplier(asset_class)
        
        fused_signal = (
            quantum_amplitude * self.spectral_weights['quantum'] * class_multiplier +
            emotion_phase * self.spectral_weights['emotional'] +
            volatility_energy * self.spectral_weights['volatility'] +
            trend_momentum * self.spectral_weights['trend'] +
            entropy_chaos * self.spectral_weights['entropy'] +
            void_silence * self.spectral_weights['void']
        ) * spectral_power
        
        final_signal = self._apply_spectral_transform(fused_signal)
        
        self.fusion_history.append({
            'timestamp': datetime.utcnow(),
            'signal': final_signal,
            'components': components,
            'asset_class': asset_class
        })
        
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]
            
        return final_signal
        
    def predict(self, data: Union[pd.DataFrame, Dict]) -> float:
        """Generate prediction signal from market data"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'Close' in data.columns:
                    closes = data['Close'].values
                else:
                    return 0.0
            else:
                return 0.0
                
            components = self._analyze_price_spectrum(closes)
            
            signal = self.fuse_signals('crypto', components)
            
            return np.tanh(signal * 2)
            
        except Exception:
            return 0.0
            
    def train(self, data):
        """Train spectral fusion model on historical data"""
        if hasattr(data, 'items'):
            for key, df in data.items():
                if hasattr(df, 'columns') and 'Close' in df.columns:
                    spectrum = self._analyze_price_spectrum(df['Close'].values)
                    self.spectral_cache[key] = spectrum
                    
    def _extract_components_from_signals(self) -> SpectralComponents:
        """Extract components from input signals"""
        if not self.signals:
            return SpectralComponents()
            
        components = SpectralComponents()
        
        for signal in self.signals:
            if isinstance(signal, dict):
                signal_type = signal.get('type', 'unknown')
                value = signal.get('value', 0.0)
                
                if signal_type == 'quantum':
                    components.quantum = value
                elif signal_type == 'emotional':
                    components.emotion = value
                elif signal_type == 'volatility':
                    components.volatility = value
                elif signal_type == 'trend':
                    components.trend = value
                elif signal_type == 'entropy':
                    components.entropy = value
                elif signal_type == 'void':
                    components.void = value
                    
        return components
        
    def _create_complex_signal(self, amplitude: float, phase: float, energy: float) -> np.ndarray:
        """Create complex signal for Hilbert analysis"""
        t = np.linspace(0, 2*np.pi, 100)
        
        real_part = amplitude * np.cos(t + phase) * (1 + 0.1 * energy)
        imag_part = amplitude * np.sin(t + phase) * (1 + 0.1 * energy)
        
        return real_part + 1j * imag_part
        
    def _calculate_spectral_power(self, complex_signal: np.ndarray) -> float:
        """Calculate spectral power using FFT"""
        try:
            fft_result = fft(complex_signal)
            fft_magnitude = np.abs(fft_result)
            power_spectrum = fft_magnitude * fft_magnitude
            
            total_power = np.sum(power_spectrum)
            
            normalized_power = min(2.0, total_power / len(complex_signal))
            
            return normalized_power
            
        except Exception:
            return 1.0
            
    def _get_asset_class_multiplier(self, asset_class: str) -> float:
        """Get asset class specific multiplier"""
        multipliers = {
            'crypto': 1.2,
            'forex': 1.0,
            'commodities': 0.9,
            'indices': 0.8,
            'stocks': 1.1
        }
        
        return multipliers.get(asset_class.lower(), 1.0)
        
    def _apply_spectral_transform(self, signal: float) -> float:
        """Apply non-linear spectral transformation"""
        transformed = np.tanh(signal * 1.5)
        
        harmonic = 0.1 * np.sin(signal * np.pi * 3)
        
        final = transformed + harmonic
        
        return max(-1.0, min(1.0, final))
        
    def _analyze_price_spectrum(self, prices) -> SpectralComponents:
        """Analyze price data to extract spectral components"""
        if len(prices) < 10:
            return SpectralComponents()
            
        returns = np.diff(prices) / prices[:-1]
        
        quantum = self._extract_quantum_component(returns)
        
        emotion = self._extract_emotional_component(returns)
        
        volatility = np.std(returns)
        
        trend = np.mean(returns)
        
        entropy = self._calculate_entropy(returns)
        
        void = self._extract_void_component(returns)
        
        return SpectralComponents(
            quantum=quantum,
            emotion=emotion,
            volatility=float(volatility),
            trend=float(trend),
            entropy=entropy,
            void=void
        )
        
    def _extract_quantum_component(self, returns: np.ndarray) -> float:
        """Extract quantum oscillation component"""
        if len(returns) < 5:
            return 0.5
            
        high_freq = returns[-5:]
        quantum_energy = np.var(high_freq) / (np.var(returns) + 1e-10)
        
        return min(1.0, float(quantum_energy))
        
    def _extract_emotional_component(self, returns: np.ndarray) -> float:
        """Extract emotional momentum component"""
        if len(returns) < 10:
            return 0.0
            
        short_momentum = np.mean(returns[-5:])
        long_momentum = np.mean(returns[-10:])
        
        emotion = (short_momentum - long_momentum) * 10
        
        return max(-1.0, min(1.0, float(emotion)))
        
    def _calculate_entropy(self, returns: np.ndarray) -> float:
        """Calculate Shannon entropy of returns"""
        if len(returns) == 0:
            return 0.0
            
        bins = np.linspace(returns.min(), returns.max(), 10)
        hist, _ = np.histogram(returns, bins=bins)
        
        probabilities = hist / len(returns)
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy / np.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
        
    def _extract_void_component(self, returns: np.ndarray) -> float:
        """Extract void (low activity) component"""
        if len(returns) < 5:
            return 0.1
            
        threshold = np.std(returns) * 0.5
        low_activity = np.abs(returns) < threshold
        
        void_ratio = np.sum(low_activity) / len(returns)
        
        return void_ratio
        
    def analyze_spectral_composition(self, prices: np.ndarray) -> Dict:
        """Analyze complete spectral composition of price data"""
        components = self._analyze_price_spectrum(prices)
        
        spectral_analysis = {
            'quantum_oscillations': components.quantum,
            'emotional_momentum': components.emotion,
            'volatility_energy': components.volatility,
            'trend_direction': components.trend,
            'entropy_chaos': components.entropy,
            'void_silence': components.void,
            'dominant_frequency': self._find_dominant_frequency(prices),
            'spectral_density': self._calculate_spectral_density(prices),
            'harmonic_content': self._analyze_harmonics(prices)
        }
        
        return spectral_analysis
        
    def _find_dominant_frequency(self, prices: np.ndarray) -> float:
        """Find dominant frequency in price data"""
        if len(prices) < 10:
            return 0.0
            
        returns = np.diff(prices) / prices[:-1]
        
        try:
            freqs, psd = welch(returns, nperseg=min(len(returns)//2, 32))
            dominant_freq_idx = np.argmax(psd)
            return freqs[dominant_freq_idx]
        except:
            return 0.0
            
    def _calculate_spectral_density(self, prices: np.ndarray) -> float:
        """Calculate spectral density"""
        if len(prices) < 10:
            return 0.0
            
        returns = np.diff(prices) / prices[:-1]
        
        try:
            freqs, psd = welch(returns, nperseg=min(len(returns)//2, 32))
            return np.mean(psd)
        except:
            return 0.0
            
    def _analyze_harmonics(self, prices: np.ndarray) -> List[float]:
        """Analyze harmonic content"""
        if len(prices) < 20:
            return [0.0, 0.0, 0.0]
            
        returns = np.diff(prices) / prices[:-1]
        
        try:
            fft_result = fft(returns)
            fft_magnitude = np.abs(fft_result)
            power_spectrum = fft_magnitude * fft_magnitude
            
            n_harmonics = min(3, len(power_spectrum) // 4)
            harmonics = []
            
            for i in range(1, n_harmonics + 1):
                harmonic_power = power_spectrum[i] if i < len(power_spectrum) else 0.0
                harmonics.append(float(harmonic_power))
                
            return harmonics
            
        except:
            return [0.0, 0.0, 0.0]
            
    def get_fusion_strength(self) -> float:
        """Get current fusion strength based on recent history"""
        if not self.fusion_history:
            return 0.5
            
        recent_signals = [entry['signal'] for entry in self.fusion_history[-10:]]
        
        return float(np.mean(np.abs(recent_signals)))
        
    def optimize_weights(self, performance_data: List[Dict]):
        """Optimize spectral weights based on performance data"""
        if len(performance_data) < 10:
            return
            
        best_performance = max(performance_data, key=lambda x: x.get('performance', 0))
        best_components = best_performance.get('components')
        
        if best_components:
            total_weight = sum(self.spectral_weights.values())
            
            for component, weight in self.spectral_weights.items():
                component_value = getattr(best_components, component, 0.0)
                adjustment = component_value * 0.1
                
                new_weight = max(0.05, min(0.5, weight + adjustment))
                self.spectral_weights[component] = new_weight
                
            current_total = sum(self.spectral_weights.values())
            normalization_factor = total_weight / current_total
            
            for component in self.spectral_weights:
                self.spectral_weights[component] *= normalization_factor
