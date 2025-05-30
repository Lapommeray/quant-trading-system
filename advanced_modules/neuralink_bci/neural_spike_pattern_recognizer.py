"""
Neural Spike Pattern Recognizer

This module applies Neuralink's brain-computer interface (BCI) spike detection algorithms
to financial market data, identifying neural-like patterns in price movements and
predicting market behavior with brain-inspired precision.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import math
from scipy import signal, stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_interface import AdvancedModuleInterface

class NeuralSpikePatternRecognizer(AdvancedModuleInterface):
    """
    Applies Neuralink's brain-computer interface (BCI) spike detection algorithms
    to financial market data, identifying neural-like patterns in price movements and
    predicting market behavior with brain-inspired precision.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Neural Spike Pattern Recognizer."""
        super().__init__(config)
        self.module_name = "NeuralSpikePatternRecognizer"
        self.module_category = "neuralink_bci"
        
        self.sampling_rate = self.config.get("sampling_rate", 1000)  # Hz
        self.detection_threshold = self.config.get("detection_threshold", 3.5)  # Standard deviations
        self.refractory_period = self.config.get("refractory_period", 2)  # Bars
        self.window_size = self.config.get("window_size", 50)  # Bars for analysis
        self.spike_history = []
        self.neuron_states = {}
        self.activation_patterns = {}
        
    def initialize(self) -> bool:
        """Initialize the Neural Spike Pattern Recognizer."""
        try:
            self._initialize_spike_detection()
            
            self._initialize_neural_network()
            
            self._initialize_pattern_recognition()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Neural Spike Pattern Recognizer: {e}")
            return False
            
    def _initialize_spike_detection(self) -> None:
        """Initialize the spike detection system"""
        self.spike_params = {
            "threshold_multiplier": 3.5,  # Standard deviations above baseline
            "min_spike_height": 0.005,  # Minimum price movement to be considered a spike
            "max_spike_width": 3,  # Maximum width of a spike in bars
            "min_spike_slope": 0.002,  # Minimum slope of a spike
            "baseline_window": 20,  # Window for calculating baseline
            "detection_window": 5,  # Window for spike detection
            "filter_order": 4,  # Order of the bandpass filter
            "filter_lowcut": 0.01,  # Low cutoff frequency (Hz)
            "filter_highcut": 0.2,  # High cutoff frequency (Hz)
        }
        
    def _initialize_neural_network(self) -> None:
        """Initialize the neural network"""
        self.neural_params = {
            "neuron_types": ["excitatory", "inhibitory"],  # Types of neurons
            "neuron_count": {"excitatory": 50, "inhibitory": 20},  # Number of neurons by type
            "connection_probability": {"excitatory": 0.3, "inhibitory": 0.5},  # Connection probability
            "synaptic_weights": {"excitatory": 0.8, "inhibitory": -0.6},  # Synaptic weights
            "activation_threshold": {"excitatory": 0.3, "inhibitory": 0.2},  # Activation threshold
            "refractory_period": {"excitatory": 3, "inhibitory": 2},  # Refractory period in bars
            "membrane_time_constant": {"excitatory": 10, "inhibitory": 5},  # Membrane time constant
            "resting_potential": -70,  # Resting membrane potential (mV)
            "spike_threshold": -55,  # Spike threshold (mV)
            "reset_potential": -75,  # Reset potential after spike (mV)
        }
        
        self.neurons = {}
        for neuron_type in self.neural_params["neuron_types"]:
            for i in range(self.neural_params["neuron_count"][neuron_type]):
                neuron_id = f"{neuron_type}_{i}"
                self.neurons[neuron_id] = {
                    "type": neuron_type,
                    "membrane_potential": self.neural_params["resting_potential"],
                    "last_spike_time": -1000,  # Long time ago
                    "spike_count": 0,
                    "connections": [],
                    "synaptic_weights": [],
                }
        
    def _initialize_pattern_recognition(self) -> None:
        """Initialize the pattern recognition system"""
        self.pattern_params = {
            "pattern_types": ["burst", "oscillation", "synchronization", "silence"],
            "pattern_windows": {"burst": 5, "oscillation": 20, "synchronization": 10, "silence": 15},
            "pattern_thresholds": {"burst": 0.7, "oscillation": 0.6, "synchronization": 0.8, "silence": 0.2},
            "min_pattern_duration": 3,  # Minimum duration of a pattern in bars
            "max_pattern_duration": 30,  # Maximum duration of a pattern in bars
            "pattern_significance": 0.95,  # Statistical significance threshold
        }
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using Neuralink spike detection algorithms."""
        if not self.initialized:
            return {"error": "Module not initialized"}
            
        try:
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            timestamps = market_data.get("timestamps", [])
            
            if not prices or len(prices) < self.window_size:
                return {"error": "Insufficient data for analysis"}
                
            preprocessed_data = self._preprocess_data(prices, volumes, timestamps)
            
            spikes = self._detect_spikes(preprocessed_data)
            
            neural_metrics = self._calculate_neural_metrics(spikes, preprocessed_data)
            
            analysis_results = {
                "spikes": spikes,
                "neural_metrics": neural_metrics,
                "timestamp": datetime.now()
            }
            
            self.last_analysis = analysis_results
            return analysis_results
            
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
            
    def _preprocess_data(self, prices: List[float], volumes: List[float], 
                        timestamps: List[datetime]) -> Dict[str, Any]:
        """Preprocess market data for spike detection."""
        returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        volume_changes = [0.0] + [(volumes[i] - volumes[i-1]) / max(float(volumes[i-1]), 1.0) for i in range(1, len(volumes))]
        
        filtered_returns = self._apply_bandpass_filter(returns)
        
        mean_return = np.mean(filtered_returns)
        std_return = np.std(filtered_returns)
        z_scores = [(r - mean_return) / float(max(float(std_return), 0.0001)) for r in filtered_returns]
        
        preprocessed_data = {
            "prices": prices,
            "volumes": volumes,
            "timestamps": timestamps,
            "returns": returns,
            "volume_changes": volume_changes,
            "filtered_returns": filtered_returns,
            "z_scores": z_scores
        }
        
        return preprocessed_data
        
    def _apply_bandpass_filter(self, data: List[float]) -> List[float]:
        """Apply a bandpass filter to the data."""
        order = self.spike_params["filter_order"]
        lowcut = self.spike_params["filter_lowcut"]
        highcut = self.spike_params["filter_highcut"]
        fs = 1.0  # Assuming 1 sample per bar
        
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = signal.filtfilt(b, a, data)
        
        return [float(x) for x in filtered_data]
        
    def _detect_spikes(self, preprocessed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect spikes in the preprocessed data."""
        z_scores = preprocessed_data["z_scores"]
        returns = preprocessed_data["returns"]
        timestamps = preprocessed_data["timestamps"]
        
        threshold = self.spike_params["threshold_multiplier"]
        min_height = self.spike_params["min_spike_height"]
        max_width = self.spike_params["max_spike_width"]
        min_slope = self.spike_params["min_spike_slope"]
        refractory_period = self.refractory_period
        
        spikes = []
        last_spike_index = -refractory_period  # Initialize to allow first spike
        
        for i in range(1, len(z_scores) - 1):
            if z_scores[i] > threshold:
                if i - last_spike_index > refractory_period:
                    if z_scores[i] > z_scores[i-1] and z_scores[i] >= z_scores[i+1]:
                        if abs(returns[i]) >= min_height:
                            slope = abs(returns[i] - returns[i-1])
                            if slope >= min_slope:
                                spike = {
                                    "index": i,
                                    "time": timestamps[i],
                                    "amplitude": returns[i],
                                    "z_score": z_scores[i],
                                    "slope": slope
                                }
                                
                                spikes.append(spike)
                                last_spike_index = i
        
        return spikes
        
    def _calculate_neural_metrics(self, spikes: List[Dict[str, Any]], 
                                preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate neural metrics based on spikes and preprocessed data."""
        neural_metrics = {}
        
        neural_metrics["spike_rate"] = len(spikes) / len(preprocessed_data["returns"]) if preprocessed_data["returns"] else 0
        
        if spikes:
            neural_metrics["avg_spike_amplitude"] = np.mean([spike["amplitude"] for spike in spikes])
            neural_metrics["max_spike_amplitude"] = max([spike["amplitude"] for spike in spikes])
        else:
            neural_metrics["avg_spike_amplitude"] = 0.0
            neural_metrics["max_spike_amplitude"] = 0.0
            
        if len(spikes) > 1:
            intervals = [spikes[i+1]["index"] - spikes[i]["index"] for i in range(len(spikes)-1)]
            neural_metrics["avg_interspike_interval"] = np.mean(intervals)
            neural_metrics["cv_interspike_interval"] = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        else:
            neural_metrics["avg_interspike_interval"] = 0.0
            neural_metrics["cv_interspike_interval"] = 0.0
            
        returns = preprocessed_data["returns"]
        if len(returns) > 1:
            acf = np.correlate(returns, returns, mode='full')
            acf = acf[len(returns)-1:] / acf[len(returns)-1]
            neural_metrics["autocorrelation"] = acf[1] if len(acf) > 1 else 0.0
            
            neural_metrics["complexity"] = self._calculate_approximate_entropy(returns)
        else:
            neural_metrics["autocorrelation"] = 0.0
            neural_metrics["complexity"] = 0.0
            
        return neural_metrics
        
    def _calculate_approximate_entropy(self, data: List[float], m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy of a time series."""
        np_data = np.array(data)
        mean_val = float(np.mean(np_data))
        std_val = float(np.std(np_data))
        
        normalized_data = data.copy()
        if std_val > 0:
            normalized_data = [(x - mean_val) / std_val for x in data]
        
        N = len(normalized_data)
        r_val = float(r * std_val)
        
        def _count_similar_patterns(m_length):
            count = np.zeros(N - m_length + 1)
            for i in range(N - m_length + 1):
                pattern = [normalized_data[i+k] for k in range(m_length)]
                for j in range(N - m_length + 1):
                    max_diff = 0.0
                    for k in range(m_length):
                        diff = abs(pattern[k] - normalized_data[j+k])
                        max_diff = max(max_diff, diff)
                    
                    if max_diff <= r_val:
                        count[i] += 1
            return count
            
        def _phi(m_length):
            count = _count_similar_patterns(m_length)
            return np.sum(np.log(count / (N - m_length + 1))) / (N - m_length + 1)
            
        if N < m + 1:
            return 0.0
            
        return abs(_phi(m) - _phi(m + 1))
        
    def get_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading signal based on market data."""
        if not self.initialized:
            return {"error": "Module not initialized"}
            
        try:
            analysis = self.analyze(market_data)
            
            if "error" in analysis:
                return {"error": analysis["error"]}
                
            spikes = analysis.get("spikes", [])
            neural_metrics = analysis.get("neural_metrics", {})
            
            direction = "NEUTRAL"
            confidence = 0.5  # Default confidence
            
            recent_spikes = [spike for spike in spikes if spike["index"] >= len(market_data["prices"]) - 5]
            
            if recent_spikes:
                latest_spike = max(recent_spikes, key=lambda s: s["index"])
                
                if latest_spike["amplitude"] > 0:
                    direction = "BUY"
                    confidence = min(0.5 + abs(latest_spike["amplitude"]) * 10, 0.95)
                else:
                    direction = "SELL"
                    confidence = min(0.5 + abs(latest_spike["amplitude"]) * 10, 0.95)
                    
                if neural_metrics.get("complexity", 0) < 0.2:
                    confidence = min(confidence + 0.1, 0.95)
                    
                if neural_metrics.get("cv_interspike_interval", 0) < 0.5:
                    confidence = min(confidence + 0.1, 0.95)
                    
            signal = {
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now(),
                "neural_metrics": neural_metrics,
                "recent_spikes": recent_spikes
            }
            
            self.last_signal = signal
            return signal
            
        except Exception as e:
            return {"error": f"Signal generation error: {e}"}
            
    def validate_signal(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trading signal against market data."""
        if not self.initialized:
            return {"error": "Module not initialized"}
            
        try:
            direction = signal.get("direction", "NEUTRAL")
            confidence = signal.get("confidence", 0.5)
            
            current_analysis = self.analyze(market_data)
            
            if "error" in current_analysis:
                return {"error": current_analysis["error"]}
                
            current_metrics = current_analysis.get("neural_metrics", {})
            
            is_valid = True
            validation_confidence = confidence
            
            if (current_metrics.get("complexity", 0) > 
                signal.get("neural_metrics", {}).get("complexity", 0) * 1.5):
                is_valid = False
                validation_confidence *= 0.5
                
            if (current_metrics.get("spike_rate", 0) > 
                signal.get("neural_metrics", {}).get("spike_rate", 0) * 2):
                is_valid = False
                validation_confidence *= 0.7
                
            validation = {
                "is_valid": is_valid,
                "original_confidence": confidence,
                "validation_confidence": validation_confidence,
                "timestamp": datetime.now()
            }
            
            return validation
            
        except Exception as e:
            return {"error": f"Signal validation error: {e}"}
