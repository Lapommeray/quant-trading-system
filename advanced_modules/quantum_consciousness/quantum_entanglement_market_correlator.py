"""
Quantum Entanglement Market Correlator

This module applies quantum entanglement principles to detect and exploit
non-local correlations between financial instruments across markets and timeframes.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import math
from scipy import stats
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module_interface import AdvancedModuleInterface

class QuantumEntanglementMarketCorrelator(AdvancedModuleInterface):
    """
    Applies quantum entanglement principles to detect and exploit non-local
    correlations between financial instruments across markets and timeframes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Quantum Entanglement Market Correlator."""
        super().__init__(config)
        self.module_name = "QuantumEntanglementMarketCorrelator"
        self.module_category = "quantum_consciousness"
        
        self.entanglement_threshold = self.config.get("entanglement_threshold", 0.85)
        self.coherence_window = self.config.get("coherence_window", 50)
        self.decoherence_factor = self.config.get("decoherence_factor", 0.05)
        self.superposition_states = self.config.get("superposition_states", 8)
        self.measurement_interval = self.config.get("measurement_interval", 5)
        
        self.entangled_pairs = []
        self.quantum_states = {}
        self.wave_function = None
        self.last_collapse_time = None
        
    def initialize(self) -> bool:
        """Initialize the Quantum Entanglement Market Correlator."""
        try:
            self._initialize_quantum_states()
            
            self._initialize_entanglement_detection()
            
            self._initialize_wave_function()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Quantum Entanglement Market Correlator: {e}")
            return False
            
    def _initialize_quantum_states(self) -> None:
        """Initialize the quantum state system."""
        self.quantum_params = {
            "basis_states": ["up", "down", "neutral"],
            "superposition_depth": 3,
            "entanglement_types": ["price", "volume", "momentum", "volatility"],
            "bell_states": ["Φ+", "Φ-", "Ψ+", "Ψ-"],
            "coherence_decay_rate": 0.02,
            "measurement_basis": ["price", "time", "momentum"],
        }
        
        self.quantum_register = {
            "size": self.superposition_states,
            "states": [0.0] * self.superposition_states,
            "amplitudes": [complex(0, 0)] * self.superposition_states,
            "phase": [0.0] * self.superposition_states,
            "entanglement_matrix": np.zeros((self.superposition_states, self.superposition_states)),
            "coherence_matrix": np.eye(self.superposition_states),
        }
        
    def _initialize_entanglement_detection(self) -> None:
        """Initialize the entanglement detection system."""
        self.entanglement_params = {
            "correlation_threshold": self.entanglement_threshold,
            "min_entanglement_duration": 10,
            "max_entanglement_distance": 100,
            "bell_inequality_threshold": 2.0,
            "quantum_discord_threshold": 0.3,
            "entanglement_entropy_threshold": 0.5,
        }
        
    def _initialize_wave_function(self) -> None:
        """Initialize the wave function."""
        self.wave_function_params = {
            "initial_state": "superposition",
            "normalization_factor": 1.0 / math.sqrt(self.superposition_states),
            "phase_coherence": 0.9,
            "amplitude_distribution": "uniform",
            "collapse_probability": 0.1,
            "measurement_effect": "partial",
        }
        
        self.wave_function = {
            "amplitudes": [complex(1.0 / math.sqrt(self.superposition_states), 0) 
                          for _ in range(self.superposition_states)],
            "phases": [0.0] * self.superposition_states,
            "probabilities": [1.0 / self.superposition_states] * self.superposition_states,
            "entanglement": np.zeros((self.superposition_states, self.superposition_states)),
            "last_update": datetime.now(),
        }
        
        self.last_collapse_time = datetime.now()
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using quantum entanglement principles."""
        if not self.initialized:
            return {"error": "Module not initialized"}
            
        try:
            assets_data = market_data.get("assets_data", {})
            timestamps = market_data.get("timestamps", [])
            
            if not assets_data or len(assets_data) < 2:
                return {"error": "Insufficient multi-asset data for entanglement analysis"}
                
            preprocessed_data = self._preprocess_data(assets_data, timestamps)
            
            quantum_states = self._update_quantum_states(preprocessed_data)
            
            entangled_pairs = self._detect_entangled_pairs(preprocessed_data, quantum_states)
            
            wave_function = self._update_wave_function(quantum_states, entangled_pairs)
            
            quantum_metrics = self._calculate_quantum_metrics(quantum_states, entangled_pairs, wave_function)
            
            analysis_results = {
                "quantum_states": quantum_states,
                "entangled_pairs": entangled_pairs,
                "wave_function": wave_function,
                "quantum_metrics": quantum_metrics,
                "timestamp": datetime.now()
            }
            
            self.last_analysis = analysis_results
            return analysis_results
            
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
            
    def _preprocess_data(self, assets_data: Dict[str, Dict[str, List[float]]], 
                        timestamps: List[datetime]) -> Dict[str, Any]:
        """Preprocess multi-asset market data for quantum analysis."""
        preprocessed_data = {
            "assets": {},
            "correlations": {},
            "phase_differences": {},
            "quantum_numbers": {},
        }
        
        for asset_id, asset_data in assets_data.items():
            prices = asset_data.get("prices", [])
            volumes = asset_data.get("volumes", [])
            
            if not prices or len(prices) < self.coherence_window:
                continue
                
            returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            momentum = [0.0, 0.0] + [(returns[i] - returns[i-1]) for i in range(2, len(returns))]
            
            window = min(20, len(returns) - 1)
            volatility = [0.0] * window
            for i in range(window, len(returns)):
                volatility.append(float(np.std(returns[i-window:i])))
                
            analytic_signal = self._hilbert_transform(returns)
            phase = np.unwrap(np.angle(analytic_signal))
            
            amplitude = np.abs(analytic_signal)
            
            preprocessed_data["assets"][asset_id] = {
                "prices": prices,
                "volumes": volumes,
                "returns": returns,
                "momentum": momentum,
                "volatility": volatility,
                "phase": phase.tolist(),
                "amplitude": amplitude.tolist(),
            }
            
            preprocessed_data["quantum_numbers"][asset_id] = self._calculate_quantum_numbers(
                returns, momentum, volatility
            )
            
        asset_ids = list(preprocessed_data["assets"].keys())
        for i, asset_id1 in enumerate(asset_ids):
            for j, asset_id2 in enumerate(asset_ids[i+1:], i+1):
                corr_key = f"{asset_id1}_{asset_id2}"
                returns1 = preprocessed_data["assets"][asset_id1]["returns"]
                returns2 = preprocessed_data["assets"][asset_id2]["returns"]
                
                min_len = min(len(returns1), len(returns2))
                if min_len > 10:
                    correlation = float(np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1])
                    
                    phase1 = preprocessed_data["assets"][asset_id1]["phase"][-min_len:]
                    phase2 = preprocessed_data["assets"][asset_id2]["phase"][-min_len:]
                    phase_diff = [phase1[i] - phase2[i] for i in range(min_len)]
                    phase_coherence = self._calculate_phase_coherence(phase_diff)
                    
                    preprocessed_data["correlations"][corr_key] = correlation
                    preprocessed_data["phase_differences"][corr_key] = {
                        "mean_diff": float(np.mean(phase_diff)),
                        "coherence": phase_coherence,
                    }
        
        return preprocessed_data
        
    def _hilbert_transform(self, signal: List[float]) -> np.ndarray:
        """Apply Hilbert transform to get analytic signal."""
        np_signal = np.array(signal)
        
        from scipy import signal as sp_signal
        analytic_signal = sp_signal.hilbert(np_signal)
        
        return np.array(analytic_signal)
        
    def _calculate_phase_coherence(self, phase_diff: List[float]) -> float:
        """Calculate phase coherence from phase differences."""
        np_phase_diff = np.array(phase_diff)
        
        mean_cos = np.mean(np.cos(np_phase_diff))
        mean_sin = np.mean(np.sin(np_phase_diff))
        
        coherence = np.sqrt(mean_cos**2 + mean_sin**2)
        
        return float(coherence)
        
    def _calculate_quantum_numbers(self, returns: List[float], 
                                 momentum: List[float], 
                                 volatility: List[float]) -> Dict[str, Any]:
        """Calculate quantum numbers for an asset."""
        mean_vol = float(np.mean(volatility[-20:]))
        if mean_vol < 0.005:
            n = 1  # Low volatility
        elif mean_vol < 0.01:
            n = 2  # Medium volatility
        else:
            n = 3  # High volatility
            
        mean_mom = float(np.mean(momentum[-20:]))
        std_mom = float(np.std(momentum[-20:]))
        if abs(mean_mom) < 0.5 * std_mom:
            l = 0  # No clear momentum
        elif abs(mean_mom) < std_mom:
            l = 1  # Moderate momentum
        else:
            l = 2  # Strong momentum
            
        if mean_mom > 0:
            m = l  # Positive momentum
        elif mean_mom < 0:
            m = -l  # Negative momentum
        else:
            m = 0  # Neutral
            
        recent_returns = returns[-5:]
        if sum(1 for r in recent_returns if r > 0) >= 3:
            s = 0.5  # Mostly positive returns
        else:
            s = -0.5  # Mostly negative returns
            
        return {
            "n": n,
            "l": l,
            "m": m,
            "s": s
        }
        
    def _update_quantum_states(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update quantum states based on preprocessed data."""
        quantum_states = {
            "asset_states": {},
            "superposition": {},
            "entanglement_matrix": np.zeros((len(preprocessed_data["assets"]), len(preprocessed_data["assets"]))),
        }
        
        for i, (asset_id, asset_data) in enumerate(preprocessed_data["assets"].items()):
            quantum_numbers = preprocessed_data["quantum_numbers"][asset_id]
            
            returns = asset_data["returns"]
            momentum = asset_data["momentum"]
            volatility = asset_data["volatility"]
            
            recent_returns = returns[-10:]
            up_count = sum(1 for r in recent_returns if r > 0.001)
            down_count = sum(1 for r in recent_returns if r < -0.001)
            neutral_count = 10 - up_count - down_count
            
            p_up = up_count / 10
            p_down = down_count / 10
            p_neutral = neutral_count / 10
            
            phase = asset_data["phase"][-1] if asset_data["phase"] else 0.0
            amp_up = complex(math.sqrt(p_up) * math.cos(phase), math.sqrt(p_up) * math.sin(phase))
            amp_down = complex(math.sqrt(p_down) * math.cos(phase + math.pi), math.sqrt(p_down) * math.sin(phase + math.pi))
            amp_neutral = complex(math.sqrt(p_neutral) * math.cos(phase + math.pi/2), math.sqrt(p_neutral) * math.sin(phase + math.pi/2))
            
            quantum_states["asset_states"][asset_id] = {
                "quantum_numbers": quantum_numbers,
                "probabilities": {
                    "up": p_up,
                    "down": p_down,
                    "neutral": p_neutral
                },
                "amplitudes": {
                    "up": amp_up,
                    "down": amp_down,
                    "neutral": amp_neutral
                },
                "phase": phase,
                "energy_level": quantum_numbers["n"],
                "momentum_state": quantum_numbers["l"],
                "direction": quantum_numbers["m"],
                "spin": quantum_numbers["s"]
            }
            
            quantum_states["entanglement_matrix"][i, i] = 1.0
            
        asset_ids = list(preprocessed_data["assets"].keys())
        for i, asset_id1 in enumerate(asset_ids):
            for j, asset_id2 in enumerate(asset_ids[i+1:], i+1):
                corr_key = f"{asset_id1}_{asset_id2}"
                if corr_key in preprocessed_data["correlations"]:
                    correlation = preprocessed_data["correlations"][corr_key]
                    phase_coherence = preprocessed_data["phase_differences"][corr_key]["coherence"]
                    
                    entanglement = abs(correlation) * phase_coherence
                    
                    quantum_states["entanglement_matrix"][i, j] = entanglement
                    quantum_states["entanglement_matrix"][j, i] = entanglement
        
        quantum_states["superposition"] = self._calculate_superposition_state(
            quantum_states["asset_states"], quantum_states["entanglement_matrix"]
        )
        
        return quantum_states
        
    def _calculate_superposition_state(self, asset_states: Dict[str, Dict[str, Any]], 
                                     entanglement_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate the superposition state of all assets."""
        superposition = {
            "amplitudes": [complex(0, 0)] * self.superposition_states,
            "probabilities": [0.0] * self.superposition_states,
            "basis_states": [],
            "entanglement_entropy": 0.0,
        }
        
        asset_ids = list(asset_states.keys())
        n_assets = len(asset_ids)
        
        n_basis = min(self.superposition_states, 2**n_assets)
        
        import itertools
        basis_states = []
        for i in range(n_basis):
            binary = format(i, f'0{n_assets}b')
            basis_state = {}
            for j, asset_id in enumerate(asset_ids):
                if binary[j] == '0':
                    basis_state[asset_id] = "up"
                else:
                    basis_state[asset_id] = "down"
            basis_states.append(basis_state)
            
        superposition["basis_states"] = basis_states
        
        for i, basis_state in enumerate(basis_states):
            amplitude = complex(1, 0)
            for asset_id, state in basis_state.items():
                amplitude *= asset_states[asset_id]["amplitudes"][state]
                
            entanglement_factor = 1.0
            for j, asset_id1 in enumerate(asset_ids):
                for k, asset_id2 in enumerate(asset_ids[j+1:], j+1):
                    if basis_state[asset_id1] == basis_state[asset_id2]:
                        entanglement_factor *= (1.0 + entanglement_matrix[j, k])
                    else:
                        entanglement_factor *= (1.0 - entanglement_matrix[j, k])
                        
            amplitude *= entanglement_factor
            
            superposition["amplitudes"][i] = amplitude
            
        norm_factor = math.sqrt(sum(abs(a)**2 for a in superposition["amplitudes"]))
        if norm_factor > 0:
            superposition["amplitudes"] = [a / norm_factor for a in superposition["amplitudes"]]
            
        superposition["probabilities"] = [abs(a)**2 for a in superposition["amplitudes"]]
        
        entropy = -sum(p * math.log(p) for p in superposition["probabilities"] if p > 0)
        superposition["entanglement_entropy"] = entropy
        
        return superposition
        
    def _detect_entangled_pairs(self, preprocessed_data: Dict[str, Any], 
                              quantum_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect entangled pairs of assets."""
        entangled_pairs = []
        
        asset_ids = list(preprocessed_data["assets"].keys())
        
        for i, asset_id1 in enumerate(asset_ids):
            for j, asset_id2 in enumerate(asset_ids[i+1:], i+1):
                corr_key = f"{asset_id1}_{asset_id2}"
                
                if corr_key in preprocessed_data["correlations"]:
                    correlation = preprocessed_data["correlations"][corr_key]
                    phase_coherence = preprocessed_data["phase_differences"][corr_key]["coherence"]
                    
                    if (abs(correlation) >= self.entanglement_params["correlation_threshold"] and
                        phase_coherence >= 0.7):
                        
                        bell_value = self._calculate_bell_inequality(
                            quantum_states["asset_states"][asset_id1],
                            quantum_states["asset_states"][asset_id2],
                            correlation,
                            phase_coherence
                        )
                        
                        if bell_value > self.entanglement_params["bell_inequality_threshold"]:
                            if correlation > 0:
                                entanglement_type = "Φ+"  # Aligned
                            else:
                                entanglement_type = "Φ-"  # Anti-aligned
                                
                            entanglement_strength = abs(correlation) * phase_coherence
                            
                            entangled_pair = {
                                "asset1": asset_id1,
                                "asset2": asset_id2,
                                "correlation": correlation,
                                "phase_coherence": phase_coherence,
                                "bell_value": bell_value,
                                "entanglement_type": entanglement_type,
                                "entanglement_strength": entanglement_strength
                            }
                            
                            entangled_pairs.append(entangled_pair)
        
        entangled_pairs.sort(key=lambda p: p["entanglement_strength"], reverse=True)
        
        return entangled_pairs
        
    def _calculate_bell_inequality(self, state1: Dict[str, Any], state2: Dict[str, Any],
                                correlation: float, phase_coherence: float) -> float:
        """Calculate Bell's inequality value for a pair of quantum states."""
        
        a = 0
        a_prime = math.pi / 2
        b = math.pi / 4
        b_prime = 3 * math.pi / 4
        
        E_ab = correlation * math.cos(a - b)
        E_ab_prime = correlation * math.cos(a - b_prime)
        E_a_prime_b = correlation * math.cos(a_prime - b)
        E_a_prime_b_prime = correlation * math.cos(a_prime - b_prime)
        
        S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)
        
        S *= phase_coherence
        
        return S
        
    def _update_wave_function(self, quantum_states: Dict[str, Any], 
                            entangled_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update the wave function based on quantum states and entangled pairs."""
        if self.wave_function is None:
            self._initialize_wave_function()
            
        if self.wave_function is None:
            current_wave_function = {
                "amplitudes": [complex(1.0 / math.sqrt(self.superposition_states), 0) 
                              for _ in range(self.superposition_states)],
                "phases": [0.0] * self.superposition_states,
                "probabilities": [1.0 / self.superposition_states] * self.superposition_states,
                "entanglement": np.zeros((self.superposition_states, self.superposition_states)),
                "last_update": datetime.now(),
            }
        else:
            current_wave_function = self.wave_function.copy()
        
        superposition = quantum_states["superposition"]
        
        for i in range(min(len(superposition["amplitudes"]), len(current_wave_function["amplitudes"]))):
            time_diff = (datetime.now() - current_wave_function["last_update"]).total_seconds()
            decoherence_factor = math.exp(-self.decoherence_factor * time_diff)
            
            old_amp = current_wave_function["amplitudes"][i]
            new_amp = superposition["amplitudes"][i] if i < len(superposition["amplitudes"]) else complex(0, 0)
            
            mixed_amp = old_amp * decoherence_factor + new_amp * (1 - decoherence_factor)
            
            current_wave_function["amplitudes"][i] = mixed_amp
            
        norm_factor = math.sqrt(sum(abs(a)**2 for a in current_wave_function["amplitudes"]))
        if norm_factor > 0:
            current_wave_function["amplitudes"] = [a / norm_factor for a in current_wave_function["amplitudes"]]
            
        current_wave_function["probabilities"] = [abs(a)**2 for a in current_wave_function["amplitudes"]]
        
        if entangled_pairs:
            n = len(current_wave_function["entanglement"])
            new_entanglement = np.zeros((n, n))
            
            for i in range(n):
                new_entanglement[i, i] = 1.0
                
            for pair in entangled_pairs:
                asset_ids = list(quantum_states["asset_states"].keys())
                if pair["asset1"] in asset_ids and pair["asset2"] in asset_ids:
                    i = asset_ids.index(pair["asset1"])
                    j = asset_ids.index(pair["asset2"])
                    
                    if i < n and j < n:
                        new_entanglement[i, j] = pair["entanglement_strength"]
                        new_entanglement[j, i] = pair["entanglement_strength"]
            
            time_diff = (datetime.now() - current_wave_function["last_update"]).total_seconds()
            decoherence_factor = math.exp(-self.decoherence_factor * time_diff)
            
            current_wave_function["entanglement"] = (
                current_wave_function["entanglement"] * decoherence_factor +
                new_entanglement * (1 - decoherence_factor)
            )
            
        current_wave_function["last_update"] = datetime.now()
        
        if self.last_collapse_time is None:
            self.last_collapse_time = datetime.now()
            time_since_collapse = 0
        else:
            time_since_collapse = (datetime.now() - self.last_collapse_time).total_seconds()
            
        if time_since_collapse > self.measurement_interval:
            current_wave_function = self._collapse_wave_function(current_wave_function)
            self.last_collapse_time = datetime.now()
            
        return current_wave_function
        
    def _collapse_wave_function(self, wave_function: Dict[str, Any]) -> Dict[str, Any]:
        """Collapse the wave function to a definite state."""
        probabilities = wave_function["probabilities"]
        
        import random
        r = random.random()
        cumulative_prob = 0.0
        collapsed_state = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                collapsed_state = i
                break
                
        collapsed_wave_function = wave_function.copy()
        collapsed_wave_function["amplitudes"] = [complex(0, 0)] * len(probabilities)
        collapsed_wave_function["amplitudes"][collapsed_state] = complex(1, 0)
        collapsed_wave_function["probabilities"] = [0.0] * len(probabilities)
        collapsed_wave_function["probabilities"][collapsed_state] = 1.0
        collapsed_wave_function["collapsed_state"] = collapsed_state
        collapsed_wave_function["collapse_time"] = datetime.now()
        
        return collapsed_wave_function
        
    def _calculate_quantum_metrics(self, quantum_states: Dict[str, Any],
                                 entangled_pairs: List[Dict[str, Any]],
                                 wave_function: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantum metrics based on quantum states and wave function."""
        quantum_metrics = {}
        
        quantum_metrics["entanglement_count"] = len(entangled_pairs)
        quantum_metrics["avg_entanglement_strength"] = (
            sum(pair["entanglement_strength"] for pair in entangled_pairs) / len(entangled_pairs)
            if entangled_pairs else 0.0
        )
        quantum_metrics["max_entanglement_strength"] = (
            max(pair["entanglement_strength"] for pair in entangled_pairs)
            if entangled_pairs else 0.0
        )
        
        quantum_metrics["wave_function_entropy"] = -sum(
            p * math.log(p) for p in wave_function["probabilities"] if p > 0
        )
        quantum_metrics["wave_function_purity"] = sum(
            abs(a)**4 for a in wave_function["amplitudes"]
        )
        
        quantum_metrics["superposition_entropy"] = quantum_states["superposition"]["entanglement_entropy"]
        quantum_metrics["effective_dimension"] = math.exp(quantum_metrics["superposition_entropy"])
        
        quantum_metrics["quantum_prediction"] = self._calculate_quantum_prediction(
            quantum_states, entangled_pairs, wave_function
        )
        
        return quantum_metrics
        
    def _calculate_quantum_prediction(self, quantum_states: Dict[str, Any],
                                    entangled_pairs: List[Dict[str, Any]],
                                    wave_function: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantum prediction based on quantum states and wave function."""
        prediction = {
            "market_direction": "NEUTRAL",
            "confidence": 0.5,
            "entanglement_signals": [],
            "quantum_probabilities": {},
        }
        
        if "collapsed_state" in wave_function:
            collapsed_state = wave_function["collapsed_state"]
            if collapsed_state < len(quantum_states["superposition"]["basis_states"]):
                basis_state = quantum_states["superposition"]["basis_states"][collapsed_state]
                
                up_count = sum(1 for state in basis_state.values() if state == "up")
                down_count = sum(1 for state in basis_state.values() if state == "down")
                
                if up_count > down_count:
                    prediction["market_direction"] = "BUY"
                    prediction["confidence"] = 0.5 + 0.5 * (up_count - down_count) / len(basis_state)
                elif down_count > up_count:
                    prediction["market_direction"] = "SELL"
                    prediction["confidence"] = 0.5 + 0.5 * (down_count - up_count) / len(basis_state)
        else:
            up_prob = 0.0
            down_prob = 0.0
            
            for i, prob in enumerate(wave_function["probabilities"]):
                if i < len(quantum_states["superposition"]["basis_states"]):
                    basis_state = quantum_states["superposition"]["basis_states"][i]
                    
                    up_count = sum(1 for state in basis_state.values() if state == "up")
                    down_count = sum(1 for state in basis_state.values() if state == "down")
                    
                    if up_count > down_count:
                        up_prob += prob
                    elif down_count > up_count:
                        down_prob += prob
                        
            if up_prob > down_prob:
                prediction["market_direction"] = "BUY"
                prediction["confidence"] = 0.5 + 0.5 * (up_prob - down_prob)
            elif down_prob > up_prob:
                prediction["market_direction"] = "SELL"
                prediction["confidence"] = 0.5 + 0.5 * (down_prob - up_prob)
                
        for pair in entangled_pairs:
            if pair["entanglement_strength"] > self.entanglement_threshold:
                signal = {
                    "asset1": pair["asset1"],
                    "asset2": pair["asset2"],
                    "entanglement_type": pair["entanglement_type"],
                    "entanglement_strength": pair["entanglement_strength"],
                    "signal": "ALIGNED" if pair["correlation"] > 0 else "ANTI_ALIGNED",
                }
                
                prediction["entanglement_signals"].append(signal)
                
        for i, prob in enumerate(wave_function["probabilities"]):
            if i < len(quantum_states["superposition"]["basis_states"]):
                basis_state = quantum_states["superposition"]["basis_states"][i]
                state_key = "_".join(f"{asset_id}:{state}" for asset_id, state in basis_state.items())
                prediction["quantum_probabilities"][state_key] = prob
                
        return prediction
        
    def get_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading signal based on market data."""
        if not self.initialized:
            return {"error": "Module not initialized"}
            
        try:
            analysis = self.analyze(market_data)
            
            if "error" in analysis:
                return {"error": analysis["error"]}
                
            quantum_metrics = analysis.get("quantum_metrics", {})
            quantum_prediction = quantum_metrics.get("quantum_prediction", {})
            entangled_pairs = analysis.get("entangled_pairs", [])
            
            direction = quantum_prediction.get("market_direction", "NEUTRAL")
            confidence = quantum_prediction.get("confidence", 0.5)
            
            if entangled_pairs:
                max_entanglement = max(pair["entanglement_strength"] for pair in entangled_pairs)
                confidence = min(confidence + 0.2 * max_entanglement, 0.95)
                
            signal = {
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now(),
                "quantum_metrics": quantum_metrics,
                "entangled_pairs": entangled_pairs[:3]  # Include top 3 entangled pairs
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
                
            current_metrics = current_analysis.get("quantum_metrics", {})
            current_prediction = current_metrics.get("quantum_prediction", {})
            
            is_valid = True
            validation_confidence = confidence
            
            if current_prediction.get("market_direction", "NEUTRAL") != direction:
                is_valid = False
                validation_confidence *= 0.5
                
            current_entangled = current_analysis.get("entangled_pairs", [])
            original_entangled = signal.get("entangled_pairs", [])
            
            if len(current_entangled) > 0 and len(original_entangled) > 0:
                top_current = set(pair["asset1"] + "_" + pair["asset2"] for pair in current_entangled[:3])
                top_original = set(pair["asset1"] + "_" + pair["asset2"] for pair in original_entangled[:3])
                
                if len(top_current.intersection(top_original)) < 2:
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
