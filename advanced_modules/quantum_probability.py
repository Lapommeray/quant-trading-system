#!/usr/bin/env python3
"""
Quantum Probability Module

Implements quantum probability theory and non-ergodic economics for financial markets:
- Quantum measurement theory for financial markets
- Non-ergodic economics models challenging efficient market hypothesis
- Quantum entanglement effects in portfolio correlations
- Quantum superposition for modeling market uncertainty
- Quantum decision theory for optimal trading strategies

This module provides a rigorous mathematical foundation for quantum finance
beyond classical probability theory.
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumProbability")

class QuantumProbability:
    """
    Quantum Probability for financial markets
    
    Implements quantum probability theory and non-ergodic economics:
    - Quantum measurement theory
    - Non-ergodic economics
    - Quantum entanglement
    - Quantum superposition
    - Quantum decision theory
    
    Provides a rigorous mathematical foundation for quantum finance
    beyond classical probability theory.
    """
    
    def __init__(self, precision: int = 64, confidence_level: float = 0.99,
                entanglement_threshold: float = 0.7, non_ergodicity_factor: float = 0.5,
                hilbert_space_dim: int = 4):
        """
        Initialize Quantum Probability
        
        Parameters:
        - precision: Numerical precision for calculations (default: 64 bits)
        - confidence_level: Statistical confidence level (default: 0.99)
        - entanglement_threshold: Threshold for quantum entanglement (default: 0.7)
        - non_ergodicity_factor: Factor for non-ergodic effects (default: 0.5)
        - hilbert_space_dim: Dimension of Hilbert space (default: 4)
        """
        self.precision = precision
        self.confidence_level = confidence_level
        self.entanglement_threshold = entanglement_threshold
        self.non_ergodicity_factor = non_ergodicity_factor
        self.hilbert_space_dim = hilbert_space_dim
        self.history = []
        
        np.random.seed(42)  # For reproducibility
        
        logger.info(f"Initialized QuantumProbability with precision={precision}, "
                   f"confidence_level={confidence_level}, "
                   f"hilbert_space_dim={hilbert_space_dim}")
    
    
    def create_density_matrix(self, returns: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Create density matrix from returns data
        
        Parameters:
        - returns: Array of asset returns
        - normalize: Whether to normalize the density matrix (default: True)
        
        Returns:
        - Density matrix representing quantum state
        """
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        if returns.size == 0 or returns.shape[0] <= 1 or returns.shape[1] == 0:
            logger.warning("Insufficient data for density matrix creation, using identity")
            dim = max(1, returns.shape[1] if len(returns.shape) > 1 else 1)
            return np.eye(dim)
            
        if normalize:
            std_dev = np.std(returns, axis=0) + 1e-8
            returns = (returns - np.mean(returns, axis=0)) / std_dev
        
        try:
            cov_matrix = np.cov(returns, rowvar=False)
            
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[float(cov_matrix)]])
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative eigenvalues
            
            density_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            if normalize:
                trace = np.trace(density_matrix)
                if trace > 0:
                    density_matrix = density_matrix / trace
                else:
                    density_matrix = np.eye(density_matrix.shape[0]) / density_matrix.shape[0]
        except Exception as e:
            logger.warning(f"Error creating density matrix: {e}, using identity")
            dim = returns.shape[1]
            density_matrix = np.eye(dim)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_density_matrix',
            'returns_shape': returns.shape,
            'density_matrix_shape': density_matrix.shape,
            'eigenvalues_min': float(min(eigenvalues)) if 'eigenvalues' in locals() else 0.0,
            'eigenvalues_max': float(max(eigenvalues)) if 'eigenvalues' in locals() else 1.0
        })
        
        return density_matrix
    
    def quantum_measurement(self, density_matrix: np.ndarray, 
                           observable: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Perform quantum measurement on density matrix
        
        Parameters:
        - density_matrix: Quantum state as density matrix
        - observable: Hermitian operator representing observable
        
        Returns:
        - Tuple of (expected value, post-measurement density matrix)
        """
        if not np.allclose(observable, observable.conj().T):
            logger.warning("Observable is not Hermitian, symmetrizing")
            observable = 0.5 * (observable + observable.conj().T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        expected_value = np.trace(density_matrix @ observable)
        
        post_density_matrix = np.zeros_like(density_matrix)
        
        for i, eigenvector in enumerate(eigenvectors.T):
            projector = np.outer(eigenvector, eigenvector.conj())
            
            probability = np.real(np.trace(projector @ density_matrix))
            
            if probability > 1e-10:  # Avoid numerical issues
                post_density_matrix += probability * projector
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'quantum_measurement',
            'density_matrix_shape': density_matrix.shape,
            'observable_shape': observable.shape,
            'expected_value': float(np.real(expected_value))
        })
        
        return float(np.real(expected_value)), post_density_matrix
    
    def create_market_observable(self, assets: int, observable_type: str = 'momentum') -> np.ndarray:
        """
        Create market observable operator
        
        Parameters:
        - assets: Number of assets
        - observable_type: Type of observable ('momentum', 'volatility', 'correlation')
        
        Returns:
        - Hermitian operator representing market observable
        """
        if observable_type == 'momentum':
            observable = np.zeros((assets, assets))
            for i in range(assets):
                observable[i, i] = i - assets/2  # Centered values
        elif observable_type == 'volatility':
            observable = np.eye(assets)
            for i in range(assets-1):
                observable[i, i+1] = observable[i+1, i] = -0.5
        elif observable_type == 'correlation':
            observable = np.zeros((assets, assets))
            for i in range(assets):
                for j in range(assets):
                    if i != j:
                        observable[i, j] = 1.0 / (1.0 + abs(i - j))
        else:
            logger.warning(f"Unknown observable type: {observable_type}, using identity")
            observable = np.eye(assets)
        
        observable = 0.5 * (observable + observable.conj().T)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_market_observable',
            'assets': assets,
            'observable_type': observable_type,
            'observable_shape': observable.shape
        })
        
        return observable
    
    
    def time_average_vs_ensemble_average(self, returns: np.ndarray, 
                                        window_size: int = 20) -> Dict:
        """
        Compare time averages vs. ensemble averages to detect non-ergodicity
        
        Parameters:
        - returns: Array of asset returns with shape (time_steps, assets)
        - window_size: Window size for time averaging (default: 20)
        
        Returns:
        - Dictionary with non-ergodicity metrics
        """
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            time_steps, assets = returns.shape
        else:
            time_steps, assets = returns.shape
        
        # Calculate ensemble statistics
        ensemble_means = np.mean(returns, axis=1)
        ensemble_vars = np.var(returns, axis=1, ddof=1)
        
        time_means = np.zeros((time_steps - window_size + 1, assets))
        time_vars = np.zeros((time_steps - window_size + 1, assets))
        
        for t in range(time_steps - window_size + 1):
            time_means[t] = np.mean(returns[t:t+window_size], axis=0)
            time_vars[t] = np.var(returns[t:t+window_size], axis=0)
        
        mean_diff = np.mean(np.abs(np.mean(time_means, axis=0) - np.mean(ensemble_means)))
        
        var_ratio = np.mean(np.mean(time_vars, axis=0) / np.mean(ensemble_vars))
        
        eb_parameter = np.abs(var_ratio - 1.0)
        
        non_ergodicity_index = np.tanh(eb_parameter)
        
        result = {
            'mean_difference': float(mean_diff),
            'variance_ratio': float(var_ratio),
            'ergodicity_breaking': float(eb_parameter),
            'non_ergodicity_index': float(non_ergodicity_index),
            'is_ergodic': bool(non_ergodicity_index < self.non_ergodicity_factor)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'time_average_vs_ensemble_average',
            'returns_shape': returns.shape,
            'window_size': window_size,
            'result': result
        })
        
        return result
    
    def calculate_non_ergodic_kelly(self, returns: np.ndarray, quantum_state: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate non-ergodic Kelly criterion using quantum state information
        
        Parameters:
        - returns: Array of asset returns
        - quantum_state: Quantum state representation of the market (optional)
        
        Returns:
        - Dictionary with optimal fraction and confidence level
        """
        if returns is None or len(returns) < 2:
            logger.warning("Insufficient data for non-ergodic Kelly calculation")
            return {
                "optimal_fraction": 0.5,  # Default to 50% allocation
                "confidence": self.confidence_level,
                "is_ergodic": False,
                "non_ergodicity_index": 0.5
            }
            
        risk_free_rate = 0.0
        time_horizon = 252
        
        if quantum_state is not None:
            try:
                if isinstance(quantum_state, dict):
                    quantum_amplitude = quantum_state.get('amplitude', 0.5)
                elif isinstance(quantum_state, (list, np.ndarray)) and len(quantum_state) > 0:
                    quantum_amplitude = np.abs(quantum_state[0])**2
                else:
                    quantum_amplitude = 0.5
                    logger.warning("Unknown quantum state format, using default amplitude")
            except (IndexError, TypeError, KeyError):
                logger.warning("Error processing quantum state, using default amplitude")
                quantum_amplitude = 0.5
                
            risk_free_rate = 0.01 * quantum_amplitude  # Adjust risk-free rate based on quantum state
            
        kelly_result = self.kelly_criterion_non_ergodic(returns, risk_free_rate, time_horizon)
        
        result = {
            "optimal_fraction": float(np.mean(kelly_result["kelly_weights"])) if len(kelly_result["kelly_weights"]) > 0 else 0.5,
            "confidence": self.confidence_level,
            "is_ergodic": kelly_result["is_ergodic"],
            "non_ergodicity_index": kelly_result["non_ergodicity_index"]
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'calculate_non_ergodic_kelly',
            'returns_shape': returns.shape if hasattr(returns, 'shape') else None,
            'quantum_state_shape': quantum_state.shape if quantum_state is not None and hasattr(quantum_state, 'shape') else None,
            'result': result
        })
        
        return result
        
    def kelly_criterion_non_ergodic(self, returns: np.ndarray, risk_free_rate: float = 0.0,
                                  time_horizon: int = 252) -> Dict:
        """
        Calculate Kelly criterion under non-ergodic conditions
        
        Parameters:
        - returns: Array of asset returns
        - risk_free_rate: Risk-free rate (default: 0.0)
        - time_horizon: Investment time horizon in days (default: 252)
        
        Returns:
        - Dictionary with Kelly allocations and non-ergodic adjustments
        """
        if len(returns.shape) == 1 or returns.shape[1] == 1:
            if len(returns.shape) == 1:
                returns_1d = returns
            else:
                returns_1d = returns.flatten()
                
            mean_return = np.mean(returns_1d)
            variance = np.var(returns_1d, ddof=1) if len(returns_1d) > 1 else 0.01
            
            if variance < 1e-8:
                variance = 0.01  # Default small variance
                
            excess_return = mean_return - risk_free_rate
            kelly_fraction = excess_return / variance if variance > 0 else 0.5
            
            kelly_fraction = float(max(0.0, min(1.0, float(kelly_fraction))))
            
            non_ergodic_metrics = self.time_average_vs_ensemble_average(returns)
            is_ergodic = non_ergodic_metrics['is_ergodic']
            
            if not is_ergodic:
                non_ergodic_factor = non_ergodic_metrics['non_ergodicity_index']
                kelly_fraction *= (1.0 - 0.5 * non_ergodic_factor)
            
            return {
                'kelly_weights': [float(kelly_fraction)],
                'is_ergodic': is_ergodic,
                'non_ergodicity_index': float(non_ergodic_metrics['non_ergodicity_index']),
                'expected_growth_rate': float(kelly_fraction * excess_return - 0.5 * kelly_fraction**2 * variance)
            }
        
        try:
            if len(returns.shape) == 1:
                returns_2d = returns.reshape(-1, 1)
            else:
                returns_2d = returns
                
            mean_returns = np.mean(returns_2d, axis=0)
            
            if len(returns_2d) <= 1:
                logger.warning("Insufficient data for covariance calculation, using identity matrix")
                cov_matrix = np.eye(returns_2d.shape[1]) * 0.01
            else:
                cov_matrix = np.cov(returns_2d, rowvar=False)
                
                if isinstance(cov_matrix, (int, float)):
                    cov_matrix = np.array([[float(cov_matrix)]], dtype=np.float64)
                elif hasattr(cov_matrix, 'ndim') and cov_matrix.ndim == 0:
                    try:
                        scalar_value = float(cov_matrix.item())
                        cov_matrix = np.array([[scalar_value]], dtype=np.float64)
                    except (AttributeError, ValueError):
                        logger.warning("Failed to convert scalar covariance matrix, using default value")
                        cov_matrix = np.array([[0.01]], dtype=np.float64)
                
                try:
                    min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                    if min_eig < 1e-8:
                        cov_matrix += np.eye(cov_matrix.shape[0]) * (1e-8 - min_eig if min_eig < 0 else 1e-8)
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(f"Error calculating eigenvalues: {e}. Using regularized matrix.")
                    cov_matrix = np.eye(cov_matrix.shape[0]) * 0.01
            
            non_ergodic_metrics = self.time_average_vs_ensemble_average(returns)
            is_ergodic = non_ergodic_metrics['is_ergodic']
            
            excess_returns = mean_returns - risk_free_rate
            
            try:
                cov_matrix_np = np.asarray(cov_matrix, dtype=np.float64)
                excess_returns_np = np.asarray(excess_returns, dtype=np.float64)
                
                kelly_weights_ergodic = np.dot(np.linalg.pinv(cov_matrix_np), excess_returns_np)
            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                logger.warning(f"Matrix singular, using pseudo-inverse for Kelly calculation: {e}")
                if hasattr(excess_returns, 'shape') and len(excess_returns.shape) > 0:
                    kelly_weights_ergodic = np.ones(excess_returns.shape[0]) / excess_returns.shape[0]
                else:
                    kelly_weights_ergodic = np.array([1.0])
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Error in Kelly calculation: {e}. Using default allocation.")
            kelly_weights_ergodic = np.ones(returns.shape[1]) / returns.shape[1]
            is_ergodic = True
            
        if not is_ergodic:
            growth_rates = np.log(1 + returns)
            time_avg_growth = np.mean(growth_rates, axis=0)
            
            non_ergodic_factor = non_ergodic_metrics['non_ergodicity_index']
            adjusted_returns = time_avg_growth * (1 - non_ergodic_factor) + mean_returns * non_ergodic_factor
            
            excess_returns_adjusted = adjusted_returns - risk_free_rate
            try:
                cov_matrix_np = np.asarray(cov_matrix, dtype=np.float64)
                excess_returns_np = np.asarray(excess_returns_adjusted, dtype=np.float64)
                
                kelly_weights_non_ergodic = np.dot(np.linalg.pinv(cov_matrix_np), excess_returns_np)
                scaling_factor = 1.0 - 0.5 * non_ergodic_factor
                kelly_weights = kelly_weights_non_ergodic * scaling_factor
            except (np.linalg.LinAlgError, ValueError, TypeError):
                logger.warning("Error in non-ergodic Kelly calculation. Using ergodic weights.")
                kelly_weights = kelly_weights_ergodic
        else:
            kelly_weights = kelly_weights_ergodic
        
        # Normalize weights if sum is positive
        if np.sum(kelly_weights) > 0:
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        
        result = {
            'kelly_weights': kelly_weights.tolist(),
            'is_ergodic': is_ergodic,
            'non_ergodicity_index': float(non_ergodic_metrics['non_ergodicity_index']),
            'expected_growth_rate': float(np.sum(kelly_weights * excess_returns) - 
                                        0.5 * np.dot(kelly_weights, np.dot(cov_matrix, kelly_weights)))
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'kelly_criterion_non_ergodic',
            'returns_shape': returns.shape,
            'risk_free_rate': risk_free_rate,
            'time_horizon': time_horizon,
            'result': result
        })
        
        return result
        
        if not is_ergodic:
            growth_rates = np.log(1 + returns)
            time_avg_growth = np.mean(growth_rates, axis=0)
            
            non_ergodic_factor = non_ergodic_metrics['non_ergodicity_index']
            adjusted_returns = time_avg_growth * (1 - non_ergodic_factor) + mean_returns * non_ergodic_factor
            
            excess_returns_adjusted = adjusted_returns - risk_free_rate
            kelly_weights_non_ergodic = np.linalg.solve(cov_matrix, excess_returns_adjusted)
            
            scaling_factor = 1.0 - 0.5 * non_ergodic_factor
            kelly_weights = kelly_weights_non_ergodic * scaling_factor
        else:
            kelly_weights = kelly_weights_ergodic
        
        if np.sum(kelly_weights) > 0:
            kelly_weights = kelly_weights / np.sum(kelly_weights)
        
        result = {
            'kelly_weights': kelly_weights.tolist(),
            'is_ergodic': is_ergodic,
            'non_ergodicity_index': float(non_ergodic_metrics['non_ergodicity_index']),
            'expected_growth_rate': float(np.sum(kelly_weights * excess_returns) - 
                                        0.5 * np.dot(kelly_weights, np.dot(cov_matrix, kelly_weights)))
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'kelly_criterion_non_ergodic',
            'returns_shape': returns.shape,
            'risk_free_rate': risk_free_rate,
            'time_horizon': time_horizon,
            'result': result
        })
        
        return result
    
    
    def quantum_correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate quantum correlation matrix from returns
        
        Parameters:
        - returns: Array of asset returns
        
        Returns:
        - Quantum correlation matrix
        """
        density_matrix = self.create_density_matrix(returns)
        
        assets = returns.shape[1]
        
        q_corr = np.zeros((assets, assets))
        
        for i in range(assets):
            for j in range(assets):
                if i == j:
                    q_corr[i, j] = 1.0
                else:
                    obs_i = np.zeros((assets, assets))
                    obs_i[i, i] = 1.0
                    
                    obs_j = np.zeros((assets, assets))
                    obs_j[j, j] = 1.0
                    
                    expected_ij, _ = self.quantum_measurement(density_matrix, obs_i @ obs_j)
                    expected_i, _ = self.quantum_measurement(density_matrix, obs_i)
                    expected_j, _ = self.quantum_measurement(density_matrix, obs_j)
                    
                    if expected_i > 0 and expected_j > 0:
                        q_corr[i, j] = expected_ij / np.sqrt(expected_i * expected_j)
                    else:
                        q_corr[i, j] = 0.0
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'quantum_correlation_matrix',
            'returns_shape': returns.shape,
            'q_corr_shape': q_corr.shape
        })
        
        return q_corr
    
    def detect_quantum_entanglement(self, returns: np.ndarray) -> Dict:
        """
        Detect quantum entanglement in asset returns
        
        Parameters:
        - returns: Array of asset returns
        
        Returns:
        - Dictionary with entanglement metrics
        """
        density_matrix = self.create_density_matrix(returns)
        
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        purity = np.trace(density_matrix @ density_matrix)
        
        q_corr = self.quantum_correlation_matrix(returns)
        
        entangled_pairs = []
        for i in range(q_corr.shape[0]):
            for j in range(i+1, q_corr.shape[1]):
                if abs(q_corr[i, j]) > self.entanglement_threshold:
                    entangled_pairs.append((i, j, float(q_corr[i, j])))
        
        entanglement_measure = 1.0 - purity
        
        result = {
            'von_neumann_entropy': float(entropy),
            'purity': float(purity),
            'entanglement_measure': float(entanglement_measure),
            'entangled_pairs': entangled_pairs,
            'is_entangled': bool(entanglement_measure > 0.1)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_quantum_entanglement',
            'returns_shape': returns.shape,
            'result': result
        })
        
        return result
    
    
    def create_market_quantum_state(self, returns: np.ndarray, n_qubits: int = 5) -> Dict:
        """
        Create quantum state representing market conditions
        
        Parameters:
        - returns: Array of asset returns
        - n_qubits: Number of qubits to use (default: 5)
        
        Returns:
        - Dictionary with quantum state information
        """
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        time_steps, assets = returns.shape
        
        normalized_returns = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + 1e-8)
        
        # Create density matrix
        density_matrix = self.create_density_matrix(normalized_returns)
        
        # Create quantum state vector
        pca = PCA(n_components=min(n_qubits, assets))
        principal_components = pca.fit_transform(normalized_returns)
        
        # Map to quantum amplitudes
        amplitudes = np.zeros(2**n_qubits, dtype=np.complex128)
        
        for i in range(min(time_steps, 2**n_qubits)):
            idx = 0
            for j in range(min(n_qubits, principal_components.shape[1])):
                if principal_components[i % time_steps, j] > 0:
                    idx |= (1 << j)
            
            amplitudes[idx] += 1.0
            
        # Normalize quantum state
        norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
        if norm > 0:
            amplitudes = amplitudes / norm
            
        # Calculate quantum properties
        entropy = -np.sum(np.abs(amplitudes)**2 * np.log2(np.abs(amplitudes)**2 + 1e-10))
        
        # Detect market regime from quantum state
        regime_idx = np.argmax(np.abs(amplitudes)**2)
        
        regimes = ["bullish", "bearish", "volatile", "stable", "trending", "reversal", "unknown"]
        regime = regimes[regime_idx % len(regimes)]
        
        confidence = np.abs(amplitudes[regime_idx])**2
        
        result = {
            "quantum_state": amplitudes.tolist(),
            "density_matrix": density_matrix.tolist(),
            "entropy": float(entropy),
            "n_qubits": n_qubits,
            "regime": regime,
            "confidence": float(confidence),
            "eigenvalues": np.linalg.eigvalsh(density_matrix).tolist()
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_market_quantum_state',
            'returns_shape': returns.shape,
            'n_qubits': n_qubits,
            'entropy': float(entropy),
            'regime': regime,
            'confidence': float(confidence)
        })
        
        return result
        
    def create_superposition_state(self, weights: np.ndarray) -> np.ndarray:
        """
        Create quantum superposition state from weights
        
        Parameters:
        - weights: Array of weights for different states
        
        Returns:
        - Quantum state vector
        """
        norm = np.sqrt(np.sum(np.abs(weights)**2))
        if norm > 0:
            normalized_weights = weights / norm
        else:
            normalized_weights = np.ones_like(weights) / np.sqrt(len(weights))
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_superposition_state',
            'weights_shape': weights.shape,
            'state_shape': normalized_weights.shape
        })
        
        return normalized_weights
    
    def market_regime_superposition(self, returns: np.ndarray, n_regimes: int = 3) -> Dict:
        """
        Model market regimes as quantum superposition states
        
        Parameters:
        - returns: Array of asset returns
        - n_regimes: Number of market regimes to identify
        
        Returns:
        - Dictionary with market regime superposition
        """
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(returns)
        
        regime_centroids = kmeans.cluster_centers_
        
        regime_counts = np.bincount(regime_labels, minlength=n_regimes)
        regime_probs = regime_counts / len(regime_labels)
        
        superposition_state = self.create_superposition_state(np.sqrt(regime_probs))
        
        density_matrix = np.outer(superposition_state, superposition_state.conj())
        
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        if len(returns) > 0:
            latest_return = returns[-1:]
            current_regime = kmeans.predict(latest_return)[0]
        else:
            current_regime = 0
        
        result = {
            'n_regimes': n_regimes,
            'regime_probabilities': regime_probs.tolist(),
            'superposition_state': superposition_state.tolist(),
            'entropy': float(entropy),
            'current_regime': int(current_regime),
            'regime_centroids': regime_centroids.tolist()
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'market_regime_superposition',
            'returns_shape': returns.shape,
            'n_regimes': n_regimes,
            'result': result
        })
        
        return result
    
    
    def detect_market_regime_quantum(self, quantum_state: Dict, returns: np.ndarray) -> Dict:
        """
        Detect market regime using quantum state
        
        Parameters:
        - quantum_state: Quantum state dictionary from create_market_quantum_state
        - returns: Array of asset returns
        
        Returns:
        - Dictionary with market regime detection results
        """
        if not isinstance(quantum_state, dict):
            logger.warning("Quantum state must be a dictionary")
            return {"regime": "unknown", "confidence": 0.0}
            
        regime = quantum_state.get("regime", "unknown")
        confidence = quantum_state.get("confidence", 0.0)
        entropy = quantum_state.get("entropy", 0.0)
        
        if len(returns) >= 5:
            recent_returns = returns[-5:]
            mean_return = np.mean(recent_returns)
            volatility = np.std(recent_returns)
            
            if mean_return > 0.5 * volatility and regime != "bearish":
                confidence = min(1.0, confidence * 1.2)
            elif mean_return < -0.5 * volatility and regime != "bullish":
                confidence = min(1.0, confidence * 1.2)
                
            if volatility > 0.02 and regime not in ["volatile", "reversal"]:
                confidence = max(0.5, confidence * 0.8)
        
        result = {
            "regime": regime,
            "confidence": float(confidence),
            "entropy": float(entropy),
            "is_high_entropy": bool(entropy > 2.0),
            "is_low_entropy": bool(entropy < 1.0)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_market_regime_quantum',
            'regime': regime,
            'confidence': float(confidence),
            'entropy': float(entropy)
        })
        
        return result
    
    def quantum_decision_amplitude(self, returns: np.ndarray, decision_type: str = 'buy') -> Dict:
        """
        Calculate quantum decision amplitude for trading decisions
        
        Parameters:
        - returns: Array of asset returns
        - decision_type: Type of decision ('buy', 'sell', 'hold')
        
        Returns:
        - Dictionary with decision amplitudes
        """
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        assets = returns.shape[1]
        
        density_matrix = self.create_density_matrix(returns)
        
        buy_projector = np.zeros((assets, assets))
        sell_projector = np.zeros((assets, assets))
        hold_projector = np.zeros((assets, assets))
        
        mean_returns = np.mean(returns[-20:], axis=0) if len(returns) >= 20 else np.mean(returns, axis=0)
        std_returns = np.std(returns[-20:], axis=0) if len(returns) >= 20 else np.std(returns, axis=0)
        
        for i in range(assets):
            if mean_returns[i] > 0.5 * std_returns[i]:
                buy_projector[i, i] = 1.0
            elif mean_returns[i] < -0.5 * std_returns[i]:
                sell_projector[i, i] = 1.0
            else:
                hold_projector[i, i] = 1.0
        
        buy_prob, _ = self.quantum_measurement(density_matrix, buy_projector)
        sell_prob, _ = self.quantum_measurement(density_matrix, sell_projector)
        hold_prob, _ = self.quantum_measurement(density_matrix, hold_projector)
        
        total_prob = buy_prob + sell_prob + hold_prob
        if total_prob > 0:
            buy_prob /= total_prob
            sell_prob /= total_prob
            hold_prob /= total_prob
        
        buy_amplitude = np.sqrt(buy_prob)
        sell_amplitude = np.sqrt(sell_prob)
        hold_amplitude = np.sqrt(hold_prob)
        
        if decision_type == 'buy':
            decision_amplitude = buy_amplitude
            confidence = buy_prob
        elif decision_type == 'sell':
            decision_amplitude = sell_amplitude
            confidence = sell_prob
        else:  # hold
            decision_amplitude = hold_amplitude
            confidence = hold_prob
        
        result = {
            'buy_amplitude': float(buy_amplitude),
            'sell_amplitude': float(sell_amplitude),
            'hold_amplitude': float(hold_amplitude),
            'buy_probability': float(buy_prob),
            'sell_probability': float(sell_prob),
            'hold_probability': float(hold_prob),
            'decision_amplitude': float(decision_amplitude),
            'confidence': float(confidence),
            'recommended_decision': max(['buy', 'sell', 'hold'], 
                                      key=lambda x: {'buy': buy_prob, 'sell': sell_prob, 'hold': hold_prob}[x])
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'quantum_decision_amplitude',
            'returns_shape': returns.shape,
            'decision_type': decision_type,
            'result': result
        })
        
        return result
    
    def quantum_portfolio_optimization(self, returns: np.ndarray, 
                                      risk_aversion: float = 1.0) -> Dict:
        """
        Optimize portfolio using quantum probability theory
        
        Parameters:
        - returns: Array of asset returns
        - risk_aversion: Risk aversion parameter (default: 1.0)
        
        Returns:
        - Dictionary with optimized portfolio weights
        """
        assets = returns.shape[1]
        
        density_matrix = self.create_density_matrix(returns)
        
        entanglement_result = self.detect_quantum_entanglement(returns)
        is_entangled = entanglement_result['is_entangled']
        
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns, rowvar=False)
        
        inv_cov = np.linalg.inv(cov_matrix)
        markowitz_weights = np.dot(inv_cov, mean_returns) / risk_aversion
        
        if np.sum(markowitz_weights) > 0:
            markowitz_weights = markowitz_weights / np.sum(markowitz_weights)
        
        if is_entangled:
            q_corr = self.quantum_correlation_matrix(returns)
            
            std_returns = np.std(returns, axis=0)
            q_cov = np.zeros_like(q_corr)
            for i in range(assets):
                for j in range(assets):
                    q_cov[i, j] = q_corr[i, j] * std_returns[i] * std_returns[j]
            
            q_inv_cov = np.linalg.inv(q_cov)
            quantum_weights = np.dot(q_inv_cov, mean_returns) / risk_aversion
            
            if np.sum(quantum_weights) > 0:
                quantum_weights = quantum_weights / np.sum(quantum_weights)
        else:
            quantum_weights = markowitz_weights.copy()
        
        markowitz_return = np.sum(markowitz_weights * mean_returns)
        markowitz_risk = np.sqrt(np.dot(markowitz_weights, np.dot(cov_matrix, markowitz_weights)))
        
        quantum_return = np.sum(quantum_weights * mean_returns)
        quantum_risk = np.sqrt(np.dot(quantum_weights, np.dot(cov_matrix, quantum_weights)))
        
        result = {
            'markowitz_weights': markowitz_weights.tolist(),
            'quantum_weights': quantum_weights.tolist(),
            'markowitz_return': float(markowitz_return),
            'markowitz_risk': float(markowitz_risk),
            'quantum_return': float(quantum_return),
            'quantum_risk': float(quantum_risk),
            'is_entangled': is_entangled,
            'entanglement_measure': float(entanglement_result['entanglement_measure'])
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'quantum_portfolio_optimization',
            'returns_shape': returns.shape,
            'risk_aversion': risk_aversion,
            'result': result
        })
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about quantum probability usage
        
        Returns:
        - Dictionary with usage statistics
        """
        if not self.history:
            return {'count': 0}
            
        operations = {}
        for h in self.history:
            op = h.get('operation', 'unknown')
            operations[op] = operations.get(op, 0) + 1
            
        return {
            'count': len(self.history),
            'operations': operations,
            'precision': self.precision,
            'confidence_level': self.confidence_level,
            'hilbert_space_dim': self.hilbert_space_dim
        }
