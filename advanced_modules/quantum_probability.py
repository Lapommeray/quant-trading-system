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
        if returns.size == 0 or len(returns.shape) < 2:
            logger.warning(f"Insufficient data for density matrix creation: shape={returns.shape}, size={returns.size}")
            return np.eye(2, dtype=complex)
            
        if returns.shape[0] < 2 or returns.shape[1] < 1:
            logger.warning(f"Insufficient samples for density matrix: shape={returns.shape}")
            return np.eye(max(2, returns.shape[1]), dtype=complex)
        
        if normalize:
            std_dev = np.std(returns, axis=0)
            std_dev = np.where(std_dev < 1e-10, 1.0, std_dev)
            returns = (returns - np.mean(returns, axis=0)) / std_dev
        
        try:
            cov_matrix = np.cov(returns, rowvar=False)
            
            if cov_matrix.size <= 1:
                logger.warning("Covariance matrix too small, using fallback")
                return np.eye(max(2, returns.shape[1]), dtype=complex)
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative eigenvalues
            
            density_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            if normalize and np.trace(density_matrix) > 0:
                density_matrix = density_matrix / np.trace(density_matrix)
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'create_density_matrix',
                'returns_shape': returns.shape,
                'density_matrix_shape': density_matrix.shape,
                'eigenvalues_min': float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0,
                'eigenvalues_max': float(np.max(eigenvalues)) if eigenvalues.size > 0 else 0.0
            })
            
            return density_matrix
            
        except Exception as e:
            logger.warning(f"Error in density matrix creation: {str(e)}")
            return np.eye(max(2, returns.shape[1]), dtype=complex)
    
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
        - returns: Array of asset returns with shape (time_steps, assets) or (time_steps,)
        - window_size: Window size for time averaging (default: 20)
        
        Returns:
        - Dictionary with non-ergodicity metrics
        """
        if returns is None or returns.size == 0:
            return {
                'mean_difference': 0.0,
                'variance_ratio': 1.0,
                'ergodicity_breaking': 0.0,
                'non_ergodicity_index': 0.0,
                'is_ergodic': True
            }
            
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        time_steps, assets = returns.shape
        if time_steps < window_size + 1:
            return {
                'mean_difference': 0.0,
                'variance_ratio': 1.0,
                'ergodicity_breaking': 0.0,
                'non_ergodicity_index': 0.0,
                'is_ergodic': True
            }
        
        ensemble_means = np.mean(returns, axis=1)
        ensemble_vars = np.var(returns, axis=1)
        
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
        return self.calculate_non_ergodic_kelly(returns, risk_free_rate, time_horizon)
        
    def calculate_non_ergodic_kelly(self, returns: np.ndarray, risk_free_rate: Union[float, Dict] = 0.0,
                                  time_horizon: int = 252) -> Dict:
        """
        Calculate Kelly criterion under non-ergodic conditions
        
        Parameters:
        - returns: Array of asset returns
        - risk_free_rate: Risk-free rate (default: 0.0) or dict with rate information
        - time_horizon: Investment time horizon in days (default: 252)
        
        Returns:
        - Dictionary with Kelly allocations and non-ergodic adjustments
        """
        if returns is None or returns.size == 0:
            return {
                'kelly_weights': [0.0],
                'is_ergodic': True,
                'non_ergodicity_index': 0.0,
                'expected_growth_rate': 0.0,
                'optimal_fraction': 0.0,
                'confidence': 0.5
            }
            
        if isinstance(risk_free_rate, dict):
            rf_rate = risk_free_rate.get('rate', 0.0)
            if not isinstance(rf_rate, (int, float)):
                rf_rate = 0.0
        else:
            rf_rate = float(risk_free_rate)
            
        mean_returns = np.mean(returns, axis=0)
        
        try:
            cov_matrix = np.cov(returns, rowvar=False)
            if cov_matrix.size == 0 or np.isnan(cov_matrix).any():
                raise ValueError("Invalid covariance matrix")
        except Exception as e:
            logger.warning(f"Covariance matrix calculation error: {e}, using fallback")
            if len(returns.shape) > 1 and returns.shape[1] > 1:
                cov_matrix = np.diag(np.var(returns, axis=0))
            else:
                cov_matrix = np.array([[np.var(returns)]])
        
        non_ergodic_metrics = self.time_average_vs_ensemble_average(returns)
        is_ergodic = non_ergodic_metrics['is_ergodic']
        
        if np.isscalar(mean_returns):
            mean_returns = np.array([mean_returns])
            
        # Calculate excess returns
        excess_returns = mean_returns - rf_rate
        
        if np.isscalar(excess_returns):
            excess_returns = np.array([excess_returns])
            
        try:
            if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim < 2 or cov_matrix.shape[0] == 0 or cov_matrix.shape[1] == 0:
                raise ValueError("Covariance matrix has invalid dimensions")
                
            if len(excess_returns.shape) == 0:  # 0-dimensional array
                excess_returns = np.array([float(excess_returns)])
                cov_matrix = np.array([[1.0]])
            elif cov_matrix.shape[0] != excess_returns.shape[0]:
                raise ValueError(f"Dimension mismatch: cov_matrix {cov_matrix.shape}, excess_returns {excess_returns.shape}")
                
            kelly_weights_ergodic = np.linalg.solve(cov_matrix, excess_returns)
            
            if not is_ergodic:
                growth_rates = np.log(1 + returns)
                time_avg_growth = np.mean(growth_rates, axis=0)
                
                if np.isscalar(time_avg_growth):
                    time_avg_growth = np.array([time_avg_growth])
                
                non_ergodic_factor = non_ergodic_metrics['non_ergodicity_index']
                adjusted_returns = time_avg_growth * (1 - non_ergodic_factor) + mean_returns * non_ergodic_factor
                
                excess_returns_adjusted = adjusted_returns - rf_rate
                kelly_weights_non_ergodic = np.linalg.solve(cov_matrix, excess_returns_adjusted)
                
                scaling_factor = 1.0 - 0.5 * non_ergodic_factor
                kelly_weights = kelly_weights_non_ergodic * scaling_factor
            else:
                kelly_weights = kelly_weights_ergodic
        except Exception as e:
            logger.warning(f"Error in Kelly calculation: {e}, using fallback")
            kelly_weights = np.array([0.5]) if np.isscalar(mean_returns) else np.ones_like(mean_returns) / len(mean_returns)
        
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
    
    
    def quantum_decision_amplitude(self, returns: np.ndarray, decision_type: str = 'buy') -> Dict:
        """
        Calculate quantum decision amplitude for trading decisions
        
        Parameters:
        - returns: Array of asset returns
        - decision_type: Type of decision ('buy', 'sell', 'hold')
        
        Returns:
        - Dictionary with decision amplitudes
        """
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
    
    def create_market_quantum_state(self, returns: np.ndarray, n_qubits: int = 5) -> Dict:
        """
        Create quantum state representation of market returns
        
        Parameters:
        - returns: Array of asset returns
        - n_qubits: Number of qubits to use for representation
        
        Returns:
        - Dictionary with quantum state representation
        """
        target_dim = 2**n_qubits
        
        if returns is None or returns.size == 0:
            logger.warning("Empty returns array for quantum state creation")
            return {
                "state_vector": np.zeros(target_dim, dtype=complex),
                "density_matrix": np.eye(target_dim, dtype=complex),
                "eigenvalues": np.array([1.0] + [0.0] * (target_dim - 1)),
                "entropy": 0.0,
                "n_qubits": n_qubits,
                "dimension": target_dim
            }
        
        try:
            if len(returns.shape) == 1:
                std_dev = np.std(returns) + 1e-8
                returns_norm = (returns - np.mean(returns)) / std_dev
                returns_norm = returns_norm.reshape(-1, 1)
            else:
                std_dev = np.std(returns, axis=0) + 1e-8
                returns_norm = (returns - np.mean(returns, axis=0)) / std_dev
                
            # Create density matrix
            density_matrix = self.create_density_matrix(returns_norm)
            
            current_dim = density_matrix.shape[0]
            
            if current_dim < target_dim:
                padded_matrix = np.zeros((target_dim, target_dim), dtype=complex)
                padded_matrix[:current_dim, :current_dim] = density_matrix
                trace = np.trace(padded_matrix)
                density_matrix = padded_matrix / trace if trace > 0 else padded_matrix
            elif current_dim > target_dim:
                density_matrix = density_matrix[:target_dim, :target_dim]
                trace = np.trace(density_matrix)
                density_matrix = density_matrix / trace if trace > 0 else density_matrix
                
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
            
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Get principal eigenvector as state vector
            state_vector = eigenvectors[:, 0]
            
            # Calculate quantum entropy safely
            positive_mask = eigenvalues > 1e-10
            positive_eigenvalues = eigenvalues[positive_mask]
            if positive_eigenvalues.size > 0:
                entropy = float(np.sum(-positive_eigenvalues * np.log2(positive_eigenvalues)))
            else:
                entropy = 0.0
            
            result = {
                "state_vector": state_vector,
                "density_matrix": density_matrix,
                "eigenvalues": eigenvalues,
                "entropy": entropy,
                "n_qubits": n_qubits,
                "dimension": target_dim
            }
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'create_market_quantum_state',
                'returns_shape': returns.shape,
                'n_qubits': n_qubits,
                'entropy': entropy
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in quantum state creation: {str(e)}")
            return {
                "state_vector": np.zeros(target_dim, dtype=complex),
                "density_matrix": np.eye(target_dim, dtype=complex),
                "eigenvalues": np.array([1.0] + [0.0] * (target_dim - 1)),
                "entropy": 0.0,
                "n_qubits": n_qubits,
                "dimension": target_dim
            }
    
    def detect_market_regime_quantum(self, quantum_state: Dict, returns: np.ndarray) -> Dict:
        """
        Detect market regime using quantum state
        
        Parameters:
        - quantum_state: Quantum state representation from create_market_quantum_state
        - returns: Array of asset returns
        
        Returns:
        - Dictionary with detected market regime
        """
        if not isinstance(quantum_state, dict) or "density_matrix" not in quantum_state:
            logger.warning("Invalid quantum state for regime detection")
            return {"regime": "neutral", "confidence": 0.0, 
                    "bull_probability": 0.0, "bear_probability": 0.0, "neutral_probability": 1.0}
            
        try:
            density_matrix = quantum_state.get("density_matrix")
            if density_matrix is None or not isinstance(density_matrix, np.ndarray):
                logger.warning("Invalid density matrix for regime detection")
                return {"regime": "neutral", "confidence": 0.0, 
                        "bull_probability": 0.0, "bear_probability": 0.0, "neutral_probability": 1.0}
            
            dim = density_matrix.shape[0]
            
            bull_size = max(1, dim // 4)
            bull_observable = np.zeros_like(density_matrix)
            bull_observable[:bull_size, :bull_size] = np.eye(bull_size)
            
            bear_size = max(1, dim // 4)
            bear_observable = np.zeros_like(density_matrix)
            bear_observable[-bear_size:, -bear_size:] = np.eye(bear_size)
            
            # Neutral regime: middle eigenvalues
            neutral_size = max(1, dim // 2)
            start_idx = max(0, (dim - neutral_size) // 2)
            neutral_observable = np.zeros_like(density_matrix)
            neutral_observable[start_idx:start_idx+neutral_size, start_idx:start_idx+neutral_size] = np.eye(neutral_size)
            
            # Perform quantum measurements
            bull_prob, _ = self.quantum_measurement(density_matrix, bull_observable)
            bear_prob, _ = self.quantum_measurement(density_matrix, bear_observable)
            neutral_prob, _ = self.quantum_measurement(density_matrix, neutral_observable)
            
            total_prob = bull_prob + bear_prob + neutral_prob
            if total_prob > 0:
                bull_prob /= total_prob
                bear_prob /= total_prob
                neutral_prob /= total_prob
            else:
                bull_prob, bear_prob, neutral_prob = 0.0, 0.0, 1.0
            
            # Determine regime with highest probability
            probs = {"bullish": bull_prob, "bearish": bear_prob, "neutral": neutral_prob}
            regime = "neutral"  # Default
            max_prob = 0.0
            
            for key, value in probs.items():
                if value > max_prob:
                    max_prob = value
                    regime = key
                    
            confidence = probs[regime]
            
            if returns is not None and returns.size >= 5:
                if len(returns.shape) == 1:
                    recent_returns = returns[-5:]
                else:
                    recent_returns = returns[-5:, 0] if returns.shape[1] > 0 else np.zeros(5)
                    
                recent_trend = np.mean(recent_returns)
                if recent_trend > 0 and regime == "bearish" and bull_prob > 0.3:
                    regime = "bullish"
                    confidence = bull_prob
                elif recent_trend < 0 and regime == "bullish" and bear_prob > 0.3:
                    regime = "bearish"
                    confidence = bear_prob
            
            result = {
                "regime": regime,
                "confidence": float(confidence),
                "bull_probability": float(bull_prob),
                "bear_probability": float(bear_prob),
                "neutral_probability": float(neutral_prob),
                "entropy": float(quantum_state.get("entropy", 0.0))
            }
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'detect_market_regime_quantum',
                'regime': regime,
                'confidence': float(confidence)
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in market regime detection: {str(e)}")
            return {"regime": "neutral", "confidence": 0.0, 
                    "bull_probability": 0.0, "bear_probability": 0.0, "neutral_probability": 1.0}
    
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
