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
        if returns is None or len(returns) == 0:
            logger.warning("Empty returns data provided to create_density_matrix")
            return np.eye(1)
        
        returns = np.asarray(returns)
        
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
        
        if returns.shape[0] < 2 or returns.shape[1] < 1:
            logger.warning(f"Insufficient data for density matrix: shape {returns.shape}")
            return np.eye(max(1, returns.shape[1]))
        
        try:
            if normalize:
                std = np.std(returns, axis=0)
                std[std == 0] = 1.0  # Avoid division by zero
                returns = (returns - np.mean(returns, axis=0)) / std
            
            cov_matrix = np.cov(returns, rowvar=False)
            
            try:
                if hasattr(cov_matrix, 'shape') and len(cov_matrix.shape) == 2 and min(cov_matrix.shape) > 0:
                    pass  # Keep as is
                elif isinstance(cov_matrix, (int, float)):
                    cov_matrix = np.array([[float(max(cov_matrix, 0.0001))]])
                elif hasattr(cov_matrix, 'item') and callable(getattr(cov_matrix, 'item')):
                    try:
                        val = float(cov_matrix.item())
                        cov_matrix = np.array([[max(val, 0.0001)]])
                    except:
                        cov_matrix = np.array([[0.0001]])
                elif hasattr(cov_matrix, 'shape') and len(cov_matrix.shape) == 1 and cov_matrix.shape[0] > 0:
                    cov_matrix = np.outer(cov_matrix, cov_matrix)
                else:
                    cov_matrix = np.array([[0.0001]])
            except Exception as e:
                logger.warning(f"Error handling cov_matrix conversion: {str(e)}")
                cov_matrix = np.array([[0.0001]])
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative eigenvalues
            
            density_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            if normalize:
                trace = np.trace(density_matrix)
                if trace > 0:
                    density_matrix = density_matrix / trace
                else:
                    density_matrix = np.eye(density_matrix.shape[0])
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'create_density_matrix',
                'returns_shape': returns.shape,
                'density_matrix_shape': density_matrix.shape,
                'eigenvalues_min': float(min(eigenvalues)) if len(eigenvalues) > 0 else 0.0,
                'eigenvalues_max': float(max(eigenvalues)) if len(eigenvalues) > 0 else 1.0
            })
            
            return density_matrix
            
        except Exception as e:
            logger.warning(f"Error creating density matrix: {str(e)}")
            dim = returns.shape[1] if len(returns.shape) > 1 else 1
            return np.eye(dim)
    
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
        time_steps, assets = returns.shape
        
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
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns, rowvar=False)
        
        non_ergodic_metrics = self.time_average_vs_ensemble_average(returns)
        is_ergodic = non_ergodic_metrics['is_ergodic']
        
        excess_returns = mean_returns - risk_free_rate
        kelly_weights_ergodic = np.linalg.solve(cov_matrix, excess_returns)
        
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
    
    def create_market_quantum_state(self, returns: Union[List, np.ndarray], 
                                 n_qubits: int = 4,
                                 volumes: Optional[Union[List, np.ndarray]] = None) -> Dict:
        """
        Create quantum state representation of market data
        
        Parameters:
        - returns: Array of asset returns
        - n_qubits: Number of qubits for quantum state (default: 4)
        - volumes: Optional array of trading volumes
        
        Returns:
        - Dictionary with quantum state representation
        """
        try:
            if returns is None or len(returns) == 0:
                logger.warning("Empty returns data provided to create_market_quantum_state")
                # Return default quantum state
                return self._create_default_quantum_state(n_qubits)
            
            returns = np.asarray(returns)
            
            if len(returns.shape) == 1:
                returns = returns.reshape(-1, 1)
                
            if returns.shape[0] < 2:
                logger.warning(f"Insufficient data points for quantum state: {returns.shape[0]}")
                return self._create_default_quantum_state(n_qubits)
                
            if volumes is not None:
                volumes = np.asarray(volumes)
                if len(volumes.shape) == 1:
                    volumes = volumes.reshape(-1, 1)
            
            # Create density matrix from returns
            density_matrix = self.create_density_matrix(returns)
            
            try:
                market_regimes = self.market_regime_superposition(returns)
            except Exception as e:
                logger.warning(f"Error in market_regime_superposition: {str(e)}")
                market_regimes = {
                    "n_regimes": 3,
                    "regime_probabilities": [0.33, 0.33, 0.34],
                    "current_regime": 0,
                    "entropy": 1.0
                }
            
            try:
                kelly = self.kelly_criterion_non_ergodic(returns)
            except Exception as e:
                logger.warning(f"Error in kelly_criterion_non_ergodic: {str(e)}")
                kelly = {
                    "kelly_weights": [0.0] * returns.shape[1],
                    "is_ergodic": True,
                    "non_ergodicity_index": 0.0,
                    "expected_growth_rate": 0.0
                }
            
            try:
                entanglement = self.detect_quantum_entanglement(returns)
            except Exception as e:
                logger.warning(f"Error in detect_quantum_entanglement: {str(e)}")
                entanglement = {
                    "von_neumann_entropy": 0.0,
                    "purity": 1.0,
                    "entanglement_measure": 0.0,
                    "entangled_pairs": [],
                    "is_entangled": False
                }
            
            try:
                decision = self.quantum_decision_amplitude(returns)
            except Exception as e:
                logger.warning(f"Error in quantum_decision_amplitude: {str(e)}")
                decision = {
                    "buy_confidence": 0.33,
                    "sell_confidence": 0.33,
                    "hold_confidence": 0.34,
                    "decision": "hold",
                    "amplitude": 0.5
                }
            
            try:
                confidence = max(
                    decision.get("buy_confidence", 0.0),
                    market_regimes.get("regime_probabilities", [0.0])[market_regimes.get("current_regime", 0)] 
                        if market_regimes.get("regime_probabilities") and len(market_regimes.get("regime_probabilities", [])) > market_regimes.get("current_regime", 0) else 0.0,
                    kelly.get("expected_growth_rate", 0.0) if kelly.get("expected_growth_rate", 0.0) > 0 else 0.0
                )
            except Exception as e:
                logger.warning(f"Error calculating confidence: {str(e)}")
                confidence = 0.5
            
            # Create combined quantum state
            quantum_state = {
                "density_matrix": density_matrix.tolist() if hasattr(density_matrix, 'tolist') else [[1.0]],
                "market_regimes": market_regimes,
                "kelly_criterion": kelly,
                "entanglement": entanglement,
                "decision_amplitudes": decision,
                "confidence": confidence
            }
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'create_market_quantum_state',
                'returns_shape': returns.shape,
                'volumes_shape': volumes.shape if volumes is not None else None,
                'result': {k: v for k, v in quantum_state.items() if k != "density_matrix"}
            })
            
            return quantum_state
            
        except Exception as e:
            logger.warning(f"Error in create_market_quantum_state: {str(e)}")
            return self._create_default_quantum_state(n_qubits)
    
    def _create_default_quantum_state(self, n_qubits: int = 4) -> Dict:
        """
        Create default quantum state when data is insufficient
        
        Parameters:
        - n_qubits: Number of qubits for quantum state
        
        Returns:
        - Default quantum state dictionary
        """
        density_matrix = np.eye(2**n_qubits) / (2**n_qubits)
        
        quantum_state = {
            "density_matrix": density_matrix.tolist(),
            "market_regimes": {
                "n_regimes": 3,
                "regime_probabilities": [0.33, 0.33, 0.34],
                "superposition_state": [0.57735, 0.57735, 0.57735],
                "entropy": 1.58496,
                "current_regime": 0,
                "regime_centroids": [[0.0]]
            },
            "kelly_criterion": {
                "kelly_weights": [0.0] * n_qubits,
                "is_ergodic": True,
                "non_ergodicity_index": 0.0,
                "expected_growth_rate": 0.0
            },
            "entanglement": {
                "von_neumann_entropy": 0.0,
                "purity": 1.0,
                "entanglement_measure": 0.0,
                "entangled_pairs": [],
                "is_entangled": False
            },
            "decision_amplitudes": {
                "buy_confidence": 0.33,
                "sell_confidence": 0.33,
                "hold_confidence": 0.34,
                "decision": "hold",
                "amplitude": 0.5
            },
            "confidence": 0.5
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_default_quantum_state',
            'n_qubits': n_qubits
        })
        
        return quantum_state
    
    def detect_market_regime_quantum(self, quantum_state: Dict, returns: np.ndarray) -> Dict:
        """
        Detect market regime using quantum state information
        
        Parameters:
        - quantum_state: Quantum state representation of market data
        - returns: Array of asset returns
        
        Returns:
        - Dictionary with market regime information
        """
        # Extract market regime information from quantum state
        market_regimes = quantum_state.get("market_regimes", {})
        
        current_regime = market_regimes.get("current_regime", 0)
        regime_probs = market_regimes.get("regime_probabilities", [0.33, 0.33, 0.34])
        
        regime_labels = ["bullish", "neutral", "bearish"]
        if current_regime < len(regime_labels):
            regime_name = regime_labels[current_regime]
        else:
            regime_name = "unknown"
            
        # Calculate confidence based on probability of current regime
        if isinstance(regime_probs, list) and len(regime_probs) > current_regime:
            confidence = regime_probs[current_regime]
        else:
            confidence = 0.5
            
        if regime_name == "bullish":
            signal = 1
        elif regime_name == "bearish":
            signal = -1
        else:
            signal = 0
            
        # Get additional information from quantum state
        entanglement = quantum_state.get("entanglement", {})
        kelly = quantum_state.get("kelly_criterion", {})
        decision = quantum_state.get("decision_amplitudes", {})
        
        result = {
            "current_regime": regime_name,
            "regime_index": current_regime,
            "confidence": float(confidence),
            "signal": signal,
            "is_entangled": entanglement.get("is_entangled", False),
            "optimal_fraction": kelly.get("optimal_fraction", 0.0),
            "decision": decision.get("decision", "hold")
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'detect_market_regime_quantum',
            'returns_shape': returns.shape if hasattr(returns, 'shape') else None,
            'result': result
        })
        
        return result
    
    def calculate_non_ergodic_kelly(self, returns: np.ndarray, quantum_state: Dict) -> Dict:
        """
        Calculate non-ergodic Kelly criterion using quantum state
        
        Parameters:
        - returns: Array of asset returns
        - quantum_state: Quantum state representation of market data
        
        Returns:
        - Dictionary with non-ergodic Kelly criterion
        """
        if "kelly_criterion" in quantum_state:
            kelly = quantum_state["kelly_criterion"]
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'calculate_non_ergodic_kelly',
                'returns_shape': returns.shape if hasattr(returns, 'shape') else None,
                'result': {k: v for k, v in kelly.items() if k != "kelly_weights"}
            })
            
            return kelly
            
        kelly = self.kelly_criterion_non_ergodic(returns)
        
        return kelly
    
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
