#!/usr/bin/env python3
"""
Quantum Risk Measures based on quantum entropy and non-Kolmogorovian probability

Implements coherent risk measures using quantum entropy (e.g., von Neumann entropy)
for stress-testing portfolios under quantum-correlated crashes.
Based on Accardi-Regoli framework.
"""

import numpy as np
import scipy.stats as stats
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumRiskMeasures")

class QuantumRiskMeasures:
    """
    Quantum Risk Measures based on quantum entropy and non-Kolmogorovian probability
    
    Implements coherent risk measures using quantum entropy for stress-testing portfolios
    under quantum-correlated crashes. Based on Accardi-Regoli framework.
    """
    
    def __init__(self, confidence_level=0.95, quantum_factor=0.3, crisis_amplification=1.5):
        """Initialize Quantum Risk Measures"""
        self.confidence_level = confidence_level
        self.quantum_factor = quantum_factor
        self.crisis_amplification = crisis_amplification
        self.min_entropy = 0.01
        self.max_entropy = 0.99
        self.history = []
        
        logger.info(f"Initialized QuantumRiskMeasures with confidence_level={confidence_level}, "
                   f"quantum_factor={quantum_factor}, crisis_amplification={crisis_amplification}")
        
    def _von_neumann_entropy(self, density_matrix):
        """Calculate von Neumann entropy: S(ρ) = -Tr(ρ log ρ) = -∑ λᵢ log λᵢ"""
        try:
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            return -np.sum(eigenvalues * np.log(eigenvalues))
        except Exception as e:
            logger.error(f"Error calculating von Neumann entropy: {str(e)}")
            return 0.0
        
    def _create_density_matrix(self, returns, correlation_matrix=None):
        """Create density matrix from returns data"""
        if len(returns.shape) > 1:
            n_assets = returns.shape[1]
        else:
            n_assets = 1
            
        if correlation_matrix is None:
            if n_assets > 1:
                correlation_matrix = np.corrcoef(returns.T)
            else:
                correlation_matrix = np.array([[1.0]])
                
        density_matrix = (correlation_matrix + np.eye(n_assets)) / (2 * n_assets)
        return density_matrix
        
    def _detect_crisis(self, returns, window=20, threshold=3.0):
        """Detect crisis conditions in returns data"""
        if len(returns) < window:
            return {'is_crisis': False, 'crisis_score': 0.0}
            
        vol = np.zeros(len(returns) - window + 1)
        for i in range(len(vol)):
            vol[i] = np.std(returns[i:i+window])
            
        current_vol = vol[-1]
        avg_vol = np.mean(vol[:-1]) if len(vol) > 1 else current_vol
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        returns_mean = np.mean(returns[-window:])
        returns_std = np.std(returns[-window:])
        returns_skew = 0
        if returns_std > 0:
            returns_skew = np.mean(((returns[-window:] - returns_mean) / returns_std)**3)
            
        returns_kurt = 3  # Normal distribution kurtosis
        if returns_std > 0:
            returns_kurt = np.mean(((returns[-window:] - returns_mean) / returns_std)**4)
            
        abs_returns = np.abs(returns[-window:])
        autocorr = 0
        if len(abs_returns) > 1:
            autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            
        crisis_score = (
            0.4 * (vol_ratio - 1) +
            0.3 * max(0, -returns_skew) +
            0.2 * max(0, (returns_kurt - 3) / 3) +
            0.1 * max(0, autocorr)
        )
        
        is_crisis = crisis_score > threshold
        
        return {
            'is_crisis': is_crisis,
            'crisis_score': float(crisis_score),
            'vol_ratio': float(vol_ratio),
            'returns_skew': float(returns_skew),
            'returns_kurt': float(returns_kurt),
            'autocorr': float(autocorr)
        }
        
    def quantum_var(self, returns, confidence_level=None, correlation_matrix=None):
        """Calculate Quantum Value at Risk (QVaR)"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(returns.shape) > 1:
            portfolio_returns = np.mean(returns, axis=1)
        else:
            portfolio_returns = returns
            
        classical_var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        crisis_result = self._detect_crisis(portfolio_returns)
        is_crisis = crisis_result['is_crisis']
        crisis_score = crisis_result['crisis_score']
        
        density_matrix = self._create_density_matrix(returns, correlation_matrix)
        entropy = self._von_neumann_entropy(density_matrix)
        
        max_entropy = np.log(density_matrix.shape[0])
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        normalized_entropy = max(self.min_entropy, min(self.max_entropy, normalized_entropy))
        
        quantum_factor = self.quantum_factor
        if is_crisis:
            quantum_factor *= self.crisis_amplification
            
        quantum_correction = normalized_entropy * np.std(portfolio_returns) * quantum_factor
        
        quantum_var = classical_var + quantum_correction
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'var',
            'classical_var': float(classical_var),
            'quantum_correction': float(quantum_correction),
            'quantum_var': float(quantum_var),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'confidence_level': float(confidence_level),
            'is_crisis': bool(is_crisis),
            'crisis_score': float(crisis_score)
        })
        
        return quantum_var
        
    def quantum_cvar(self, returns, confidence_level=None, correlation_matrix=None):
        """Calculate Quantum Conditional Value at Risk (QCVaR)"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(returns.shape) > 1:
            portfolio_returns = np.mean(returns, axis=1)
        else:
            portfolio_returns = returns
            
        var_cutoff = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        tail_returns = portfolio_returns[portfolio_returns <= var_cutoff]
        
        if len(tail_returns) > 0:
            classical_cvar = -np.mean(tail_returns)
        else:
            classical_cvar = -var_cutoff
            
        crisis_result = self._detect_crisis(portfolio_returns)
        is_crisis = crisis_result['is_crisis']
        crisis_score = crisis_result['crisis_score']
        
        density_matrix = self._create_density_matrix(returns, correlation_matrix)
        entropy = self._von_neumann_entropy(density_matrix)
        
        max_entropy = np.log(density_matrix.shape[0])
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        normalized_entropy = max(self.min_entropy, min(self.max_entropy, normalized_entropy))
        
        quantum_factor = self.quantum_factor * 1.5  # CVaR amplification
        if is_crisis:
            quantum_factor *= self.crisis_amplification
            
        quantum_correction = normalized_entropy * np.std(portfolio_returns) * quantum_factor
        
        quantum_cvar = classical_cvar + quantum_correction
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'cvar',
            'classical_cvar': float(classical_cvar),
            'quantum_correction': float(quantum_correction),
            'quantum_cvar': float(quantum_cvar),
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'confidence_level': float(confidence_level),
            'is_crisis': bool(is_crisis),
            'crisis_score': float(crisis_score)
        })
        
        return quantum_cvar
        
    def stress_test_portfolio(self, returns, weights, scenarios=1000, correlation_boost=0.3, 
                             volatility_boost=0.5, confidence_level=None):
        """Stress test portfolio under quantum-correlated crash scenarios"""
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
            
        n_assets = returns.shape[1]
        if len(weights) != n_assets:
            raise ValueError("Weights length must match number of assets")
            
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns, rowvar=False)
        
        stressed_cov = cov_matrix.copy()
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    if stressed_cov[i, i] > 0 and stressed_cov[j, j] > 0:
                        corr_ij = stressed_cov[i, j] / np.sqrt(stressed_cov[i, i] * stressed_cov[j, j])
                        
                        new_corr = min(1.0, corr_ij + correlation_boost * (1 - corr_ij))
                        
                        stressed_cov[i, j] = new_corr * np.sqrt(stressed_cov[i, i] * stressed_cov[j, j])
                        stressed_cov[j, i] = stressed_cov[i, j]
        
        for i in range(n_assets):
            stressed_cov[i, i] *= (1 + volatility_boost)
            
        np.random.seed(42)  # For reproducibility
        stressed_returns = np.random.multivariate_normal(mean_returns, stressed_cov, size=scenarios)
        
        portfolio_stressed_returns = np.dot(stressed_returns, weights)
        
        diag_sqrt = np.sqrt(np.diag(stressed_cov))
        diag_outer = np.outer(diag_sqrt, diag_sqrt)
        stressed_corr = stressed_cov / diag_outer
        
        stressed_var = self.quantum_var(stressed_returns, confidence_level, correlation_matrix=stressed_corr)
        stressed_cvar = self.quantum_cvar(stressed_returns, confidence_level, correlation_matrix=stressed_corr)
        
        worst_case_loss = -np.min(portfolio_stressed_returns)
        
        extreme_loss_threshold = 0.1  # 10% loss
        prob_extreme_loss = np.mean(portfolio_stressed_returns < -extreme_loss_threshold)
        
        tail_mean = np.mean(portfolio_stressed_returns[portfolio_stressed_returns < -stressed_var])
        tail_std = np.std(portfolio_stressed_returns[portfolio_stressed_returns < -stressed_var])
        
        quantum_tail_index = 0.0
        if len(portfolio_stressed_returns[portfolio_stressed_returns < -stressed_var]) > 10:
            tail_returns = stressed_returns[portfolio_stressed_returns < -stressed_var]
            tail_corr = np.corrcoef(tail_returns.T)
            
            n = tail_corr.shape[0]
            if n > 1:
                avg_tail_corr = (np.sum(tail_corr) - n) / (n**2 - n)
                
                quantum_tail_index = 0.7 * avg_tail_corr + 0.3 * (tail_std / abs(tail_mean) if tail_mean != 0 else 0)
        
        results = {
            'stressed_var': float(stressed_var),
            'stressed_cvar': float(stressed_cvar),
            'worst_case_loss': float(worst_case_loss),
            'prob_extreme_loss': float(prob_extreme_loss),
            'mean_stress_return': float(np.mean(portfolio_stressed_returns)),
            'stress_volatility': float(np.std(portfolio_stressed_returns)),
            'tail_mean': float(tail_mean) if not np.isnan(tail_mean) else None,
            'tail_std': float(tail_std) if not np.isnan(tail_std) else None,
            'quantum_tail_index': float(quantum_tail_index),
            'scenarios': scenarios
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'stress_test',
            'stress_test_results': results,
            'n_assets': n_assets,
            'scenarios': scenarios
        })
        
        return results
        
    def calculate_portfolio_risk(self, portfolio, confidence_level=None):
        """
        Calculate portfolio risk metrics using quantum risk measures
        
        Parameters:
        - portfolio: Dictionary with asset positions
        - confidence_level: Confidence level for risk calculations
        
        Returns:
        - Dictionary with risk metrics
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        assets = list(portfolio.keys())
        weights = [position.get("weight", 0.0) for position in portfolio.values()]
        
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
            
        n_days = 252  # One year of trading days
        n_assets = len(assets)
        
        portfolio_vol = 0.0
        for position in portfolio.values():
            if "volatility" in position:
                portfolio_vol += position.get("volatility", 0.0)
                
        if portfolio_vol == 0.0:
            portfolio_vol = 0.15  # Default annual volatility
            
        daily_vol = portfolio_vol / np.sqrt(252)
        
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, daily_vol, (n_days, n_assets))
        
        var = self.quantum_var(returns, confidence_level)
        cvar = self.quantum_cvar(returns, confidence_level)
        
        stress_results = self.stress_test_portfolio(
            returns, weights, scenarios=500, confidence_level=confidence_level
        )
        
        risk_metrics = {
            "var": float(var),
            "cvar": float(cvar),
            "stressed_var": stress_results.get("stressed_var", 0.0),
            "stressed_cvar": stress_results.get("stressed_cvar", 0.0),
            "worst_case_loss": stress_results.get("worst_case_loss", 0.0),
            "prob_extreme_loss": stress_results.get("prob_extreme_loss", 0.0),
            "confidence_level": float(confidence_level),
            "assets_count": n_assets
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'portfolio_risk',
            'risk_metrics': risk_metrics,
            'n_assets': n_assets
        })
        
        return risk_metrics
        
    def get_statistics(self):
        """Get statistics about risk measure calculations"""
        if not self.history:
            return {'count': 0}
            
        var_count = sum(1 for h in self.history if h.get('type') == 'var')
        cvar_count = sum(1 for h in self.history if h.get('type') == 'cvar')
        stress_test_count = sum(1 for h in self.history if h.get('type') == 'stress_test')
        
        var_values = [h.get('quantum_var', 0) for h in self.history if h.get('type') == 'var']
        cvar_values = [h.get('quantum_cvar', 0) for h in self.history if h.get('type') == 'cvar']
        
        avg_var = np.mean(var_values) if var_values else 0
        avg_cvar = np.mean(cvar_values) if cvar_values else 0
        
        crisis_count = sum(1 for h in self.history if h.get('is_crisis', False))
        
        var_corrections = [h.get('quantum_correction', 0) for h in self.history if h.get('type') == 'var']
        cvar_corrections = [h.get('quantum_correction', 0) for h in self.history if h.get('type') == 'cvar']
        
        avg_var_correction = np.mean(var_corrections) if var_corrections else 0
        avg_cvar_correction = np.mean(cvar_corrections) if cvar_corrections else 0
        
        entropy_values = [h.get('entropy', 0) for h in self.history if 'entropy' in h]
        avg_entropy = np.mean(entropy_values) if entropy_values else 0
        
        stats = {
            'count': len(self.history),
            'var_count': var_count,
            'cvar_count': cvar_count,
            'stress_test_count': stress_test_count,
            'crisis_count': crisis_count,
            'avg_var': float(avg_var),
            'avg_cvar': float(avg_cvar),
            'avg_var_correction': float(avg_var_correction),
            'avg_cvar_correction': float(avg_cvar_correction),
            'avg_entropy': float(avg_entropy)
        }
        
        return stats
        
    def clear_history(self):
        """Clear calculation history"""
        self.history = []
        logger.info("Calculation history cleared")


if __name__ == "__main__":
    import unittest
    
    class TestQuantumRiskMeasures(unittest.TestCase):
        """Unit tests for QuantumRiskMeasures"""
        
        def setUp(self):
            """Set up test fixtures"""
            self.qrm = QuantumRiskMeasures(confidence_level=0.95, quantum_factor=0.3)
            
            np.random.seed(42)
            self.n_days = 252
            self.n_assets = 3
            
            self.normal_returns = np.random.normal(0.0005, 0.01, (self.n_days, self.n_assets))
            
            self.crisis_returns = self.normal_returns.copy()
            
            self.crisis_correlation = np.array([
                [1.0, 0.8, 0.7],
                [0.8, 1.0, 0.9],
                [0.7, 0.9, 1.0]
            ])
            
            for i in range(self.n_days - 20, self.n_days):
                self.crisis_returns[i] = np.random.multivariate_normal(
                    [-0.02, -0.02, -0.02],
                    self.crisis_correlation * 0.03**2,
                    1
                )[0]
                
            self.weights = np.array([0.4, 0.3, 0.3])
            
        def test_von_neumann_entropy(self):
            """Test von Neumann entropy calculation"""
            density_matrix = np.array([
                [0.5, 0.0],
                [0.0, 0.5]
            ])
            
            entropy = self.qrm._von_neumann_entropy(density_matrix)
            
            expected_entropy = np.log(2)
            
            self.assertAlmostEqual(entropy, expected_entropy, delta=1e-6)
            
        def test_quantum_var(self):
            """Test Quantum Value at Risk calculation"""
            normal_var = self.qrm.quantum_var(self.normal_returns)
            
            crisis_var = self.qrm.quantum_var(self.crisis_returns)
            
            self.assertGreater(crisis_var, normal_var)
            
        def test_quantum_cvar(self):
            """Test Quantum Conditional Value at Risk calculation"""
            normal_cvar = self.qrm.quantum_cvar(self.normal_returns)
            
            crisis_cvar = self.qrm.quantum_cvar(self.crisis_returns)
            
            self.assertGreater(crisis_cvar, normal_cvar)
            
            normal_var = self.qrm.quantum_var(self.normal_returns)
            crisis_var = self.qrm.quantum_var(self.crisis_returns)
            
            self.assertGreaterEqual(normal_cvar, normal_var)
            self.assertGreaterEqual(crisis_cvar, crisis_var)
            
        def test_stress_test_portfolio(self):
            """Test portfolio stress testing"""
            normal_results = self.qrm.stress_test_portfolio(
                self.normal_returns, self.weights, scenarios=100
            )
            
            crisis_results = self.qrm.stress_test_portfolio(
                self.crisis_returns, self.weights, scenarios=100
            )
            
            self.assertGreater(crisis_results['stressed_var'], normal_results['stressed_var'])
            self.assertGreater(crisis_results['stressed_cvar'], normal_results['stressed_cvar'])
            self.assertGreater(crisis_results['worst_case_loss'], normal_results['worst_case_loss'])
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
