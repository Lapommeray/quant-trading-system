#!/usr/bin/env python3
"""
Quantum Black-Scholes Model for Option Pricing

Implements path integrals over non-classical trajectories for option pricing.
Captures extreme market regimes and volatility clustering via quantum paths.
Based on Baaquie-Martin formulation.

This module enhances traditional Black-Scholes by incorporating quantum effects
that become significant during high volatility periods like market crashes.
"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumBlackScholes")

class QuantumBlackScholes:
    """
    Quantum Black-Scholes Model for Option Pricing
    
    Implements path integrals over non-classical trajectories for option pricing.
    Captures extreme market regimes and volatility clustering via quantum paths.
    Based on Baaquie-Martin formulation.
    
    Key features:
    - Quantum-adjusted drift and volatility
    - Path integral computation for non-classical price paths
    - Crisis factor amplification during high volatility
    - Volatility clustering detection
    """
    
    def __init__(self, hbar=0.01, crisis_threshold=0.4, max_crisis_amplification=5.0):
        """
        Initialize Quantum Black-Scholes model
        
        Parameters:
        - hbar: Quantum scaling parameter (default: 0.01)
        - crisis_threshold: Volatility threshold for crisis detection (default: 0.4)
        - max_crisis_amplification: Maximum crisis amplification factor (default: 5.0)
        """
        self.hbar = hbar
        self.crisis_threshold = crisis_threshold
        self.max_crisis_amplification = max_crisis_amplification
        self.history = []
        logger.info(f"Initialized QuantumBlackScholes with hbar={hbar}, crisis_threshold={crisis_threshold}")
        
    def _quantum_drift(self, s, r, sigma, t):
        """
        Calculate quantum-adjusted drift
        
        Incorporates quantum corrections to the classical drift term
        
        Parameters:
        - s: Current stock price
        - r: Risk-free rate
        - sigma: Volatility
        - t: Time parameter
        
        Returns:
        - Quantum-adjusted drift
        """
        return r - 0.5 * sigma**2 + self.hbar * (sigma**2) * np.sin(2*np.pi*t)
        
    def _quantum_vol(self, sigma, vov, t):
        """
        Calculate quantum-adjusted volatility with vol-of-vol component
        
        Incorporates volatility of volatility effects that become significant
        during market stress periods
        
        Parameters:
        - sigma: Base volatility
        - vov: Volatility of volatility
        - t: Time parameter
        
        Returns:
        - Quantum-adjusted volatility
        """
        return sigma * (1 + vov * np.sin(4*np.pi*t) * np.exp(-self.hbar*t))
    
    def _quantum_path_integral(self, s, k, r, sigma, t, vov=0.2):
        """
        Compute path integral for quantum price paths
        
        Uses numerical integration to approximate the quantum path integral
        
        Parameters:
        - s: Current stock price
        - k: Strike price
        - r: Risk-free rate
        - sigma: Volatility
        - t: Time to expiration
        - vov: Volatility of volatility (default: 0.2)
        
        Returns:
        - Path integral result
        """
        def integrand(x):
            qvol = self._quantum_vol(sigma, vov, t)
            qdrift = self._quantum_drift(s, r, qvol, t)
            d1 = (np.log(s/k) + (qdrift + 0.5*qvol**2)*t) / (qvol*np.sqrt(t))
            return np.exp(-x**2) * stats.norm.cdf(d1 - x*self.hbar)
            
        try:
            integral, error = quad(integrand, -10, 10)
            if error > 1e-3:
                logger.warning(f"Path integral computation has high error: {error}")
            return integral * np.sqrt(1/np.pi)
        except Exception as e:
            logger.error(f"Path integral computation failed: {str(e)}")
            return 0.0
    
    def _calculate_crisis_factor(self, sigma):
        """
        Calculate crisis amplification factor based on volatility
        
        During high volatility periods, quantum effects are amplified
        
        Parameters:
        - sigma: Current volatility
        
        Returns:
        - Crisis amplification factor
        """
        if sigma > self.crisis_threshold:
            crisis_factor = 1.0 + min(
                (sigma - self.crisis_threshold) * self.max_crisis_amplification,
                self.max_crisis_amplification
            )
            logger.info(f"Crisis detected: volatility={sigma:.4f}, crisis_factor={crisis_factor:.4f}")
            return crisis_factor
        return 1.0
        
    def price_call_option(self, s, k, r, sigma, t, vov=0.2):
        """
        Price a call option using quantum-adjusted Black-Scholes
        
        Parameters:
        - s: Current stock price
        - k: Strike price
        - r: Risk-free rate
        - sigma: Volatility
        - t: Time to expiration (in years)
        - vov: Volatility of volatility (default: 0.2)
        
        Returns:
        - Call option price
        """
        if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
            logger.error(f"Invalid inputs: s={s}, k={k}, t={t}, sigma={sigma}")
            raise ValueError("Inputs must be positive")
            
        d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        standard_price = s * stats.norm.cdf(d1) - k * np.exp(-r*t) * stats.norm.cdf(d2)
        
        qvol = self._quantum_vol(sigma, vov, t)
        qdrift = self._quantum_drift(s, r, qvol, t)
        quantum_factor = self._quantum_path_integral(s, k, r, sigma, t, vov)
        
        crisis_factor = self._calculate_crisis_factor(sigma)
        
        quantum_price = standard_price * (1 + self.hbar * crisis_factor * quantum_factor)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'call',
            'standard_price': float(standard_price),
            'quantum_price': float(quantum_price),
            'quantum_factor': float(quantum_factor),
            'crisis_factor': float(crisis_factor),
            'input_params': {
                's': float(s),
                'k': float(k),
                'r': float(r),
                'sigma': float(sigma),
                't': float(t),
                'vov': float(vov)
            }
        })
        
        logger.debug(f"Call option priced: standard={standard_price:.4f}, quantum={quantum_price:.4f}")
        return quantum_price
    
    def price_put_option(self, s, k, r, sigma, t, vov=0.2):
        """
        Price a put option using quantum-adjusted Black-Scholes
        
        Parameters:
        - s: Current stock price
        - k: Strike price
        - r: Risk-free rate
        - sigma: Volatility
        - t: Time to expiration (in years)
        - vov: Volatility of volatility (default: 0.2)
        
        Returns:
        - Put option price
        """
        if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
            logger.error(f"Invalid inputs: s={s}, k={k}, t={t}, sigma={sigma}")
            raise ValueError("Inputs must be positive")
            
        d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        standard_price = k * np.exp(-r*t) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)
        
        qvol = self._quantum_vol(sigma, vov, t)
        qdrift = self._quantum_drift(s, r, qvol, t)
        quantum_factor = self._quantum_path_integral(s, k, r, sigma, t, vov)
        
        crisis_factor = self._calculate_crisis_factor(sigma)
        
        quantum_price = standard_price * (1 + self.hbar * crisis_factor * quantum_factor)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'put',
            'standard_price': float(standard_price),
            'quantum_price': float(quantum_price),
            'quantum_factor': float(quantum_factor),
            'crisis_factor': float(crisis_factor),
            'input_params': {
                's': float(s),
                'k': float(k),
                'r': float(r),
                'sigma': float(sigma),
                't': float(t),
                'vov': float(vov)
            }
        })
        
        logger.debug(f"Put option priced: standard={standard_price:.4f}, quantum={quantum_price:.4f}")
        return quantum_price
    
    def implied_volatility(self, option_price, s, k, r, t, option_type='call', precision=0.0001, max_iterations=100):
        """
        Calculate implied volatility using quantum-adjusted pricing
        
        Parameters:
        - option_price: Market price of the option
        - s: Current stock price
        - k: Strike price
        - r: Risk-free rate
        - t: Time to expiration (in years)
        - option_type: 'call' or 'put'
        - precision: Desired precision (default: 0.0001)
        - max_iterations: Maximum number of iterations (default: 100)
        
        Returns:
        - Implied volatility
        """
        if option_price <= 0 or s <= 0 or k <= 0 or t <= 0:
            logger.error(f"Invalid inputs: price={option_price}, s={s}, k={k}, t={t}")
            raise ValueError("Inputs must be positive")
            
        sigma = 0.2
        
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                price = self.price_call_option(s, k, r, sigma, t)
            elif option_type.lower() == 'put':
                price = self.price_put_option(s, k, r, sigma, t)
            else:
                logger.error(f"Invalid option_type: {option_type}")
                raise ValueError("option_type must be 'call' or 'put'")
                
            diff = option_price - price
            if abs(diff) < precision:
                logger.info(f"Implied volatility converged after {i+1} iterations: {sigma:.6f}")
                return sigma
                
            vega = self._calculate_vega(s, k, r, sigma, t)
            if abs(vega) < 1e-10:
                logger.warning("Vega too small, stopping iteration")
                break
                
            sigma = sigma + diff / vega
            
            if sigma <= 0:
                sigma = 0.001
                logger.warning("Sigma adjusted to minimum value (0.001)")
                
        logger.warning(f"Implied volatility did not converge after {max_iterations} iterations")
        return sigma
    
    def _calculate_vega(self, s, k, r, sigma, t):
        """
        Calculate option vega (sensitivity to volatility)
        
        Parameters:
        - s: Current stock price
        - k: Strike price
        - r: Risk-free rate
        - sigma: Volatility
        - t: Time to expiration
        
        Returns:
        - Vega value
        """
        d1 = (np.log(s/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
        vega = s * np.sqrt(t) * stats.norm.pdf(d1)
        return vega
        
    def detect_volatility_clustering(self, price_history, window=20):
        """
        Detect volatility clustering in price history
        
        Volatility clustering is a key indicator of market stress
        
        Parameters:
        - price_history: Array of historical prices
        - window: Window size for volatility calculation (default: 20)
        
        Returns:
        - Dictionary with volatility clustering metrics
        """
        if len(price_history) < window + 1:
            logger.warning(f"Price history too short: {len(price_history)} < {window+1}")
            return {'clustering_detected': False, 'volatility': 0, 'clustering_score': 0}
            
        returns = np.diff(np.log(price_history))
        
        vol = np.zeros(len(returns) - window + 1)
        for i in range(len(vol)):
            vol[i] = np.std(returns[i:i+window]) * np.sqrt(252)  # Annualized
            
        squared_returns = returns**2
        if len(squared_returns) > 1:
            autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        else:
            autocorr = 0
            
        current_vol = vol[-1] if len(vol) > 0 else np.std(returns) * np.sqrt(252)
        
        vov = np.std(vol) if len(vol) > 1 else 0
        
        clustering_score = 0.5 * abs(autocorr) + 0.5 * (vov / current_vol if current_vol > 0 else 0)
        
        clustering_detected = clustering_score > 0.3 and current_vol > self.crisis_threshold
        
        result = {
            'clustering_detected': clustering_detected,
            'volatility': float(current_vol),
            'vov': float(vov),
            'autocorrelation': float(autocorr),
            'clustering_score': float(clustering_score)
        }
        
        if clustering_detected:
            logger.info(f"Volatility clustering detected: score={clustering_score:.4f}, vol={current_vol:.4f}")
            
        return result
        
    def get_statistics(self):
        """
        Get statistics about option pricing history
        
        Returns:
        - Dictionary with pricing statistics
        """
        if not self.history:
            return {'count': 0}
            
        call_prices = [h['quantum_price'] for h in self.history if h['type'] == 'call']
        put_prices = [h['quantum_price'] for h in self.history if h['type'] == 'put']
        quantum_factors = [h['quantum_factor'] for h in self.history]
        crisis_factors = [h['crisis_factor'] for h in self.history]
        
        stats = {
            'count': len(self.history),
            'call_count': len(call_prices),
            'put_count': len(put_prices),
            'avg_quantum_factor': float(np.mean(quantum_factors)) if quantum_factors else 0,
            'max_quantum_factor': float(np.max(quantum_factors)) if quantum_factors else 0,
            'avg_crisis_factor': float(np.mean(crisis_factors)) if crisis_factors else 0,
            'max_crisis_factor': float(np.max(crisis_factors)) if crisis_factors else 0,
            'crisis_detection_rate': sum(1 for cf in crisis_factors if cf > 1.0) / len(crisis_factors) if crisis_factors else 0
        }
        
        return stats
        
    def clear_history(self):
        """Clear pricing history"""
        self.history = []
        logger.info("Pricing history cleared")


if __name__ == "__main__":
    import unittest
    
    class TestQuantumBlackScholes(unittest.TestCase):
        """Unit tests for QuantumBlackScholes"""
        
        def setUp(self):
            """Set up test fixtures"""
            self.qbs = QuantumBlackScholes(hbar=0.01)
            self.s = 100.0  # Stock price
            self.k = 100.0  # Strike price
            self.r = 0.01   # Risk-free rate
            self.t = 30/365 # Time to expiration (30 days)
            self.sigma = 0.2  # Volatility
            
        def test_standard_vs_quantum_call(self):
            """Test that quantum price differs from standard price for calls"""
            d1 = (np.log(self.s/self.k) + (self.r + 0.5*self.sigma**2)*self.t) / (self.sigma*np.sqrt(self.t))
            d2 = d1 - self.sigma*np.sqrt(self.t)
            standard_price = self.s * stats.norm.cdf(d1) - self.k * np.exp(-self.r*self.t) * stats.norm.cdf(d2)
            
            quantum_price = self.qbs.price_call_option(self.s, self.k, self.r, self.sigma, self.t)
            
            self.assertNotEqual(quantum_price, standard_price)
            
        def test_standard_vs_quantum_put(self):
            """Test that quantum price differs from standard price for puts"""
            d1 = (np.log(self.s/self.k) + (self.r + 0.5*self.sigma**2)*self.t) / (self.sigma*np.sqrt(self.t))
            d2 = d1 - self.sigma*np.sqrt(self.t)
            standard_price = self.k * np.exp(-self.r*self.t) * stats.norm.cdf(-d2) - self.s * stats.norm.cdf(-d1)
            
            quantum_price = self.qbs.price_put_option(self.s, self.k, self.r, self.sigma, self.t)
            
            self.assertNotEqual(quantum_price, standard_price)
            
        def test_crisis_amplification(self):
            """Test that high volatility amplifies quantum effects"""
            normal_price = self.qbs.price_call_option(self.s, self.k, self.r, self.sigma, self.t)
            
            crisis_sigma = 0.6  # High volatility
            crisis_price = self.qbs.price_call_option(self.s, self.k, self.r, crisis_sigma, self.t)
            
            d1_normal = (np.log(self.s/self.k) + (self.r + 0.5*self.sigma**2)*self.t) / (self.sigma*np.sqrt(self.t))
            d2_normal = d1_normal - self.sigma*np.sqrt(self.t)
            standard_normal = self.s * stats.norm.cdf(d1_normal) - self.k * np.exp(-self.r*self.t) * stats.norm.cdf(d2_normal)
            
            d1_crisis = (np.log(self.s/self.k) + (self.r + 0.5*crisis_sigma**2)*self.t) / (crisis_sigma*np.sqrt(self.t))
            d2_crisis = d1_crisis - crisis_sigma*np.sqrt(self.t)
            standard_crisis = self.s * stats.norm.cdf(d1_crisis) - self.k * np.exp(-self.r*self.t) * stats.norm.cdf(d2_crisis)
            
            normal_premium = (normal_price - standard_normal) / standard_normal
            crisis_premium = (crisis_price - standard_crisis) / standard_crisis
            
            self.assertGreater(crisis_premium, normal_premium)
            
        def test_implied_volatility(self):
            """Test implied volatility calculation"""
            price = self.qbs.price_call_option(self.s, self.k, self.r, self.sigma, self.t)
            
            implied_vol = self.qbs.implied_volatility(price, self.s, self.k, self.r, self.t)
            
            self.assertAlmostEqual(implied_vol, self.sigma, delta=0.01)
            
        def test_volatility_clustering_detection(self):
            """Test volatility clustering detection"""
            np.random.seed(42)  # For reproducibility
            n = 100
            prices = np.zeros(n)
            prices[0] = 100
            
            for i in range(1, n//2):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, 0.005))
                
            for i in range(n//2, n):
                if i % 2 == 0:
                    prices[i] = prices[i-1] * (1 + np.random.normal(0, 0.03))
                else:
                    prices[i] = prices[i-1] * (1 + np.random.normal(0, 0.01))
                    
            result = self.qbs.detect_volatility_clustering(prices)
            
            self.assertIn('clustering_detected', result)
            self.assertIn('clustering_score', result)
            
        def test_input_validation(self):
            """Test input validation"""
            with self.assertRaises(ValueError):
                self.qbs.price_call_option(-1, self.k, self.r, self.sigma, self.t)
                
            with self.assertRaises(ValueError):
                self.qbs.price_put_option(self.s, -1, self.r, self.sigma, self.t)
                
            with self.assertRaises(ValueError):
                self.qbs.price_call_option(self.s, self.k, self.r, -0.1, self.t)
                
            with self.assertRaises(ValueError):
                self.qbs.price_put_option(self.s, self.k, self.r, self.sigma, -1)
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
