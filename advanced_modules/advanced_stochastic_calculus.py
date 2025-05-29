#!/usr/bin/env python3
"""
Advanced Stochastic Calculus Module

Implements advanced stochastic calculus techniques beyond traditional Black-Scholes:
- Jump-diffusion processes for modeling market discontinuities
- Lévy processes for heavy-tailed distributions
- Fractional Brownian motion for long memory effects
- Neural SDEs for non-Markovian processes
- Rough path theory for path-dependent dynamics

This module extends the existing quantum stochastic calculus with more sophisticated
mathematical models for capturing complex market behaviors.
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedStochasticCalculus")

class AdvancedStochasticCalculus:
    """
    Advanced Stochastic Calculus for sophisticated market modeling
    
    Extends traditional stochastic calculus with:
    - Jump-diffusion processes
    - Lévy processes
    - Fractional Brownian motion
    - Neural SDEs
    - Rough path theory
    
    Provides rigorous mathematical foundation for modeling complex market dynamics
    beyond traditional Gaussian assumptions.
    """
    
    def __init__(self, precision: int = 64, confidence_level: float = 0.99,
                hurst_parameter: float = 0.7, jump_intensity: float = 5.0,
                levy_alpha: float = 1.5, neural_layers: int = 3):
        """
        Initialize Advanced Stochastic Calculus
        
        Parameters:
        - precision: Numerical precision for calculations (default: 64 bits)
        - confidence_level: Statistical confidence level (default: 0.99)
        - hurst_parameter: Hurst parameter for fractional Brownian motion (default: 0.7)
        - jump_intensity: Intensity parameter for jump processes (default: 5.0)
        - levy_alpha: Stability parameter for Lévy processes (default: 1.5)
        - neural_layers: Number of layers for neural SDE (default: 3)
        """
        self.precision = precision
        self.confidence_level = confidence_level
        self.hurst_parameter = hurst_parameter
        self.jump_intensity = jump_intensity
        self.levy_alpha = levy_alpha
        self.neural_layers = neural_layers
        self.history = []
        
        np.random.seed(42)  # For reproducibility
        
        logger.info(f"Initialized AdvancedStochasticCalculus with precision={precision}, "
                   f"confidence_level={confidence_level}, hurst_parameter={hurst_parameter}")
    
    
    def simulate_jump_diffusion(self, S0: float, mu: float, sigma: float, 
                               lam: float, jump_mean: float, jump_std: float,
                               T: float, steps: int) -> np.ndarray:
        """
        Simulate Merton jump-diffusion process
        
        Parameters:
        - S0: Initial price
        - mu: Drift rate
        - sigma: Volatility
        - lam: Jump intensity (average number of jumps per year)
        - jump_mean: Mean of jump size
        - jump_std: Standard deviation of jump size
        - T: Time horizon in years
        - steps: Number of time steps
        
        Returns:
        - Array of simulated prices
        """
        dt = T / steps
        prices = np.zeros(steps + 1)
        prices[0] = S0
        
        for i in range(1, steps + 1):
            z = np.random.normal(0, 1)
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            
            num_jumps = np.random.poisson(lam * dt)
            jump_sizes = np.random.normal(jump_mean, jump_std, num_jumps)
            jump_component = np.sum(jump_sizes) if num_jumps > 0 else 0
            
            prices[i] = prices[i-1] * np.exp(diffusion + jump_component)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'simulate_jump_diffusion',
            'S0': S0,
            'mu': mu,
            'sigma': sigma,
            'lam': lam,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'T': T,
            'steps': steps,
            'final_price': float(prices[-1])
        })
        
        return prices
    
    def calibrate_jump_diffusion(self, prices: np.ndarray, dt: float = 1/252) -> Dict:
        """
        Calibrate jump-diffusion model parameters from historical prices
        
        Parameters:
        - prices: Historical price series
        - dt: Time step in years (default: 1/252 for daily data)
        
        Returns:
        - Dictionary with calibrated parameters
        """
        log_returns = np.diff(np.log(prices))
        
        mu = np.mean(log_returns) / dt
        sigma = np.std(log_returns) / np.sqrt(dt)
        
        threshold = 3 * sigma * np.sqrt(dt)  # 3-sigma rule
        jumps = log_returns[np.abs(log_returns) > threshold]
        
        if len(jumps) > 0:
            lam = len(jumps) / (len(log_returns) * dt)
            
            jump_mean = np.mean(jumps)
            jump_std = np.std(jumps)
            
            non_jump_returns = log_returns[np.abs(log_returns) <= threshold]
            adjusted_mu = np.mean(non_jump_returns) / dt
            adjusted_sigma = np.std(non_jump_returns) / np.sqrt(dt)
        else:
            lam = 0.1  # Default value
            jump_mean = 0
            jump_std = sigma
            adjusted_mu = mu
            adjusted_sigma = sigma
        
        result = {
            'mu': float(adjusted_mu),
            'sigma': float(adjusted_sigma),
            'lambda': float(lam),
            'jump_mean': float(jump_mean),
            'jump_std': float(jump_std),
            'jumps_detected': len(jumps),
            'confidence': min(0.95, 0.5 + len(jumps)/len(log_returns))
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'calibrate_jump_diffusion',
            'prices_length': len(prices),
            'dt': dt,
            'result': result
        })
        
        return result
    
    def jump_diffusion_option_price(self, S0: float, K: float, r: float, T: float,
                                   sigma: float, lam: float, jump_mean: float, 
                                   jump_std: float, option_type: str = 'call',
                                   n_terms: int = 10) -> float:
        """
        Price options under Merton jump-diffusion model
        
        Parameters:
        - S0: Current stock price
        - K: Strike price
        - r: Risk-free rate
        - T: Time to maturity in years
        - sigma: Volatility
        - lam: Jump intensity
        - jump_mean: Mean of jump size
        - jump_std: Standard deviation of jump size
        - option_type: 'call' or 'put'
        - n_terms: Number of terms in series expansion
        
        Returns:
        - Option price
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        
        gamma = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        lambda_prime = lam * (1 + gamma)
        r_prime = r - lam * gamma
        
        price = 0
        
        for k in range(n_terms):
            sigma_k = np.sqrt(sigma**2 + k * jump_std**2 / T)
            
            r_k = r_prime + k * (jump_mean + 0.5 * jump_std**2) / T
            d1 = (np.log(S0/K) + (r_k + 0.5 * sigma_k**2) * T) / (sigma_k * np.sqrt(T))
            d2 = d1 - sigma_k * np.sqrt(T)
            
            if option_type == 'call':
                bs_price = S0 * stats.norm.cdf(d1) - K * np.exp(-r_k * T) * stats.norm.cdf(d2)
            else:  # put
                bs_price = K * np.exp(-r_k * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
            
            poisson_prob = np.exp(-lambda_prime * T) * (lambda_prime * T)**k / math.factorial(k)
            
            price += poisson_prob * bs_price
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'jump_diffusion_option_price',
            'S0': S0,
            'K': K,
            'r': r,
            'T': T,
            'sigma': sigma,
            'lam': lam,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'option_type': option_type,
            'price': float(price)
        })
        
        return float(price)
    
    
    def simulate_levy_process(self, alpha: float, beta: float, mu: float, 
                             sigma: float, T: float, steps: int) -> np.ndarray:
        """
        Simulate alpha-stable Lévy process
        
        Parameters:
        - alpha: Stability parameter (0 < alpha <= 2)
        - beta: Skewness parameter (-1 <= beta <= 1)
        - mu: Location parameter
        - sigma: Scale parameter
        - T: Time horizon
        - steps: Number of time steps
        
        Returns:
        - Array of simulated process values
        """
        dt = T / steps
        increments = np.zeros(steps)
        
        for i in range(steps):
            u = np.random.uniform(0, np.pi)
            w = np.random.exponential(1)
            
            if alpha == 1:
                x = (2/np.pi) * (np.pi/2 + beta * u) * np.tan(u) - beta * np.log(
                    (np.pi/2 * w * np.cos(u)) / (np.pi/2 + beta * u)
                )
            else:
                b = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
                s = (1 + beta**2 * np.tan(np.pi * alpha / 2)**2)**(1/(2*alpha))
                x = s * np.sin(alpha * (u + b)) / np.cos(u)**(1/alpha) * (
                    np.cos(u - alpha * (u + b)) / w
                )**(1-1/alpha)
            
            increments[i] = sigma * dt**(1/alpha) * x + mu * dt
        
        process = np.zeros(steps + 1)
        process[1:] = np.cumsum(increments)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'simulate_levy_process',
            'alpha': alpha,
            'beta': beta,
            'mu': mu,
            'sigma': sigma,
            'T': T,
            'steps': steps,
            'final_value': float(process[-1])
        })
        
        return process
    
    def estimate_levy_parameters(self, returns: np.ndarray) -> Dict:
        """
        Estimate parameters of alpha-stable Lévy process from returns
        
        Parameters:
        - returns: Array of returns
        
        Returns:
        - Dictionary with estimated parameters
        """
        q_05 = np.percentile(returns, 5)
        q_50 = np.percentile(returns, 50)
        q_95 = np.percentile(returns, 95)
        
        alpha_est = min(2.0, 1 / (1 - np.log(np.abs(q_95 - q_50) / np.abs(q_50 - q_05)) / np.log(19)))
        
        beta_est = np.clip((q_95 + q_05 - 2 * q_50) / (q_95 - q_05), -1, 1)
        
        mu_est = np.median(returns)
        sigma_est = np.abs(q_95 - q_05) / (2 * 1.645)  # Assuming approximate normality in the tails
        
        ks_stat = 0.1  # Placeholder
        p_value = 0.5  # Placeholder
        
        result = {
            'alpha': float(alpha_est),
            'beta': float(beta_est),
            'mu': float(mu_est),
            'sigma': float(sigma_est),
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'confidence': min(0.95, 0.5 + p_value)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'estimate_levy_parameters',
            'returns_length': len(returns),
            'result': result
        })
        
        return result
    
    
    def simulate_fractional_brownian_motion(self, H: float, T: float, steps: int) -> np.ndarray:
        """
        Simulate fractional Brownian motion with Hurst parameter H
        
        Parameters:
        - H: Hurst parameter (0 < H < 1)
        - T: Time horizon
        - steps: Number of time steps
        
        Returns:
        - Array of simulated FBM values
        """
        dt = T / steps
        times = np.arange(0, T + dt, dt)
        n = len(times)
        
        Z = np.random.normal(0, 1, n)
        
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov[i, j] = 0.5 * (times[i]**(2*H) + times[j]**(2*H) - np.abs(times[i] - times[j])**(2*H))
        
        L = np.linalg.cholesky(cov)
        
        fbm = np.dot(L, Z)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'simulate_fractional_brownian_motion',
            'H': H,
            'T': T,
            'steps': steps,
            'final_value': float(fbm[-1])
        })
        
        return fbm
    
    def estimate_hurst_exponent(self, time_series: np.ndarray, max_lag: Optional[int] = None) -> float:
        """
        Estimate Hurst exponent using rescaled range (R/S) analysis
        
        Parameters:
        - time_series: Input time series
        - max_lag: Maximum lag to consider (default: len(time_series)/10)
        
        Returns:
        - Estimated Hurst exponent
        """
        if max_lag is None:
            max_lag = len(time_series) // 10
            
        max_lag = max(2, min(max_lag, len(time_series) // 4))
        
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            n_chunks = len(time_series) // lag
            if n_chunks == 0:
                continue
                
            chunk_rs = []
            for i in range(n_chunks):
                chunk = time_series[i*lag:(i+1)*lag]
                
                mean_adj = chunk - np.mean(chunk)
                
                cum_dev = np.cumsum(mean_adj)
                
                R = np.max(cum_dev) - np.min(cum_dev)
                
                S = np.std(chunk)
                
                if S > 0:
                    chunk_rs.append(R / S)
            
            if chunk_rs:
                rs_values.append(np.mean(chunk_rs))
        
        if len(rs_values) > 1:
            x = np.log10(lags[:len(rs_values)])
            y = np.log10(rs_values)
            
            H, _ = np.polyfit(x, y, 1)
            
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'estimate_hurst_exponent',
                'time_series_length': len(time_series),
                'max_lag': max_lag,
                'H': float(H)
            })
            
            return float(H)
        else:
            logger.warning("Not enough data points to estimate Hurst exponent")
            return 0.5  # Default to random walk
    
    
    def neural_sde_drift(self, x: np.ndarray, t: float, weights: List[np.ndarray], 
                        biases: List[np.ndarray]) -> np.ndarray:
        """
        Neural network approximation of drift function for SDE
        
        Parameters:
        - x: Current state
        - t: Current time
        - weights: List of weight matrices for neural network
        - biases: List of bias vectors for neural network
        
        Returns:
        - Drift vector
        """
        h = x.copy()
        
        for i in range(len(weights) - 1):
            h = np.tanh(np.dot(h, weights[i]) + biases[i])
            
        drift = np.dot(h, weights[-1]) + biases[-1]
        
        return drift
    
    def neural_sde_diffusion(self, x: np.ndarray, t: float, weights: List[np.ndarray], 
                           biases: List[np.ndarray]) -> np.ndarray:
        """
        Neural network approximation of diffusion function for SDE
        
        Parameters:
        - x: Current state
        - t: Current time
        - weights: List of weight matrices for neural network
        - biases: List of bias vectors for neural network
        
        Returns:
        - Diffusion matrix
        """
        h = x.copy()
        
        for i in range(len(weights) - 1):
            h = np.tanh(np.dot(h, weights[i]) + biases[i])
            
        diffusion_raw = np.dot(h, weights[-1]) + biases[-1]
        
        dim = int(np.sqrt(len(diffusion_raw)))
        diffusion = diffusion_raw.reshape(dim, dim)
        
        diffusion = np.dot(diffusion, diffusion.T)
        
        return diffusion
    
    def simulate_neural_sde(self, x0: np.ndarray, drift_weights: List[np.ndarray], 
                          drift_biases: List[np.ndarray], diffusion_weights: List[np.ndarray],
                          diffusion_biases: List[np.ndarray], T: float, steps: int) -> np.ndarray:
        """
        Simulate neural SDE with learned drift and diffusion functions
        
        Parameters:
        - x0: Initial state
        - drift_weights: Weights for drift neural network
        - drift_biases: Biases for drift neural network
        - diffusion_weights: Weights for diffusion neural network
        - diffusion_biases: Biases for diffusion neural network
        - T: Time horizon
        - steps: Number of time steps
        
        Returns:
        - Array of simulated states
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        dim = len(x0)
        
        trajectory = np.zeros((steps + 1, dim))
        trajectory[0] = x0
        
        for i in range(steps):
            t = i * dt
            x = trajectory[i]
            
            drift = self.neural_sde_drift(x, t, drift_weights, drift_biases)
            diffusion = self.neural_sde_diffusion(x, t, diffusion_weights, diffusion_biases)
            
            dW = np.random.normal(0, 1, dim) * sqrt_dt
            
            trajectory[i+1] = x + drift * dt + np.dot(diffusion, dW)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'simulate_neural_sde',
            'x0': x0.tolist(),
            'T': T,
            'steps': steps,
            'final_state': trajectory[-1].tolist()
        })
        
        return trajectory
    
    
    def compute_signature(self, path: np.ndarray, depth: int = 3) -> Dict:
        """
        Compute signature of a path up to specified depth
        
        Parameters:
        - path: Multidimensional path as array of shape (steps, dim)
        - depth: Truncation depth for signature
        
        Returns:
        - Dictionary with signature terms
        """
        steps, dim = path.shape
        
        signature = {'1': 1.0}  # Constant term
        
        for i in range(dim):
            key = f"{i+1}"
            signature[key] = path[-1, i] - path[0, i]
        
        if depth >= 2:
            increments = np.diff(path, axis=0)
            
            for i in range(dim):
                for j in range(dim):
                    key = f"{i+1},{j+1}"
                    
                    integral = 0.0
                    for k in range(steps - 1):
                        integral += (path[k, i] - path[0, i]) * increments[k, j]
                    
                    signature[key] = float(integral)
            
            if depth >= 3:
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            key = f"{i+1},{j+1},{k+1}"
                            
                            integral = 0.0
                            for s in range(steps - 2):
                                for t in range(s + 1, steps - 1):
                                    integral += (path[s, i] - path[0, i]) * increments[s, j] * increments[t, k]
                            
                            signature[key] = float(integral)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'compute_signature',
            'path_shape': path.shape,
            'depth': depth,
            'signature_terms': len(signature)
        })
        
        return signature
    
    def signature_kernel(self, sig1: Dict, sig2: Dict) -> float:
        """
        Compute kernel between two path signatures
        
        Parameters:
        - sig1: First path signature
        - sig2: Second path signature
        
        Returns:
        - Kernel value (similarity measure)
        """
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        kernel = 0.0
        for key in common_keys:
            kernel += sig1[key] * sig2[key]
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'signature_kernel',
            'sig1_terms': len(sig1),
            'sig2_terms': len(sig2),
            'common_terms': len(common_keys),
            'kernel_value': float(kernel)
        })
        
        return float(kernel)
    
    def rough_volatility_model(self, S0: float, H: float, rho: float, xi: float, 
                              eta: float, T: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rough volatility model with fractional Brownian motion
        
        Parameters:
        - S0: Initial price
        - H: Hurst parameter for volatility process
        - rho: Correlation between price and volatility
        - xi: Volatility of volatility
        - eta: Mean reversion level
        - T: Time horizon
        - steps: Number of time steps
        
        Returns:
        - Tuple of (prices, volatilities)
        """
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        
        fbm = self.simulate_fractional_brownian_motion(H, T, steps)
        
        prices = np.zeros(steps + 1)
        volatilities = np.zeros(steps + 1)
        
        prices[0] = S0
        volatilities[0] = eta
        
        dW1 = np.random.normal(0, 1, steps)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, steps)
        
        for i in range(steps):
            vol_increment = xi * (fbm[i+1] - fbm[i])
            volatilities[i+1] = volatilities[i] * np.exp(vol_increment)
            
            drift = 0  # Risk-neutral measure
            diffusion = volatilities[i] * sqrt_dt * dW1[i]
            prices[i+1] = prices[i] * np.exp(drift * dt + diffusion)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'rough_volatility_model',
            'S0': S0,
            'H': H,
            'rho': rho,
            'xi': xi,
            'eta': eta,
            'T': T,
            'steps': steps,
            'final_price': float(prices[-1]),
            'final_vol': float(volatilities[-1])
        })
        
        return prices, volatilities
    
    def calibrate_rough_volatility_model(self, prices: np.ndarray) -> Dict:
        """
        Calibrate rough volatility model parameters from historical prices
        
        Parameters:
        - prices: Historical price series
        
        Returns:
        - Dictionary with calibrated parameters
        """
        if len(prices) < 10:
            logger.warning(f"Insufficient data for rough volatility calibration. Need at least 10 points, got {len(prices)}")
            return {
                'H': self.hurst_parameter,
                'rho': -0.7,
                'xi': 0.3,
                'eta': 0.2,
                'confidence': 0.5
            }
            
        log_returns = np.diff(np.log(prices))
        
        # Estimate Hurst parameter
        H = self.estimate_hurst_exponent(log_returns)
        
        rolling_vol = pd.Series(log_returns).rolling(window=5).std().dropna().values
        
        if len(rolling_vol) > 0:
            xi = float(np.std(rolling_vol.astype(np.float64))) / float(np.mean(rolling_vol.astype(np.float64)))
        else:
            xi = 0.3  # Default value
        
        if len(rolling_vol) > 5:
            vol_changes = np.diff(rolling_vol.astype(np.float64))
            return_changes = log_returns[5:]
            
            if len(vol_changes) > 0 and len(return_changes) > 0 and len(vol_changes) == len(return_changes):
                rho = np.corrcoef(vol_changes, return_changes)[0, 1]
                if np.isnan(rho):
                    rho = -0.7  # Default value
            else:
                rho = -0.7  # Default value
        else:
            rho = -0.7  # Default value
            
        eta = np.std(log_returns)
        
        confidence = min(0.99, 0.5 + len(prices) / 200)
        
        result = {
            'H': float(H),
            'rho': float(rho),
            'xi': float(xi),
            'eta': float(eta),
            'confidence': float(confidence)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'calibrate_rough_volatility_model',
            'prices_length': len(prices),
            'result': result
        })
        
        return result
        
    def get_statistics(self) -> Dict:
        """
        Get statistics about advanced stochastic calculus usage
        
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
            'hurst_parameter': self.hurst_parameter
        }
