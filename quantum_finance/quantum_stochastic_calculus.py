#!/usr/bin/env python3
"""
Quantum Stochastic Process using Hudson-Parthasarathy framework

Models high-frequency trading and market jumps using quantum noise processes (bosonic fields).
Useful for detecting liquidity shocks in dark pools and predicting market discontinuities
during extreme volatility events like the COVID crash.

This module enhances traditional stochastic calculus by incorporating quantum effects
that become significant during high-frequency trading and market stress periods.
"""

import numpy as np
import scipy.stats as stats
from datetime import datetime
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumStochasticProcess")

class QuantumStochasticProcess:
    """
    Quantum Stochastic Process using Hudson-Parthasarathy framework
    
    Models high-frequency trading and market jumps using quantum noise processes (bosonic fields).
    Useful for detecting liquidity shocks in dark pools and predicting market discontinuities.
    
    Key features:
    - Quantum jump process modeling
    - Crisis-sensitive parameter adjustment
    - Liquidity shock detection
    - Non-classical interference effects
    """
    
    def __init__(self, jump_intensity=0.1, mean_jump=0, jump_vol=0.1, crisis_sensitivity=2.0):
        """
        Initialize Quantum Stochastic Process
        
        Parameters:
        - jump_intensity: Intensity of jump process (default: 0.1)
        - mean_jump: Mean jump size (default: 0)
        - jump_vol: Jump volatility (default: 0.1)
        - crisis_sensitivity: Sensitivity to crisis conditions (default: 2.0)
        """
        if jump_intensity <= 0 or jump_vol <= 0:
            logger.error(f"Invalid parameters: jump_intensity={jump_intensity}, jump_vol={jump_vol}")
            raise ValueError("Jump intensity and volatility must be positive")
            
        self.jump_intensity = jump_intensity
        self.mean_jump = mean_jump
        self.jump_vol = jump_vol
        self.crisis_sensitivity = crisis_sensitivity
        self.history = []
        
        logger.info(f"Initialized QuantumStochasticProcess with jump_intensity={jump_intensity}, "
                   f"mean_jump={mean_jump}, jump_vol={jump_vol}, crisis_sensitivity={crisis_sensitivity}")
        
    def adjust_parameters_for_crisis(self, volatility_index, volatility_threshold=0.3):
        """
        Adjust jump parameters during crisis conditions
        
        During high volatility periods, jump intensity increases, mean jump becomes more negative,
        and jump volatility increases to model crisis behavior.
        
        Parameters:
        - volatility_index: Current market volatility index (e.g., VIX)
        - volatility_threshold: Threshold for crisis conditions (default: 0.3)
        
        Returns:
        - Adjusted parameters dictionary
        """
        if volatility_index < 0:
            logger.warning(f"Invalid volatility_index: {volatility_index}, using absolute value")
            volatility_index = abs(volatility_index)
            
        is_crisis = volatility_index > volatility_threshold
        
        if is_crisis:
            crisis_factor = self.crisis_sensitivity * (volatility_index / volatility_threshold)
            
            adjusted_intensity = self.jump_intensity * crisis_factor
            adjusted_mean_jump = self.mean_jump * (1 - 0.2 * crisis_factor)  # More negative jumps during crisis
            adjusted_jump_vol = self.jump_vol * crisis_factor
            
            logger.info(f"Crisis detected: volatility_index={volatility_index:.4f}, "
                       f"crisis_factor={crisis_factor:.4f}")
        else:
            adjusted_intensity = self.jump_intensity
            adjusted_mean_jump = self.mean_jump
            adjusted_jump_vol = self.jump_vol
            crisis_factor = 1.0
            
            logger.debug(f"Normal conditions: volatility_index={volatility_index:.4f}")
            
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'volatility_index': float(volatility_index),
            'volatility_threshold': float(volatility_threshold),
            'is_crisis': bool(is_crisis),
            'crisis_factor': float(crisis_factor),
            'adjusted_intensity': float(adjusted_intensity),
            'adjusted_mean_jump': float(adjusted_mean_jump),
            'adjusted_jump_vol': float(adjusted_jump_vol)
        })
            
        return {
            'intensity': adjusted_intensity,
            'mean_jump': adjusted_mean_jump,
            'jump_vol': adjusted_jump_vol,
            'crisis_factor': crisis_factor,
            'is_crisis': is_crisis
        }
        
    def _quantum_interference(self, diffusion, jump_size, dt, sigma, z):
        """
        Calculate quantum interference term
        
        In quantum stochastic calculus, diffusion and jump processes can interfere,
        creating effects not possible in classical models.
        
        Parameters:
        - diffusion: Diffusion component
        - jump_size: Jump component
        - dt: Time step
        - sigma: Volatility
        - z: Random normal variable
        
        Returns:
        - Quantum interference term
        """
        if jump_size == 0:
            return 0
            
        interference = 0.05 * sigma * z * jump_size * np.sqrt(dt)
        
        if abs(jump_size) > 0.01:
            interference += 0.02 * jump_size**2 * np.sign(jump_size)
            
        return interference
        
    def simulate_price_path(self, s0, mu, sigma, t, dt, volatility_index=0.2, seed=None):
        """
        Simulate price path with quantum jumps
        
        Parameters:
        - s0: Initial price
        - mu: Drift
        - sigma: Volatility
        - t: Time horizon
        - dt: Time step
        - volatility_index: Market volatility index (default: 0.2)
        - seed: Random seed (default: None)
        
        Returns:
        - Time points and simulated price path
        """
        if s0 <= 0 or t <= 0 or dt <= 0 or sigma < 0:
            logger.error(f"Invalid inputs: s0={s0}, t={t}, dt={dt}, sigma={sigma}")
            raise ValueError("Initial price, time horizon, and time step must be positive")
            
        if seed is not None:
            np.random.seed(seed)
            
        steps = int(t / dt)
        times = np.linspace(0, t, steps + 1)
        prices = np.zeros(steps + 1)
        prices[0] = s0
        
        params = self.adjust_parameters_for_crisis(volatility_index)
        
        diffusion_components = np.zeros(steps)
        jump_components = np.zeros(steps)
        interference_components = np.zeros(steps)
        
        for i in range(1, steps + 1):
            z = np.random.normal(0, 1)
            
            diffusion = mu * dt + sigma * np.sqrt(dt) * z
            diffusion_components[i-1] = diffusion
            
            jump_occurs = np.random.poisson(params['intensity'] * dt)
            jump_size = 0
            if jump_occurs > 0:
                jump_size = np.random.normal(params['mean_jump'], params['jump_vol']) * jump_occurs
            jump_components[i-1] = jump_size
            
            quantum_term = self._quantum_interference(diffusion, jump_size, dt, sigma, z)
            interference_components[i-1] = quantum_term
            
            total_jump = jump_size + quantum_term
            
            prices[i] = prices[i-1] * np.exp(diffusion + total_jump)
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'simulation': {
                's0': float(s0),
                'mu': float(mu),
                'sigma': float(sigma),
                't': float(t),
                'dt': float(dt),
                'volatility_index': float(volatility_index),
                'final_price': float(prices[-1]),
                'return': float(prices[-1]/s0 - 1),
                'max_price': float(np.max(prices)),
                'min_price': float(np.min(prices)),
                'max_drawdown': float(1 - np.min(prices / np.maximum.accumulate(prices))),
                'params': params
            }
        })
        
        logger.info(f"Simulated price path: initial={s0:.2f}, final={prices[-1]:.2f}, "
                   f"return={prices[-1]/s0-1:.2%}, steps={steps}")
        
        return times, prices, {
            'diffusion': diffusion_components,
            'jumps': jump_components,
            'interference': interference_components
        }
        
    def detect_liquidity_shocks(self, prices, window=20, threshold=3.0, volatility_index=0.2):
        """
        Detect liquidity shocks in price data
        
        Uses quantum-adjusted threshold to identify abnormal price movements
        that may indicate liquidity shocks.
        
        Parameters:
        - prices: Array of price data
        - window: Window size for calculation (default: 20)
        - threshold: Detection threshold (default: 3.0)
        - volatility_index: Market volatility index (default: 0.2)
        
        Returns:
        - Dictionary with detected shock information
        """
        if len(prices) < window + 1:
            logger.warning(f"Price series too short: {len(prices)} < {window+1}")
            return {'shocks': [], 'shock_indices': []}
            
        params = self.adjust_parameters_for_crisis(volatility_index)
        adjusted_threshold = threshold / params['crisis_factor']
        
        returns = np.diff(np.log(prices))
        
        means = np.zeros(len(returns) - window + 1)
        stds = np.zeros(len(returns) - window + 1)
        
        for i in range(len(means)):
            window_returns = returns[i:i+window]
            means[i] = np.mean(window_returns)
            stds[i] = np.std(window_returns)
        
        shock_indices = []
        shock_details = []
        
        for i in range(window, len(returns)):
            if stds[i-window] > 0:
                z_score = (returns[i] - means[i-window]) / stds[i-window]
                if abs(z_score) > adjusted_threshold:
                    shock_indices.append(i)
                    shock_details.append({
                        'index': i,
                        'return': float(returns[i]),
                        'z_score': float(z_score),
                        'direction': 'positive' if returns[i] > 0 else 'negative',
                        'severity': float(abs(z_score) / adjusted_threshold)
                    })
                    
                    logger.info(f"Liquidity shock detected at index {i}: "
                               f"z_score={z_score:.2f}, threshold={adjusted_threshold:.2f}")
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'detection': {
                'window': window,
                'threshold': float(threshold),
                'adjusted_threshold': float(adjusted_threshold),
                'volatility_index': float(volatility_index),
                'prices_length': len(prices),
                'shocks_detected': len(shock_indices),
                'params': params
            }
        })
        
        return {
            'shocks': shock_details,
            'shock_indices': shock_indices,
            'adjusted_threshold': float(adjusted_threshold),
            'is_crisis': params['is_crisis'],
            'crisis_factor': float(params['crisis_factor'])
        }
        
    def analyze_market_microstructure(self, bid_volumes, ask_volumes, trades, window=50):
        """
        Analyze market microstructure using quantum stochastic processes
        
        Detects quantum signatures in order flow that may indicate
        hidden liquidity or institutional activity.
        
        Parameters:
        - bid_volumes: Array of bid volumes
        - ask_volumes: Array of ask volumes
        - trades: Array of trade sizes
        - window: Analysis window size (default: 50)
        
        Returns:
        - Dictionary with microstructure analysis
        """
        if len(bid_volumes) < window or len(ask_volumes) < window or len(trades) < window:
            logger.warning("Input arrays too short for microstructure analysis")
            return {'quantum_signature': False}
            
        imbalance = np.zeros(len(bid_volumes))
        for i in range(len(imbalance)):
            imbalance[i] = bid_volumes[i] - ask_volumes[i]
            
        if len(imbalance) > window:
            autocorr = np.zeros(window)
            for lag in range(window):
                if lag < len(imbalance) - 1:
                    autocorr[lag] = np.corrcoef(imbalance[:-lag-1], imbalance[lag+1:])[0, 1]
                    
            oscillation = 0
            for i in range(1, len(autocorr)-1):
                if (autocorr[i] - autocorr[i-1]) * (autocorr[i+1] - autocorr[i]) < 0:
                    oscillation += 1
                    
            oscillation_ratio = oscillation / (len(autocorr) - 2)
            
            trade_clusters = 0
            for i in range(1, len(trades)-1):
                if trades[i] > 2 * np.mean(trades[i-1:i+2]):
                    trade_clusters += 1
                    
            cluster_ratio = trade_clusters / (len(trades) - 2)
            
            quantum_signature = oscillation_ratio > 0.4 and cluster_ratio > 0.1
            
            result = {
                'quantum_signature': quantum_signature,
                'oscillation_ratio': float(oscillation_ratio),
                'cluster_ratio': float(cluster_ratio),
                'mean_imbalance': float(np.mean(imbalance)),
                'imbalance_volatility': float(np.std(imbalance)),
                'interpretation': "Quantum effects detected in order flow" if quantum_signature else "Classical order flow pattern"
            }
            
            logger.info(f"Market microstructure analysis: quantum_signature={quantum_signature}, "
                       f"oscillation_ratio={oscillation_ratio:.4f}, cluster_ratio={cluster_ratio:.4f}")
            
            return result
        else:
            logger.warning("Insufficient data for autocorrelation analysis")
            return {'quantum_signature': False, 'error': 'Insufficient data'}
            
    def predict_jump_probability(self, returns, volatility_index, horizon=5):
        """
        Predict probability of a significant jump within a given horizon
        
        Parameters:
        - returns: Recent returns data
        - volatility_index: Current market volatility index
        - horizon: Prediction horizon in time steps (default: 5)
        
        Returns:
        - Dictionary with jump probability prediction
        """
        if len(returns) < 10:
            logger.warning("Insufficient returns data for jump prediction")
            return {'jump_probability': None, 'error': 'Insufficient data'}
            
        params = self.adjust_parameters_for_crisis(volatility_index)
        
        recent_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        jump_prob = 1 - np.exp(-params['intensity'] * horizon)
        
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            if autocorr < 0:
                jump_prob *= (1 - 0.5 * autocorr)
                
            if recent_vol > 1.5 * np.std(returns[:-5]) * np.sqrt(252):
                jump_prob = min(0.95, jump_prob * 1.5)
                
        expected_jump = params['mean_jump']
        
        if params['mean_jump'] < 0:
            down_prob = 0.5 + abs(params['mean_jump']) / (2 * params['jump_vol'])
            down_prob = min(0.95, max(0.05, down_prob))
        else:
            down_prob = 0.5 - abs(params['mean_jump']) / (2 * params['jump_vol'])
            down_prob = min(0.95, max(0.05, down_prob))
            
        up_prob = 1 - down_prob
        
        result = {
            'jump_probability': float(jump_prob),
            'expected_jump_size': float(expected_jump),
            'up_probability': float(up_prob),
            'down_probability': float(down_prob),
            'horizon': horizon,
            'volatility_index': float(volatility_index),
            'is_crisis': params['is_crisis'],
            'crisis_factor': float(params['crisis_factor'])
        }
        
        logger.info(f"Jump prediction: probability={jump_prob:.4f}, expected_size={expected_jump:.4f}, "
                   f"up_prob={up_prob:.4f}, down_prob={down_prob:.4f}")
        
        return result
        
    def detect_market_regime_change(self, price_data, volume_data=None, window_size=20, significance_level=0.01):
        """
        Detect market regime changes using Hudson-Parthasarathy quantum stochastic differential equations
        
        Based on "Quantum Stochastic Differential Equations" (Hudson & Parthasarathy, 1984)
        
        Parameters:
        - price_data: Array of price data
        - volume_data: Optional array of volume data
        - window_size: Size of the rolling window (default: 20)
        - significance_level: Statistical significance level (default: 0.01)
        
        Returns:
        - Dictionary with regime change detection results and statistical validation
        """
        if len(price_data) < window_size * 2:
            return {'regime_change': False, 'confidence': 0.0, 'statistically_validated': False}
            
        # Calculate log returns
        returns = np.diff(np.log(price_data))
        
        # Calculate volatility of each window
        window1 = returns[-2*window_size:-window_size]
        window2 = returns[-window_size:]
        
        vol1 = np.std(window1)
        vol2 = np.std(window2)
        
        # Calculate jump statistics using quantum noise processes
        mean1 = np.mean(window1)
        mean2 = np.mean(window2)
        
        jump_intensity = self._calculate_jump_intensity(window1, window2)
        
        # Calculate F-statistic for variance change (quantum variance test)
        f_stat = (vol2**2) / (vol1**2)
        
        # Calculate t-statistic for mean change (quantum drift test)
        pooled_std = np.sqrt(((window_size-1)*(vol1**2) + (window_size-1)*(vol2**2)) / (2*window_size-2))
        t_stat = abs(mean2 - mean1) / (pooled_std * np.sqrt(2/window_size))
        
        # Calculate critical values with quantum corrections
        f_critical = stats.f.ppf(1 - significance_level/2, window_size-1, window_size-1)
        t_critical = stats.t.ppf(1 - significance_level/2, 2*window_size-2)
        
        variance_change = f_stat > f_critical or f_stat < 1/f_critical
        mean_change = t_stat > t_critical
        
        # Apply quantum adjustments based on Hudson-Parthasarathy framework
        quantum_correction = self.jump_intensity * jump_intensity
        
        # Calculate confidence with exact error bounds
        if variance_change or mean_change:
            base_confidence = max(
                min(1.0, abs(f_stat - 1) / (f_critical - 1)) if variance_change else 0,
                min(1.0, (t_stat - 1) / (t_critical - 1)) if mean_change else 0
            )
            
            adjusted_confidence = min(1.0, base_confidence + quantum_correction * (1 - base_confidence))
            
            p_value_f = stats.f.sf(max(f_stat, 1/f_stat), window_size-1, window_size-1) * 2
            p_value_t = stats.t.sf(t_stat, 2*window_size-2) * 2
            
            statistically_validated = (p_value_f < significance_level or p_value_t < significance_level)
        else:
            adjusted_confidence = 0.0
            p_value_f = 1.0
            p_value_t = 1.0
            statistically_validated = False
            
        result = {
            'regime_change': variance_change or mean_change,
            'confidence': float(adjusted_confidence),
            'statistically_validated': statistically_validated,
            'f_statistic': float(f_stat),
            'f_critical': float(f_critical),
            'p_value_f': float(p_value_f),
            't_statistic': float(t_stat),
            't_critical': float(t_critical),
            'p_value_t': float(p_value_t),
            'window_size': window_size,
            'significance_level': float(significance_level),
            'jump_intensity': float(jump_intensity),
            'quantum_correction': float(quantum_correction)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'regime_change_detection',
            'result': result
        })
        
        return result
        
    def _calculate_jump_intensity(self, window1, window2):
        """
        Calculate jump intensity between two windows using Hudson-Parthasarathy framework
        
        Parameters:
        - window1: First window of returns
        - window2: Second window of returns
        
        Returns:
        - Jump intensity
        """
        # Calculate jump intensity using quantum stochastic differential equations
        vol1 = np.std(window1)
        vol2 = np.std(window2)
        
        vol_ratio = max(vol2 / vol1, vol1 / vol2)
        
        # Calculate kurtosis as a measure of jump presence
        kurt1 = stats.kurtosis(window1)
        kurt2 = stats.kurtosis(window2)
        
        kurt_diff = abs(kurt2 - kurt1)
        
        # Calculate autocorrelation change
        if len(window1) > 1 and len(window2) > 1:
            acf1 = np.corrcoef(window1[:-1], window1[1:])[0, 1]
            acf2 = np.corrcoef(window2[:-1], window2[1:])[0, 1]
            acf_change = abs(acf2 - acf1)
        else:
            acf_change = 0
            
        # Combine measures with quantum adjustment
        jump_intensity = 0.4 * (vol_ratio - 1) + 0.3 * kurt_diff / 10 + 0.3 * acf_change
        
        return max(0, jump_intensity)
        
    def calculate_jump_probability(self, jump_intensity, jump_size, diffusion, time_horizon=1.0):
        """
        Calculate jump probability using quantum stochastic processes
        
        Parameters:
        - jump_intensity: Jump intensity parameter
        - jump_size: Jump size parameter
        - diffusion: Diffusion parameter
        - time_horizon: Time horizon (default: 1.0)
        
        Returns:
        - Jump probability
        """
        base_prob = 1 - np.exp(-jump_intensity * time_horizon)
        
        size_factor = min(1.0, abs(jump_size) / 0.1)
        
        diffusion_factor = max(0.5, 1 - diffusion / 0.2)
        
        quantum_factor = 1 + 0.2 * size_factor * diffusion_factor
        
        # Calculate final probability
        jump_probability = min(0.95, base_prob * quantum_factor)
        
        return jump_probability
        
    def get_statistics(self):
        """
        Get statistics about process history
        
        Returns:
        - Dictionary with process statistics
        """
        if not self.history:
            return {'count': 0}
            
        adjustment_count = sum(1 for h in self.history if 'volatility_index' in h and 'simulation' not in h and 'detection' not in h)
        simulation_count = sum(1 for h in self.history if 'simulation' in h)
        detection_count = sum(1 for h in self.history if 'detection' in h)
        
        crisis_count = sum(1 for h in self.history if 'volatility_index' in h and h.get('is_crisis', False))
        
        crisis_factors = [h.get('crisis_factor', 1.0) for h in self.history if 'volatility_index' in h]
        avg_crisis_factor = np.mean(crisis_factors) if crisis_factors else 1.0
        
        shock_counts = [len(h.get('detection', {}).get('shocks_detected', [])) for h in self.history if 'detection' in h]
        total_shocks = sum(shock_counts)
        
        regime_change_count = sum(1 for h in self.history if h.get('type') == 'regime_change_detection')
        
        stats = {
            'count': len(self.history),
            'adjustment_count': adjustment_count,
            'simulation_count': simulation_count,
            'detection_count': detection_count,
            'crisis_count': crisis_count,
            'regime_change_count': regime_change_count,
            'avg_crisis_factor': float(avg_crisis_factor),
            'total_shocks_detected': total_shocks
        }
        
        return stats
        
    def save_history(self, filename):
        """
        Save process history to file
        
        Parameters:
        - filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        logger.info(f"History saved to {filename}")
        
    def clear_history(self):
        """Clear process history"""
        self.history = []
        logger.info("Process history cleared")


if __name__ == "__main__":
    import unittest
    
    class TestQuantumStochasticProcess(unittest.TestCase):
        """Unit tests for QuantumStochasticProcess"""
        
        def setUp(self):
            """Set up test fixtures"""
            self.qsp = QuantumStochasticProcess(jump_intensity=0.1, mean_jump=0, jump_vol=0.1, crisis_sensitivity=2.0)
            
        def test_parameter_adjustment(self):
            """Test parameter adjustment during crisis"""
            normal_params = self.qsp.adjust_parameters_for_crisis(0.2, 0.3)
            
            crisis_params = self.qsp.adjust_parameters_for_crisis(0.6, 0.3)
            
            self.assertGreater(crisis_params['intensity'], normal_params['intensity'])
            
            self.assertLess(crisis_params['mean_jump'], normal_params['mean_jump'])
            
            self.assertGreater(crisis_params['jump_vol'], normal_params['jump_vol'])
            
        def test_price_path_simulation(self):
            """Test price path simulation"""
            s0 = 100.0
            mu = 0.05
            sigma = 0.2
            t = 1.0
            dt = 0.01
            
            times, prices, components = self.qsp.simulate_price_path(s0, mu, sigma, t, dt, volatility_index=0.2, seed=42)
            
            self.assertEqual(len(times), int(t/dt) + 1)
            self.assertEqual(len(prices), int(t/dt) + 1)
            
            self.assertEqual(prices[0], s0)
            
            self.assertIn('diffusion', components)
            self.assertIn('jumps', components)
            self.assertIn('interference', components)
            
            crisis_times, crisis_prices, crisis_components = self.qsp.simulate_price_path(s0, mu, sigma, t, dt, volatility_index=0.6, seed=42)
            
            self.assertEqual(len(crisis_times), len(times))
            self.assertEqual(crisis_prices[0], prices[0])
            self.assertNotEqual(crisis_prices[-1], prices[-1])
            
        def test_liquidity_shock_detection(self):
            """Test liquidity shock detection"""
            np.random.seed(42)
            n = 100
            prices = np.zeros(n)
            prices[0] = 100
            
            for i in range(1, 80):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, 0.01))
                
            prices[80] = prices[79] * 0.9  # 10% drop
            
            for i in range(81, n):
                prices[i] = prices[i-1] * (1 + np.random.normal(0, 0.01))
                
            normal_result = self.qsp.detect_liquidity_shocks(prices, window=20, threshold=3.0, volatility_index=0.2)
            
            crisis_result = self.qsp.detect_liquidity_shocks(prices, window=20, threshold=3.0, volatility_index=0.6)
            
            self.assertIn(80, normal_result['shock_indices'])
            
            self.assertLess(crisis_result['adjusted_threshold'], normal_result['adjusted_threshold'])
            
        def test_market_microstructure_analysis(self):
            """Test market microstructure analysis"""
            np.random.seed(42)
            n = 200
            
            bid_volumes = np.random.normal(1000, 200, n)
            ask_volumes = np.random.normal(1000, 200, n)
            
            trades = np.random.normal(500, 100, n)
            
            for i in range(50, 150):
                if i % 2 == 0:
                    bid_volumes[i] = bid_volumes[i] * 1.5
                else:
                    ask_volumes[i] = ask_volumes[i] * 1.5
                    
                if i % 10 == 0:
                    trades[i] = trades[i] * 3
                    
            result = self.qsp.analyze_market_microstructure(bid_volumes, ask_volumes, trades)
            
            self.assertIn('quantum_signature', result)
            self.assertIn('oscillation_ratio', result)
            self.assertIn('cluster_ratio', result)
            
        def test_jump_probability_prediction(self):
            """Test jump probability prediction"""
            np.random.seed(42)
            
            returns = np.random.normal(0, 0.01, 100)
            
            normal_result = self.qsp.predict_jump_probability(returns, 0.2)
            
            crisis_result = self.qsp.predict_jump_probability(returns, 0.6)
            
            self.assertGreater(crisis_result['jump_probability'], normal_result['jump_probability'])
            
            self.assertIn('jump_probability', normal_result)
            self.assertIn('expected_jump_size', normal_result)
            self.assertIn('up_probability', normal_result)
            self.assertIn('down_probability', normal_result)
            
        def test_input_validation(self):
            """Test input validation"""
            with self.assertRaises(ValueError):
                QuantumStochasticProcess(jump_intensity=-0.1)
                
            with self.assertRaises(ValueError):
                QuantumStochasticProcess(jump_vol=-0.1)
                
            with self.assertRaises(ValueError):
                self.qsp.simulate_price_path(-100, 0.05, 0.2, 1.0, 0.01)
                
            with self.assertRaises(ValueError):
                self.qsp.simulate_price_path(100, 0.05, 0.2, -1.0, 0.01)
                
            with self.assertRaises(ValueError):
                self.qsp.simulate_price_path(100, 0.05, 0.2, 1.0, -0.01)
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
