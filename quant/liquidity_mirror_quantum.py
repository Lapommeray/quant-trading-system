#!/usr/bin/env python3
"""
Quantum-Enhanced Liquidity Mirror Scanner

Extends the standard Liquidity Mirror with quantum stochastic processes
for improved detection of liquidity shocks during extreme market conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_finance.quantum_stochastic_calculus import QuantumStochasticProcess

from quant.liquidity_mirror import LiquidityMirror

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LiquidityMirrorQuantum")

class LiquidityMirrorQuantum(LiquidityMirror):
    """
    Quantum-Enhanced Liquidity Mirror Scanner
    
    Extends the standard Liquidity Mirror with quantum stochastic processes
    for improved detection of liquidity shocks during extreme market conditions.
    
    Key enhancements:
    - Quantum stochastic process for modeling market jumps
    - Liquidity shock detection using quantum noise processes
    - Crisis-sensitive parameter adjustment
    - Enhanced institutional flow detection during panic selling
    """
    
    def __init__(self, min_imbalance=2.0, depth_levels=10, jump_intensity=0.1, crisis_sensitivity=2.0):
        """
        Initialize the Quantum-Enhanced Liquidity Mirror
        
        Parameters:
        - min_imbalance: Minimum bid/ask ratio to detect imbalance (default: 2.0)
        - depth_levels: Number of price levels to analyze (default: 10)
        - jump_intensity: Intensity of jump process (default: 0.1)
        - crisis_sensitivity: Sensitivity to crisis conditions (default: 2.0)
        """
        super().__init__(min_imbalance, depth_levels)
        self.quantum_process = QuantumStochasticProcess(
            jump_intensity=jump_intensity,
            crisis_sensitivity=crisis_sensitivity
        )
        self.quantum_history = []
        
        logger.info(f"Initialized LiquidityMirrorQuantum with min_imbalance={min_imbalance}, "
                   f"depth_levels={depth_levels}, jump_intensity={jump_intensity}, "
                   f"crisis_sensitivity={crisis_sensitivity}")
        
    def scan_liquidity_quantum(self, bids, asks, price_history=None, volatility_index=0.2):
        """
        Enhanced liquidity scanning with quantum jump detection
        
        Parameters:
        - bids: Dictionary of bid prices and volumes {price: volume}
        - asks: Dictionary of ask prices and volumes {price: volume}
        - price_history: Recent price history for jump detection (optional)
        - volatility_index: Market volatility index (default: 0.2)
        
        Returns:
        - Tuple of (signal, ratio, quantum_signal)
        """
        signal, ratio = super().scan_liquidity(bids, asks)
        
        if price_history is None or len(price_history) < 20:
            logger.warning("Insufficient price history for quantum liquidity scan")
            return signal, ratio, None
            
        params = self.quantum_process.adjust_parameters_for_crisis(volatility_index)
        
        try:
            shock_indices = self.quantum_process.detect_liquidity_shocks(
                price_history,
                window=min(20, len(price_history) // 2),
                threshold=3.0,
                volatility_index=volatility_index
            )
        except Exception as e:
            logger.warning(f"Error detecting liquidity shocks: {e}")
            shock_indices = []
        
        quantum_signal = None
        
        recent_shock = False
        try:
            if isinstance(shock_indices, (list, np.ndarray)) and len(shock_indices) > 0:
                if shock_indices[-1] >= len(price_history) - 3:
                    recent_shock = True
        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"Error accessing shock_indices: {e}")
        
        if recent_shock:
            if ratio > self.min_imbalance:
                quantum_signal = "QUANTUM HIDDEN BIDS DETECTED"
            elif ratio < 1/self.min_imbalance:
                quantum_signal = "QUANTUM HIDDEN ASKS DETECTED"
            else:
                quantum_signal = "QUANTUM SHOCK DETECTED"
        
        self.quantum_history.append({
            'timestamp': datetime.now().isoformat(),
            'standard_signal': signal,
            'ratio': float(ratio),
            'quantum_signal': quantum_signal,
            'shock_indices': shock_indices,
            'volatility_index': float(volatility_index),
            'crisis_factor': float(params['crisis_factor'])
        })
        
        logger.debug(f"Quantum liquidity scan: standard={signal}, "
                    f"quantum={quantum_signal}, ratio={ratio:.2f}")
        
        return signal, ratio, quantum_signal
        
    def analyze_order_book_quantum(self, order_book_data, price_history=None, volatility_index=0.2):
        """
        Enhanced order book analysis with quantum stochastic processes
        
        Parameters:
        - order_book_data: Dictionary with 'bids' and 'asks' arrays of [price, volume] pairs
        - price_history: Recent price history for jump detection (optional)
        - volatility_index: Market volatility index (default: 0.2)
        
        Returns:
        - Dictionary with quantum-enhanced order book analysis
        """
        standard_analysis = super().analyze_order_book(order_book_data)
        
        if price_history is None or len(price_history) < 20:
            logger.warning("Insufficient price history for quantum order book analysis")
            standard_analysis['quantum_enhanced'] = False
            return standard_analysis
            
        params = self.quantum_process.adjust_parameters_for_crisis(volatility_index)
        
        shock_indices = self.quantum_process.detect_liquidity_shocks(
            price_history,
            window=min(20, len(price_history) // 2),
            threshold=3.0,
            volatility_index=volatility_index
        )
        
        bids = {float(bid[0]): float(bid[1]) for bid in order_book_data['bids'][:self.depth_levels]}
        asks = {float(ask[0]): float(ask[1]) for ask in order_book_data['asks'][:self.depth_levels]}
        
        _, ratio, quantum_signal = self.scan_liquidity_quantum(bids, asks, price_history, volatility_index)
        
        if quantum_signal == "QUANTUM HIDDEN BIDS DETECTED":
            sentiment = "QUANTUM BULLISH"
            message = "Quantum analysis detected hidden institutional bids. Potential upward pressure."
        elif quantum_signal == "QUANTUM HIDDEN ASKS DETECTED":
            sentiment = "QUANTUM BEARISH"
            message = "Quantum analysis detected hidden institutional asks. Potential downward pressure."
        elif quantum_signal == "QUANTUM SHOCK DETECTED":
            sentiment = "QUANTUM VOLATILE"
            message = "Quantum analysis detected liquidity shock. Extreme caution advised."
        else:
            sentiment = standard_analysis['signal']
            message = f"Standard liquidity ratio: {ratio:.2f}"
            
        enhanced_analysis = standard_analysis.copy()
        enhanced_analysis.update({
            'quantum_signal': quantum_signal,
            'sentiment': sentiment,
            'message': message,
            'shock_indices': shock_indices,
            'volatility_index': float(volatility_index),
            'crisis_factor': float(params['crisis_factor']),
            'quantum_enhanced': True
        })
        
        logger.info(f"Quantum order book analysis: signal={quantum_signal}, "
                   f"sentiment={sentiment}, crisis_factor={params['crisis_factor']:.2f}")
        
        return enhanced_analysis
        
    def detect_dark_liquidity(self, order_book_data, price_history, volume_history, volatility_index=0.2):
        """
        Detect dark liquidity pools using quantum stochastic processes
        
        Parameters:
        - order_book_data: Dictionary with 'bids' and 'asks' arrays of [price, volume] pairs
        - price_history: Recent price history
        - volume_history: Recent volume history
        - volatility_index: Market volatility index (default: 0.2)
        
        Returns:
        - Dictionary with dark liquidity analysis
        """
        if len(price_history) < 30 or len(volume_history) < 30:
            logger.warning("Insufficient history for dark liquidity detection")
            return {'dark_liquidity_detected': False}
            
        s0 = price_history[-1]
        mu = np.mean(np.diff(np.log(price_history[-30:]))) * 252  # Annualized drift
        sigma = np.std(np.diff(np.log(price_history[-30:]))) * np.sqrt(252)  # Annualized volatility
        
        n_paths = 100
        t = 5/252  # 5 trading days
        dt = 1/252  # Daily steps
        
        paths = []
        for i in range(n_paths):
            _, path_prices = self.quantum_process.simulate_price_path(
                s0, mu, sigma, t, dt, volatility_index=volatility_index, seed=42+i
            )
            paths.append(path_prices)
            
        paths = np.array(paths)
        
        final_prices = paths[:, -1]
        hist, bin_edges = np.histogram(final_prices, bins=20)
        
        max_bin = np.argmax(hist)
        concentration_level = hist[max_bin] / n_paths
        
        bids = {float(bid[0]): float(bid[1]) for bid in order_book_data['bids'][:self.depth_levels]}
        asks = {float(ask[0]): float(ask[1]) for ask in order_book_data['asks'][:self.depth_levels]}
        
        concentration_price = (bin_edges[max_bin] + bin_edges[max_bin+1]) / 2
        
        bid_prices = sorted(bids.keys(), reverse=True)
        ask_prices = sorted(asks.keys())
        
        closest_bid = min(bid_prices, key=lambda x: abs(x - concentration_price)) if bid_prices else None
        closest_ask = min(ask_prices, key=lambda x: abs(x - concentration_price)) if ask_prices else None
        
        in_liquidity_gap = False
        if closest_bid is not None and closest_ask is not None:
            gap_size = (closest_ask - closest_bid) / s0  # Normalized gap size
            in_liquidity_gap = gap_size > 0.005  # 0.5% gap is significant
            
        dark_liquidity_detected = concentration_level > 0.3 and in_liquidity_gap
        
        result = {
            'dark_liquidity_detected': dark_liquidity_detected,
            'concentration_level': float(concentration_level),
            'concentration_price': float(concentration_price),
            'in_liquidity_gap': in_liquidity_gap,
            'gap_size': float(gap_size) if in_liquidity_gap else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Dark liquidity detection: detected={dark_liquidity_detected}, "
                   f"concentration={concentration_level:.2f}, price={concentration_price:.2f}")
        
        return result
        
    def predict_liquidity_shock(self, order_book_data, price_history, volume_history, confidence_threshold=0.95):
        """
        Predict liquidity shocks using quantum stochastic processes with statistical validation
        
        Parameters:
        - order_book_data: Dictionary with 'bids' and 'asks'
        - price_history: Recent price history
        - volume_history: Recent volume history
        - confidence_threshold: Minimum confidence threshold (default: 0.95)
        
        Returns:
        - Dictionary with liquidity shock prediction results and statistical validation
        """
        if 'bids' not in order_book_data or 'asks' not in order_book_data:
            return {'shock_probability': 0.0, 'confidence': 0.0, 'statistically_validated': False}
            
        if len(price_history) < 30 or len(volume_history) < 30:
            return {'shock_probability': 0.0, 'confidence': 0.0, 'statistically_validated': False}
            
        returns = np.diff(np.log(price_history))
        
        best_bid = float(order_book_data['bids'][0][0]) if len(order_book_data['bids']) > 0 else 0
        best_ask = float(order_book_data['asks'][0][0]) if len(order_book_data['asks']) > 0 else 0
        
        if best_bid == 0 or best_ask == 0:
            return {'shock_probability': 0.0, 'confidence': 0.0, 'statistically_validated': False}
            
        spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
        
        bid_volume = sum(float(bid[1]) for bid in order_book_data['bids'])
        ask_volume = sum(float(ask[1]) for ask in order_book_data['asks'])
        
        if bid_volume > 0 and ask_volume > 0:
            volume_imbalance = max(bid_volume / ask_volume, ask_volume / bid_volume)
        else: 
            volume_imbalance = 1.0
            
        jump_intensity, jump_size, diffusion = self.quantum_process.estimate_jump_diffusion_parameters(
            price_history, volume_history, returns
        )
        
        base_shock_probability = self.quantum_process.calculate_jump_probability(
            jump_intensity, jump_size, diffusion, time_horizon=1.0
        )
        
        spread_factor = min(1.0, spread * 10)  # Normalize spread
        imbalance_factor = min(1.0, volume_imbalance / 5)  # Normalize imbalance
        
        shock_probability = 0.3 * base_shock_probability + 0.4 * spread_factor + 0.3 * imbalance_factor
        shock_probability = max(0.0, min(1.0, shock_probability))
        
        bootstrap_samples = 1000
        bootstrap_probabilities = np.zeros(bootstrap_samples)
        
        for i in range(bootstrap_samples):
            bootstrap_indices = np.random.choice(len(returns), len(returns), replace=True)
            bootstrap_returns = returns[bootstrap_indices]
            
            bootstrap_jump_intensity, bootstrap_jump_size, bootstrap_diffusion = \
                self.quantum_process.estimate_jump_diffusion_parameters(
                    price_history[-len(bootstrap_returns)-1:],
                    volume_history[-len(bootstrap_returns)-1:],
                    bootstrap_returns
                )
            
            bootstrap_probabilities[i] = self.quantum_process.calculate_jump_probability(
                bootstrap_jump_intensity, bootstrap_jump_size, bootstrap_diffusion, time_horizon=1.0
            )
            
        bootstrap_probabilities = np.sort(bootstrap_probabilities)
        lower_bound = np.percentile(bootstrap_probabilities, 2.5)  # 2.5th percentile for 95% CI
        upper_bound = np.percentile(bootstrap_probabilities, 97.5)  # 97.5th percentile for 95% CI
        
        confidence = 1.0 - (upper_bound - lower_bound)
        
        statistically_validated = (confidence >= confidence_threshold)
        
        result = {
            'shock_probability': float(shock_probability),
            'confidence': float(confidence),
            'statistically_validated': statistically_validated,
            'jump_intensity': float(jump_intensity),
            'jump_size': float(jump_size),
            'diffusion': float(diffusion),
            'spread': float(spread),
            'volume_imbalance': float(volume_imbalance),
            'confidence_threshold': float(confidence_threshold),
            'probability_lower_bound': float(lower_bound),
            'probability_upper_bound': float(upper_bound)
        }
        
        self.quantum_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'liquidity_shock_prediction',
            'result': result
        })
        
        return result
        
    def get_statistics(self):
        """
        Get statistics about liquidity analysis
        
        Returns:
        - Dictionary with liquidity analysis statistics
        """
        standard_stats = super().get_statistics() if hasattr(super(), 'get_statistics') else {}
        
        quantum_signals = [h.get('quantum_signal') for h in self.quantum_history if 'quantum_signal' in h]
        
        hidden_bids_count = sum(1 for s in quantum_signals if s == "QUANTUM HIDDEN BIDS DETECTED")
        hidden_asks_count = sum(1 for s in quantum_signals if s == "QUANTUM HIDDEN ASKS DETECTED")
        shock_count = sum(1 for s in quantum_signals if s == "QUANTUM SHOCK DETECTED")
        
        crisis_factors = [h.get('crisis_factor', 1.0) for h in self.quantum_history if 'crisis_factor' in h]
        
        # Extract liquidity shock prediction statistics
        shock_predictions = [h for h in self.quantum_history if h.get('type') == 'liquidity_shock_prediction']
        shock_probabilities = [h.get('result', {}).get('shock_probability', 0) for h in shock_predictions]
        validated_shocks = [h.get('result', {}).get('statistically_validated', False) for h in shock_predictions]
        
        quantum_stats = {
            'quantum_history_count': len(self.quantum_history),
            'hidden_bids_count': hidden_bids_count,
            'hidden_asks_count': hidden_asks_count,
            'shock_count': shock_count,
            'avg_crisis_factor': float(np.mean(crisis_factors)) if crisis_factors else 1.0,
            'max_crisis_factor': float(np.max(crisis_factors)) if crisis_factors else 1.0,
            'liquidity_shock_prediction_count': len(shock_predictions),
            'validated_shock_count': sum(validated_shocks),
            'avg_shock_probability': float(np.mean(shock_probabilities)) if shock_probabilities else 0.0
        }
        
        return {**standard_stats, **quantum_stats}
