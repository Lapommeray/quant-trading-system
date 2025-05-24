#!/usr/bin/env python3
"""
Quantum Finance Integration Module

Integrates quantum finance models with the QMPUltraEngine and Oversoul Director.
Provides a unified interface for quantum finance functionality.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import sys
import logging

from quantum_finance.quantum_black_scholes import QuantumBlackScholes
from quantum_finance.quantum_stochastic_calculus import QuantumStochasticProcess
from quantum_finance.quantum_portfolio_optimization import QuantumPortfolioOptimizer
from quantum_finance.quantum_risk_measures import QuantumRiskMeasures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.entropy_shield_quantum import EntropyShieldQuantum
from quant.liquidity_mirror_quantum import LiquidityMirrorQuantum
from signals.legba_crossroads_quantum import LegbaCrossroadsQuantum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumFinanceIntegration")

class QuantumFinanceIntegration:
    """
    Quantum Finance Integration Module
    
    Integrates quantum finance models with the QMPUltraEngine and Oversoul Director.
    Provides a unified interface for quantum finance functionality.
    
    Key features:
    - Unified interface for all quantum finance modules
    - Integration with QMPUltraEngine and Oversoul Director
    - Enhanced market analysis with quantum finance models
    - Portfolio optimization with quantum superposition
    - Stress testing with quantum-correlated crashes
    - Comprehensive performance metrics
    """
    
    def __init__(self, config=None):
        """
        Initialize Quantum Finance Integration
        
        Parameters:
        - config: Configuration dictionary (default: None)
        """
        self.config = config or {}
        
        self.quantum_bs = QuantumBlackScholes(
            hbar=self.config.get('hbar', 0.01)
        )
        
        self.quantum_process = QuantumStochasticProcess(
            jump_intensity=self.config.get('jump_intensity', 0.1),
            crisis_sensitivity=self.config.get('crisis_sensitivity', 2.0)
        )
        
        self.quantum_portfolio = QuantumPortfolioOptimizer(
            risk_aversion=self.config.get('risk_aversion', 3.0),
            entanglement_factor=self.config.get('entanglement_factor', 0.2),
            crisis_boost=self.config.get('crisis_boost', 2.0)
        )
        
        self.quantum_risk = QuantumRiskMeasures(
            confidence_level=self.config.get('confidence_level', 0.95),
            quantum_factor=self.config.get('quantum_factor', 0.3)
        )
        
        self.entropy_shield = EntropyShieldQuantum(
            max_risk=self.config.get('max_risk', 0.02),
            min_risk=self.config.get('min_risk', 0.005),
            volatility_window=self.config.get('volatility_window', 20),
            quantum_factor=self.config.get('quantum_factor', 0.3)
        )
        
        self.liquidity_mirror = LiquidityMirrorQuantum(
            min_imbalance=self.config.get('min_imbalance', 2.0),
            depth_levels=self.config.get('depth_levels', 10),
            jump_intensity=self.config.get('jump_intensity', 0.1),
            crisis_sensitivity=self.config.get('crisis_sensitivity', 2.0)
        )
        
        self.legba_crossroads = LegbaCrossroadsQuantum(
            ema_period=self.config.get('ema_period', 21),
            volume_mult=self.config.get('volume_mult', 1.5),
            chaos_threshold=self.config.get('chaos_threshold', 2.0),
            hbar=self.config.get('hbar', 0.01)
        )
        
        self.history = []
        
        logger.info(f"Initialized QuantumFinanceIntegration with config: {self.config}")
        
    def analyze_market(self, symbol, data, volatility_index=None):
        """
        Comprehensive market analysis using quantum finance models
        
        Parameters:
        - symbol: Asset symbol
        - data: Market data dictionary with price, volume, etc.
        - volatility_index: Market volatility index (default: None, calculated from data)
        
        Returns:
        - Dictionary with quantum finance analysis results
        """
        if 'close' not in data or len(data['close']) < 30:
            logger.warning(f"Insufficient data for quantum analysis: {symbol}")
            return {'error': 'Insufficient data for quantum analysis'}
            
        close_prices = np.array(data['close'])
        
        if volatility_index is None:
            returns = np.diff(np.log(close_prices[-30:]))
            volatility_index = np.std(returns) * np.sqrt(252)
            
        returns = np.diff(np.log(close_prices[-30:]))
        implied_vol = np.std(returns) * np.sqrt(252)
        
        current_price = close_prices[-1]
        strike_price = current_price
        risk_free_rate = 0.01
        time_to_expiry = 30/365  # 30 days
        
        standard_price = self.quantum_bs.price_call_option(
            current_price, strike_price, risk_free_rate, implied_vol, time_to_expiry, vov=0
        )
        
        quantum_price = self.quantum_bs.price_call_option(
            current_price, strike_price, risk_free_rate, implied_vol, time_to_expiry, vov=0.2
        )
        
        quantum_premium = (quantum_price - standard_price) / standard_price
        
        params = self.quantum_process.adjust_parameters_for_crisis(volatility_index)
        
        quantum_var = self.quantum_risk.quantum_var(returns)
        quantum_cvar = self.quantum_risk.quantum_cvar(returns)
        
        if 'volume' in data and len(data['volume']) >= 30:
            volumes = np.array(data['volume'][-30:])
            entropy_analysis = self.entropy_shield.analyze_market_state_quantum(
                close_prices[-30:], volumes, returns=returns
            )
        else:
            entropy_analysis = self.entropy_shield.analyze_market_state_quantum(
                close_prices[-30:], np.ones(30), returns=returns
            )
            
        if 'high' in data and 'low' in data and 'volume' in data:
            high_prices = np.array(data['high'][-30:])
            low_prices = np.array(data['low'][-30:])
            volumes = np.array(data['volume'][-30:])
            
            atr = self.legba_crossroads.calculate_atr(high_prices, low_prices, close_prices[-30:])
            
            legba_signal = self.legba_crossroads.detect_breakout_quantum(
                close_prices[-30:], volumes, atr, entropy=entropy_analysis.get('quantum_entropy', 0.5)
            )
            
            volatility_analysis = self.legba_crossroads.analyze_volatility_structure(
                close_prices[-30:], high_prices, low_prices
            )
        else:
            legba_signal = None
            volatility_analysis = {'error': 'Insufficient data for volatility analysis'}
            
        if quantum_premium > 0.1 and quantum_var > 0.05:
            market_state = "QUANTUM CRISIS"
            confidence = 0.3
            direction = "neutral"
        elif quantum_premium > 0.05:
            market_state = "QUANTUM VOLATILITY"
            confidence = 0.5
            direction = "neutral" if quantum_var > 0.03 else "bullish"
        else:
            market_state = "QUANTUM NORMAL"
            confidence = 0.7
            direction = "bullish" if close_prices[-1] > close_prices[-2] else "bearish"
            
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'market_state': market_state,
            'direction': direction,
            'confidence': float(confidence),
            'quantum_premium': float(quantum_premium),
            'quantum_var': float(quantum_var),
            'quantum_cvar': float(quantum_cvar),
            'volatility_index': float(volatility_index),
            'crisis_factor': float(params['crisis_factor']),
            'entropy_analysis': entropy_analysis,
            'legba_signal': legba_signal,
            'volatility_analysis': volatility_analysis
        }
        
        self.history.append(result)
        
        logger.info(f"Market analysis for {symbol}: state={market_state}, "
                   f"direction={direction}, confidence={confidence:.2f}")
        
        return result
        
    def optimize_portfolio(self, symbols, expected_returns, cov_matrix, market_returns=None):
        """
        Optimize portfolio using quantum portfolio optimization
        
        Parameters:
        - symbols: List of asset symbols
        - expected_returns: Array of expected returns for each asset
        - cov_matrix: Covariance matrix of asset returns
        - market_returns: Historical market returns for crisis detection (default: None)
        
        Returns:
        - Dictionary with optimized portfolio weights
        """
        weights = self.quantum_portfolio.optimize_portfolio(
            expected_returns, cov_matrix, market_returns
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'weights': {symbols[i]: float(weights[i]) for i in range(len(symbols))},
            'expected_return': float(np.dot(weights, expected_returns)),
            'expected_risk': float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        }
        
        self.history.append(result)
        
        logger.info(f"Portfolio optimization: return={result['expected_return']:.4f}, "
                   f"risk={result['expected_risk']:.4f}")
        
        return result
        
    def stress_test(self, symbols, weights, returns, correlation_boost=0.3, volatility_boost=0.5):
        """
        Stress test portfolio using quantum risk measures
        
        Parameters:
        - symbols: List of asset symbols
        - weights: Portfolio weights
        - returns: Historical returns data
        - correlation_boost: How much to boost correlations in stress scenarios (default: 0.3)
        - volatility_boost: How much to boost volatility in stress scenarios (default: 0.5)
        
        Returns:
        - Dictionary with stress test results
        """
        results = self.quantum_risk.stress_test_portfolio(
            returns, weights, correlation_boost=correlation_boost, volatility_boost=volatility_boost
        )
        
        results['symbols'] = symbols
        results['weights'] = {symbols[i]: float(weights[i]) for i in range(len(symbols))}
        results['timestamp'] = datetime.now().isoformat()
        
        self.history.append(results)
        
        logger.info(f"Stress test: var={results['stressed_var']:.4f}, "
                   f"cvar={results['stressed_cvar']:.4f}, "
                   f"worst_case_loss={results['worst_case_loss']:.4f}")
        
        return results
        
    def generate_trading_signal(self, symbol, data, account_size=10000, stop_loss_pct=0.02):
        """
        Generate trading signal using quantum-enhanced modules
        
        Parameters:
        - symbol: Asset symbol
        - data: Market data dictionary with price, volume, etc.
        - account_size: Account size in currency units (default: 10000)
        - stop_loss_pct: Stop loss percentage (default: 0.02 = 2%)
        
        Returns:
        - Dictionary with trading signal information
        """
        if 'close' not in data or len(data['close']) < 30:
            logger.warning(f"Insufficient data for trading signal: {symbol}")
            return {'error': 'Insufficient data for trading signal'}
            
        close_prices = np.array(data['close'])
        current_price = close_prices[-1]
        
        returns = np.diff(np.log(close_prices[-30:]))
        
        analysis = self.analyze_market(symbol, data)
        
        if 'error' in analysis:
            return {'error': analysis['error']}
            
        market_state = analysis['market_state']
        direction = analysis['direction']
        confidence = analysis['confidence']
        legba_signal = analysis['legba_signal']
        
        signal_type = None
        if legba_signal == "⚡QUANTUM GATE OPEN⚡":
            signal_type = "QUANTUM_BREAKOUT"
        elif legba_signal == "⚡GATE OPEN⚡":
            signal_type = "BREAKOUT"
        elif market_state == "QUANTUM CRISIS":
            signal_type = "AVOID"
        elif market_state == "QUANTUM VOLATILITY" and direction == "bullish":
            signal_type = "CAUTIOUS_ENTRY"
        elif market_state == "QUANTUM NORMAL" and direction == "bullish":
            signal_type = "NORMAL_ENTRY"
        else:
            signal_type = "NO_SIGNAL"
            
        if signal_type in ["QUANTUM_BREAKOUT", "BREAKOUT", "CAUTIOUS_ENTRY", "NORMAL_ENTRY"]:
            quantum_entropy = analysis['entropy_analysis'].get('quantum_entropy', 0.5)
            
            position_info = self.entropy_shield.position_size_quantum(
                quantum_entropy, account_size, current_price, stop_loss_pct, returns=returns
            )
            
            position_size = position_info['position_size']
            risk_pct = position_info['risk_pct']
        else:
            position_size = 0
            risk_pct = 0
            
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signal': signal_type,
            'direction': direction,
            'confidence': float(confidence),
            'position_size': float(position_size),
            'risk_pct': float(risk_pct),
            'current_price': float(current_price),
            'market_state': market_state,
            'legba_signal': legba_signal
        }
        
        self.history.append(result)
        
        logger.info(f"Trading signal for {symbol}: signal={signal_type}, "
                   f"direction={direction}, position_size={position_size:.2f}")
        
        return result
        
    def save_config(self, filename='quantum_finance_config.json'):
        """
        Save configuration to a JSON file
        
        Parameters:
        - filename: Output filename (default: 'quantum_finance_config.json')
        """
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Configuration saved to {filename}")
            
    def load_config(self, filename='quantum_finance_config.json'):
        """
        Load configuration from a JSON file
        
        Parameters:
        - filename: Input filename (default: 'quantum_finance_config.json')
        
        Returns:
        - Configuration dictionary
        """
        with open(filename, 'r') as f:
            self.config = json.load(f)
            
        logger.info(f"Configuration loaded from {filename}")
            
        return self.config
        
    def predict_federal_outperformance(self, symbol, market_data, federal_indicators, confidence_threshold=0.99):
        """
        Predict outperformance versus federal institution indicators with statistical validation
        
        Parameters:
        - symbol: Asset symbol
        - market_data: Dictionary with market data (prices, volumes, etc.)
        - federal_indicators: Dictionary with federal indicator data
        - confidence_threshold: Minimum confidence threshold (default: 0.99)
        
        Returns:
        - Dictionary with outperformance prediction results and statistical validation
        """
        if 'close' not in market_data or len(market_data['close']) < 30:
            return {'outperformance': 0.0, 'confidence': 0.0, 'statistically_validated': False}
            
        analysis = self.analyze_market(symbol, market_data)
        
        market_state = analysis['market_state']
        direction = analysis['direction']
        quantum_var = analysis.get('quantum_var', 0.1)
        quantum_entropy = analysis.get('entropy_analysis', {}).get('quantum_entropy', 0.5)
        
        expected_return = 0.0
        if market_state == "QUANTUM NORMAL" and direction == "bullish":
            expected_return = 0.1
        elif market_state == "QUANTUM VOLATILITY" and direction == "bullish":
            expected_return = 0.15
        elif market_state == "QUANTUM VOLATILITY" and direction == "neutral":
            expected_return = 0.05
        elif market_state == "QUANTUM CRISIS" and direction == "bullish":
            expected_return = 0.2  # Higher return potential in crisis if bullish
        elif market_state == "QUANTUM CRISIS" and direction == "bearish":
            expected_return = -0.05
        else:
            expected_return = 0.02
            
        fed_returns = np.array([indicator['return'] for indicator in federal_indicators.values()])
        fed_mean_return = np.mean(fed_returns)
        
        base_outperformance = expected_return / fed_mean_return if fed_mean_return > 0 else 2.0
        
        quantum_adjustment = 0.3 * (1 - quantum_entropy) + 0.2 * (1 - min(quantum_var * 10, 1.0))
        
        outperformance = base_outperformance * (1 + quantum_adjustment)
        
        bootstrap_samples = 10000
        bootstrap_outperformances = np.zeros(bootstrap_samples)
        
        for i in range(bootstrap_samples):
            bootstrap_indices = np.random.choice(len(fed_returns), len(fed_returns), replace=True)
            bootstrap_fed_returns = fed_returns[bootstrap_indices]
            bootstrap_fed_mean = np.mean(bootstrap_fed_returns)
            
            bootstrap_outperformance = expected_return / bootstrap_fed_mean if bootstrap_fed_mean > 0 else 2.0
            bootstrap_outperformances[i] = bootstrap_outperformance * (1 + quantum_adjustment)
            
        bootstrap_outperformances = np.sort(bootstrap_outperformances[~np.isinf(bootstrap_outperformances)])
        if len(bootstrap_outperformances) > 0:
            lower_bound = np.percentile(bootstrap_outperformances, 2.5)  # 2.5th percentile for 95% CI
            upper_bound = np.percentile(bootstrap_outperformances, 97.5)  # 97.5th percentile for 95% CI
        else:
            lower_bound = outperformance
            upper_bound = outperformance
            
        confidence = 1.0 - (upper_bound - lower_bound) / (upper_bound + lower_bound) if (upper_bound + lower_bound) > 0 else 0.0
        
        statistically_validated = (confidence >= confidence_threshold and lower_bound >= 2.0)
        
        result = {
            'outperformance': float(outperformance),
            'confidence': float(confidence),
            'statistically_validated': statistically_validated,
            'base_outperformance': float(base_outperformance),
            'quantum_adjustment': float(quantum_adjustment),
            'expected_return': float(expected_return),
            'federal_mean_return': float(fed_mean_return),
            'market_state': market_state,
            'direction': direction,
            'target_outperformance': 2.0,  # 200% outperformance
            'meets_target': outperformance >= 2.0,
            'confidence_threshold': float(confidence_threshold),
            'outperformance_lower_bound': float(lower_bound),
            'outperformance_upper_bound': float(upper_bound)
        }
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'federal_outperformance',
            'result': result
        })
        
        return result
        
    def get_statistics(self):
        """
        Get statistics about quantum finance integration
        
        Returns:
        - Dictionary with quantum finance statistics
        """
        if not self.history:
            return {'count': 0}
            
        market_analysis_count = sum(1 for h in self.history if 'market_state' in h)
        portfolio_count = sum(1 for h in self.history if 'weights' in h)
        stress_test_count = sum(1 for h in self.history if 'stressed_var' in h)
        signal_count = sum(1 for h in self.history if 'signal' in h)
        
        quantum_breakout_count = sum(1 for h in self.history if h.get('signal') == "QUANTUM_BREAKOUT")
        breakout_count = sum(1 for h in self.history if h.get('signal') == "BREAKOUT")
        
        crisis_count = sum(1 for h in self.history if h.get('market_state') == "QUANTUM_CRISIS")
        
        # Extract federal outperformance statistics
        outperformance_predictions = [h for h in self.history if h.get('type') == 'federal_outperformance']
        outperformance_values = [h.get('result', {}).get('outperformance', 0) for h in outperformance_predictions]
        validated_outperformances = [h.get('result', {}).get('statistically_validated', False) for h in outperformance_predictions]
        target_met_count = sum(1 for h in outperformance_predictions if h.get('result', {}).get('meets_target', False))
        
        stats = {
            'count': len(self.history),
            'market_analysis_count': market_analysis_count,
            'portfolio_count': portfolio_count,
            'stress_test_count': stress_test_count,
            'signal_count': signal_count,
            'quantum_breakout_count': quantum_breakout_count,
            'breakout_count': breakout_count,
            'crisis_count': crisis_count,
            'outperformance_prediction_count': len(outperformance_predictions),
            'validated_outperformance_count': sum(validated_outperformances),
            'target_met_count': target_met_count,
            'avg_outperformance': float(np.mean(outperformance_values)) if outperformance_values else 0.0,
            'max_outperformance': float(np.max(outperformance_values)) if outperformance_values else 0.0
        }
        
        return stats
