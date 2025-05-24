#!/usr/bin/env python3
"""
Test script for quantum finance integration

This script tests the quantum finance modules and their integration
with the sacred-quant modules system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_finance.quantum_black_scholes import QuantumBlackScholes
from quantum_finance.quantum_stochastic_calculus import QuantumStochasticProcess
from quantum_finance.quantum_portfolio_optimization import QuantumPortfolioOptimizer
from quantum_finance.quantum_risk_measures import QuantumRiskMeasures
from quantum_finance.quantum_finance_integration import QuantumFinanceIntegration

from quant.entropy_shield_quantum import EntropyShieldQuantum
from quant.liquidity_mirror_quantum import LiquidityMirrorQuantum
from signals.legba_crossroads_quantum import LegbaCrossroadsQuantum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_finance_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QuantumFinanceTest")

def test_quantum_black_scholes():
    """Test Quantum Black-Scholes model"""
    logger.info("Testing Quantum Black-Scholes")
    
    qbs = QuantumBlackScholes(hbar=0.01)
    
    s = 100.0  # Current price
    k = 100.0  # Strike price
    r = 0.01   # Risk-free rate
    t = 30/365 # Time to expiry (30 days)
    
    volatilities = [0.2, 0.4, 0.6, 0.8]
    
    results = []
    
    for vol in volatilities:
        standard_price = qbs.price_call_option(s, k, r, vol, t, vov=0)
        
        quantum_price = qbs.price_call_option(s, k, r, vol, t, vov=0.2)
        
        premium = (quantum_price - standard_price) / standard_price * 100
        
        results.append({
            'volatility': vol,
            'standard_price': standard_price,
            'quantum_price': quantum_price,
            'premium_pct': premium
        })
        
        logger.info(f"Volatility: {vol:.2f}, Standard Price: ${standard_price:.4f}, Quantum Price: ${quantum_price:.4f}, Premium: {premium:.2f}%")
        
    return results
    
def test_quantum_stochastic_process():
    """Test Quantum Stochastic Process"""
    logger.info("Testing Quantum Stochastic Process")
    
    qsp = QuantumStochasticProcess(jump_intensity=0.1, crisis_sensitivity=2.0)
    
    s0 = 100.0  # Initial price
    mu = 0.05   # Drift
    sigma = 0.2 # Volatility
    t = 30/365  # Time horizon (30 days)
    dt = 1/365  # Daily steps
    
    volatility_indices = [0.2, 0.4, 0.6]
    
    crisis_adjustments = []
    
    for vol_idx in volatility_indices:
        params = qsp.adjust_parameters_for_crisis(vol_idx)
        
        crisis_adjustments.append({
            'volatility_index': vol_idx,
            'crisis_factor': params['crisis_factor'],
            'intensity': params['intensity'],
            'mean_jump': params['mean_jump'],
            'jump_vol': params['jump_vol']
        })
        
        logger.info(f"Volatility Index: {vol_idx:.2f}, Crisis Factor: {params['crisis_factor']:.2f}, Jump Intensity: {params['intensity']:.4f}")
        
    logger.info("Simulating price paths")
    
    times, normal_prices = qsp.simulate_price_path(s0, mu, sigma, t, dt, volatility_index=0.2, seed=42)
    _, crisis_prices = qsp.simulate_price_path(s0, mu, sigma, t, dt, volatility_index=0.6, seed=42)
    
    logger.info(f"Normal scenario final price: ${normal_prices[-1]:.2f}")
    logger.info(f"Crisis scenario final price: ${crisis_prices[-1]:.2f}")
    
    return {
        'crisis_adjustments': crisis_adjustments,
        'normal_prices': normal_prices.tolist(),
        'crisis_prices': crisis_prices.tolist()
    }
    
def test_quantum_portfolio_optimization():
    """Test Quantum Portfolio Optimization"""
    logger.info("Testing Quantum Portfolio Optimization")
    
    qpo = QuantumPortfolioOptimizer(risk_aversion=3.0, entanglement_factor=0.2, crisis_boost=2.0)
    
    n_assets = 4
    expected_returns = np.array([0.05, 0.07, 0.06, 0.08])
    
    correlation = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0]
    ])
    
    volatilities = np.array([0.15, 0.2, 0.18, 0.25])
    
    cov_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            cov_matrix[i, j] = correlation[i, j] * volatilities[i] * volatilities[j]
            
    np.random.seed(42)
    market_returns = np.random.normal(0.0005, 0.01, 100)
    
    crisis_returns = market_returns.copy()
    crisis_returns[80:] = np.random.normal(-0.02, 0.03, 20)  # Add crisis at the end
    
    normal_weights = qpo.optimize_portfolio(expected_returns, cov_matrix, market_returns)
    normal_return = np.dot(normal_weights, expected_returns)
    normal_risk = np.sqrt(np.dot(normal_weights.T, np.dot(cov_matrix, normal_weights)))
    normal_sharpe = normal_return / normal_risk
    
    logger.info(f"Normal scenario - Return: {normal_return:.4f}, Risk: {normal_risk:.4f}, Sharpe: {normal_sharpe:.4f}")
    logger.info(f"Normal weights: {normal_weights}")
    
    crisis_weights = qpo.optimize_portfolio(expected_returns, cov_matrix, crisis_returns)
    crisis_return = np.dot(crisis_weights, expected_returns)
    crisis_risk = np.sqrt(np.dot(crisis_weights.T, np.dot(cov_matrix, crisis_weights)))
    crisis_sharpe = crisis_return / crisis_risk
    
    logger.info(f"Crisis scenario - Return: {crisis_return:.4f}, Risk: {crisis_risk:.4f}, Sharpe: {crisis_sharpe:.4f}")
    logger.info(f"Crisis weights: {crisis_weights}")
    
    return {
        'normal': {
            'weights': normal_weights.tolist(),
            'return': float(normal_return),
            'risk': float(normal_risk),
            'sharpe': float(normal_sharpe)
        },
        'crisis': {
            'weights': crisis_weights.tolist(),
            'return': float(crisis_return),
            'risk': float(crisis_risk),
            'sharpe': float(crisis_sharpe)
        }
    }
    
def test_quantum_risk_measures():
    """Test Quantum Risk Measures"""
    logger.info("Testing Quantum Risk Measures")
    
    qrm = QuantumRiskMeasures(confidence_level=0.95, quantum_factor=0.3)
    
    np.random.seed(42)
    n_days = 252
    n_assets = 3
    
    normal_returns = np.random.normal(0.0005, 0.01, (n_days, n_assets))
    
    crisis_returns = normal_returns.copy()
    crisis_correlation = np.array([
        [1.0, 0.8, 0.7],
        [0.8, 1.0, 0.9],
        [0.7, 0.9, 1.0]
    ])
    
    for i in range(n_days - 20, n_days):
        crisis_returns[i] = np.random.multivariate_normal(
            [-0.02, -0.02, -0.02],
            crisis_correlation * 0.03**2,
            1
        )[0]
        
    normal_var = qrm.quantum_var(normal_returns)
    normal_cvar = qrm.quantum_cvar(normal_returns)
    
    normal_classical_var = qrm.history[-2]['classical_var']
    normal_classical_cvar = qrm.history[-1]['classical_cvar']
    
    logger.info(f"Normal scenario - Classical VaR: {normal_classical_var:.6f}, Quantum VaR: {normal_var:.6f}")
    logger.info(f"Normal scenario - Classical CVaR: {normal_classical_cvar:.6f}, Quantum CVaR: {normal_cvar:.6f}")
    
    crisis_var = qrm.quantum_var(crisis_returns)
    crisis_cvar = qrm.quantum_cvar(crisis_returns)
    
    crisis_classical_var = qrm.history[-2]['classical_var']
    crisis_classical_cvar = qrm.history[-1]['classical_cvar']
    
    logger.info(f"Crisis scenario - Classical VaR: {crisis_classical_var:.6f}, Quantum VaR: {crisis_var:.6f}")
    logger.info(f"Crisis scenario - Classical CVaR: {crisis_classical_cvar:.6f}, Quantum CVaR: {crisis_cvar:.6f}")
    
    weights = np.array([0.4, 0.3, 0.3])
    stress_results = qrm.stress_test_portfolio(crisis_returns, weights)
    
    logger.info(f"Stress test - Stressed VaR: {stress_results['stressed_var']:.6f}, Stressed CVaR: {stress_results['stressed_cvar']:.6f}")
    logger.info(f"Stress test - Worst-case loss: {stress_results['worst_case_loss']:.6f}, Probability of extreme loss: {stress_results['prob_extreme_loss']:.2%}")
    
    return {
        'normal': {
            'classical_var': float(normal_classical_var),
            'quantum_var': float(normal_var),
            'classical_cvar': float(normal_classical_cvar),
            'quantum_cvar': float(normal_cvar)
        },
        'crisis': {
            'classical_var': float(crisis_classical_var),
            'quantum_var': float(crisis_var),
            'classical_cvar': float(crisis_classical_cvar),
            'quantum_cvar': float(crisis_cvar)
        },
        'stress_test': stress_results
    }
    
def test_enhanced_modules():
    """Test enhanced modules"""
    logger.info("Testing enhanced modules")
    
    np.random.seed(42)
    n_days = 100
    
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))
    volumes = np.random.normal(1000, 200, n_days)
    high_prices = prices * (1 + np.random.uniform(0, 0.01, n_days))
    low_prices = prices * (1 - np.random.uniform(0, 0.01, n_days))
    
    prices[-20:] = prices[-21] * np.cumprod(1 + np.random.normal(-0.005, 0.03, 20))
    volumes[-20:] = volumes[-20:] * 2
    
    returns = np.diff(np.log(prices))
    
    logger.info("Testing EntropyShieldQuantum")
    
    es_quantum = EntropyShieldQuantum()
    
    standard_entropy = es_quantum.calc_entropy(prices)
    quantum_entropy = es_quantum.calc_quantum_entropy(prices, volumes)
    
    logger.info(f"Standard Entropy: {standard_entropy:.4f}, Quantum Entropy: {quantum_entropy:.4f}")
    
    account_size = 10000
    current_price = prices[-1]
    
    standard_position = es_quantum.position_size(standard_entropy, account_size, current_price)
    quantum_position = es_quantum.position_size_quantum(quantum_entropy, account_size, current_price, returns=returns)
    
    logger.info(f"Standard Position Size: {standard_position['position_size']:.2f}, Quantum Position Size: {quantum_position['position_size']:.2f}")
    
    logger.info("Testing LiquidityMirrorQuantum")
    
    lm_quantum = LiquidityMirrorQuantum()
    
    bids = {prices[-1] * (1 - 0.001 * i): volumes[-1] * (1 - 0.1 * i) for i in range(10)}
    asks = {prices[-1] * (1 + 0.001 * i): volumes[-1] * (1 - 0.2 * i) for i in range(10)}
    
    standard_signal, ratio = lm_quantum.scan_liquidity(bids, asks)
    _, _, quantum_signal = lm_quantum.scan_liquidity_quantum(bids, asks, prices, volatility_index=0.3)
    
    logger.info(f"Standard Signal: {standard_signal}, Quantum Signal: {quantum_signal}")
    
    logger.info("Testing LegbaCrossroadsQuantum")
    
    lc_quantum = LegbaCrossroadsQuantum()
    
    atr = lc_quantum.calculate_atr(high_prices, low_prices, prices)
    
    standard_signal = lc_quantum.detect_breakout(prices, volumes, atr)
    quantum_signal = lc_quantum.detect_breakout_quantum(prices, volumes, atr)
    
    logger.info(f"Standard Signal: {standard_signal}, Quantum Signal: {quantum_signal}")
    
    return {
        'entropy_shield': {
            'standard_entropy': float(standard_entropy),
            'quantum_entropy': float(quantum_entropy),
            'standard_position_size': float(standard_position['position_size']),
            'quantum_position_size': float(quantum_position['position_size'])
        },
        'liquidity_mirror': {
            'standard_signal': standard_signal,
            'quantum_signal': quantum_signal,
            'ratio': float(ratio)
        },
        'legba_crossroads': {
            'standard_signal': standard_signal,
            'quantum_signal': quantum_signal
        }
    }
    
def test_integration():
    """Test quantum finance integration"""
    logger.info("Testing quantum finance integration")
    
    qfi = QuantumFinanceIntegration()
    
    np.random.seed(42)
    n_days = 100
    n_assets = 3
    
    asset_prices = []
    for i in range(n_assets):
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))
        prices[-20:] = prices[-21] * np.cumprod(1 + np.random.normal(-0.005, 0.03, 20))
        asset_prices.append(prices)
        
    asset_returns = []
    for prices in asset_prices:
        returns = np.diff(np.log(prices))
        asset_returns.append(returns)
        
    asset_returns = np.array(asset_returns).T  # Shape: (n_days-1, n_assets)
    
    logger.info("Testing market analysis")
    
    data = {
        'close': asset_prices[0],
        'volume': np.random.normal(1000, 200, n_days),
        'high': asset_prices[0] * (1 + np.random.uniform(0, 0.01, n_days)),
        'low': asset_prices[0] * (1 - np.random.uniform(0, 0.01, n_days))
    }
    
    analysis = qfi.analyze_market('BTC', data, volatility_index=0.3)
    
    logger.info(f"Market State: {analysis['market_state']}, Direction: {analysis['direction']}, Confidence: {analysis['confidence']:.2f}")
    logger.info(f"Quantum Premium: {analysis['quantum_premium']:.4f}, Quantum VaR: {analysis['quantum_var']:.4f}")
    
    logger.info("Testing portfolio optimization")
    
    expected_returns = np.array([0.05, 0.07, 0.06])
    cov_matrix = np.cov(asset_returns.T)
    
    portfolio = qfi.optimize_portfolio(['BTC', 'ETH', 'XRP'], expected_returns, cov_matrix, asset_returns)
    
    logger.info(f"Expected Return: {portfolio['expected_return']:.4f}, Expected Risk: {portfolio['expected_risk']:.4f}")
    logger.info(f"Portfolio Weights: {portfolio['weights']}")
    
    logger.info("Testing stress testing")
    
    weights = np.array([0.4, 0.3, 0.3])
    stress_results = qfi.stress_test(['BTC', 'ETH', 'XRP'], weights, asset_returns)
    
    logger.info(f"Stressed VaR: {stress_results['stressed_var']:.6f}, Stressed CVaR: {stress_results['stressed_cvar']:.6f}")
    logger.info(f"Worst-case loss: {stress_results['worst_case_loss']:.6f}, Probability of extreme loss: {stress_results['prob_extreme_loss']:.2%}")
    
    logger.info("Testing trading signal generation")
    
    signal = qfi.generate_trading_signal('BTC', data, account_size=10000)
    
    logger.info(f"Trading Signal: {signal['signal']}, Direction: {signal['direction']}, Position Size: {signal['position_size']:.2f}")
    
    return {
        'market_analysis': analysis,
        'portfolio_optimization': portfolio,
        'stress_test': stress_results,
        'trading_signal': signal
    }
    
def run_all_tests():
    """Run all tests"""
    logger.info("Running all quantum finance tests")
    
    results = {}
    
    results['quantum_black_scholes'] = test_quantum_black_scholes()
    results['quantum_stochastic_process'] = test_quantum_stochastic_process()
    results['quantum_portfolio_optimization'] = test_quantum_portfolio_optimization()
    results['quantum_risk_measures'] = test_quantum_risk_measures()
    results['enhanced_modules'] = test_enhanced_modules()
    results['integration'] = test_integration()
    
    with open('quantum_finance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info("All quantum finance tests completed successfully")
    
    return results
    
if __name__ == "__main__":
    run_all_tests()
