#!/usr/bin/env python3
"""
Quantum Finance Module Map for Oversoul Director

This file defines the mapping between Oversoul Director module names
and quantum finance module names for integration with the QMPUltraEngine.
"""

QUANTUM_MODULE_MAP = {
    'quantum_black_scholes': 'quantum_bs',
    'quantum_stochastic_calculus': 'quantum_process',
    'quantum_portfolio_optimization': 'quantum_portfolio',
    'quantum_risk_measures': 'quantum_risk',
    'quantum_finance_integration': 'quantum_finance',
    
    'entropy_shield_quantum': 'entropy_shield_q',
    'liquidity_mirror_quantum': 'liquidity_mirror_q',
    'legba_crossroads_quantum': 'legba_crossroads_q'
}

QUANTUM_DEFAULT_WEIGHTS = {
    'quantum_bs': 0.08,
    'quantum_process': 0.08,
    'quantum_portfolio': 0.08,
    'quantum_risk': 0.08,
    'quantum_finance': 0.10,
    'entropy_shield_q': 0.10,
    'liquidity_mirror_q': 0.10,
    'legba_crossroads_q': 0.10
}

QUANTUM_MODULE_DESCRIPTIONS = {
    'quantum_bs': 'Quantum Black-Scholes model for option pricing during high volatility',
    'quantum_process': 'Quantum Stochastic Process for modeling market jumps and liquidity shocks',
    'quantum_portfolio': 'Quantum Portfolio Optimization for advanced risk management',
    'quantum_risk': 'Quantum Risk Measures for stress testing under quantum-correlated crashes',
    'quantum_finance': 'Quantum Finance Integration for unified quantum finance functionality',
    'entropy_shield_q': 'Quantum-Enhanced Entropy Shield for improved risk management',
    'liquidity_mirror_q': 'Quantum-Enhanced Liquidity Mirror for better liquidity shock detection',
    'legba_crossroads_q': 'Quantum-Enhanced Legba Crossroads for improved breakout detection'
}

QUANTUM_CRISIS_THRESHOLDS = {
    'volatility_index': 0.3,  # VIX equivalent > 30%
    'quantum_entropy': 0.7,   # Quantum entropy > 70%
    'correlation_spike': 0.8, # Average correlation > 80%
    'liquidity_shock': 0.5    # Liquidity shock probability > 50%
}

QUANTUM_PERFORMANCE_TARGETS = {
    'win_rate': 1.0,           # 100% win rate
    'profit_factor': float('inf'), # Infinite profit factor (no losing trades)
    'max_drawdown': 0.0,       # 0% maximum drawdown
    'sharpe_ratio': 10.0,      # Sharpe ratio > 10
    'federal_outperformance': 2.0  # 200% outperformance vs federal indicators
}
