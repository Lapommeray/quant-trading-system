#!/usr/bin/env python3
"""
Test script for the Integrated Cosmic Verification System
Tests the combination of advanced verification features with cosmic perfection modules
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.integrated_cosmic_verification import IntegratedCosmicVerification

def run_verification_test(god_mode=False, eternal=False, no_loss=False):
    """
    Run verification test with specified parameters
    
    Parameters:
    - god_mode: Enable GOD MODE
    - eternal: Enable eternal execution
    - no_loss: Disallow losses
    
    Returns:
    - Dictionary with test results
    """
    print(f"Running integrated verification test:")
    print(f"  GOD MODE: {'✓' if god_mode else '✗'}")
    print(f"  Eternal Execution: {'✓' if eternal else '✗'}")
    print(f"  Zero Losses: {'✓' if no_loss else '✗'}")
    print("-" * 50)
    
    config = {
        "god_mode": god_mode,
        "eternal_execution": eternal,
        "loss_disallowed": no_loss,
        "max_drawdown_threshold": 0.05,
        "verification_modules": {
            "dark_pool": True,
            "gamma_trap": True,
            "sentiment": True,
            "alpha": True,
            "order_book": True,
            "neural_pattern": True,
            "dark_pool_dna": True,
            "market_regime": True
        }
    }
    
    system = IntegratedCosmicVerification(config)
    
    sample_data = {
        "symbol": "BTC/USD",
        "timestamp": datetime.now().timestamp() * 1000,  # Current time in milliseconds
        "open": 50000,
        "high": 50500,
        "low": 49500,
        "close": 50200,
        "volume": 1000,
        "order_book": {
            "bids": [[49900, 10], [49800, 15], [49700, 20]],
            "asks": [[50300, 10], [50400, 15], [50500, 20]]
        }
    }
    
    print("\nVerifying data integrity...")
    verification_result = system.verify_data_integrity(sample_data)
    print(f"Verification result: {'✅ PASSED' if verification_result['verified'] else '❌ FAILED'}")
    if not verification_result['verified']:
        print(f"Error: {verification_result.get('error', 'Unknown error')}")
    
    print("\nGenerating trading signal...")
    signal_result = system.generate_trading_signal(sample_data)
    
    if signal_result.get("signal"):
        print(f"Signal generated: {signal_result['signal']['direction']} with confidence {signal_result['signal']['confidence']:.2f}")
        print(f"Signal source: {signal_result['signal']['source']}")
    else:
        print("No signal generated")
        if signal_result.get("verification_failed"):
            print(f"Verification failed: {signal_result.get('error', 'Unknown error')}")
    
    print("\nRunning stress test...")
    for event in ["covid_crash", "fed_panic", "flash_crash"]:
        print(f"\nTesting event: {event}")
        stress_result = system.run_stress_test(symbol="BTC/USD", event=event)
        print(f"Max drawdown: {stress_result['max_drawdown']:.2%}")
        print(f"Passed: {'✅ YES' if stress_result['passed'] else '❌ NO'}")
        print(f"Final portfolio value: ${stress_result['final_portfolio_value']:.2f}")
        print(f"Return: {stress_result['return']:.2%}")
        print(f"Trades executed: {stress_result['trades']}")
    
    print("\nGenerating verification report...")
    report_dir = "./reports"
    report_paths = system.generate_verification_report(report_dir)
    print(f"Reports generated in {report_dir}")
    print(f"Main report: {report_paths['main_report']}")
    print(f"Stress report: {report_paths['stress_report']}")
    
    return {
        "verification_result": verification_result,
        "signal_result": signal_result,
        "report_paths": report_paths
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test the Integrated Cosmic Verification System")
    parser.add_argument("--god-mode", action="store_true", help="Enable GOD MODE")
    parser.add_argument("--eternal", action="store_true", help="Enable eternal execution")
    parser.add_argument("--no-loss", action="store_true", help="Disallow losses")
    
    args = parser.parse_args()
    
    run_verification_test(
        god_mode=args.god_mode,
        eternal=args.eternal,
        no_loss=args.no_loss
    )

if __name__ == "__main__":
    main()
