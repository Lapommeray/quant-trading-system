"""
Market Mind Reader Integration Example

This script demonstrates how to integrate the Market Mind Reader package
with the QMP Overrider system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

from market_mind_reader import FedWhisperer, CandlestickDNASequencer, LiquidityXRay, EnhancedIndicator
from market_mind_reader.retail_dna_extractor import RetailDNAExtractor
from market_mind_reader.quantum_noise_trader import QuantumNoiseTrader

from market_warfare import ElectronicWarfare, SignalsIntelligence, PsychologicalOperations, MarketCommander

def main():
    """
    Main function to demonstrate Market Mind Reader integration
    """
    print("Market Mind Reader Integration Example")
    print("======================================")
    
    print("\nInitializing modules...")
    fed_whisperer = FedWhisperer()
    dna_sequencer = CandlestickDNASequencer()
    liquidity_xray = LiquidityXRay()
    retail_dna = RetailDNAExtractor()
    quantum_trader = QuantumNoiseTrader()
    enhanced_indicator = EnhancedIndicator()
    
    ew = ElectronicWarfare()
    sigint = SignalsIntelligence()
    psyops = PsychologicalOperations()
    commander = MarketCommander()
    
    print("\nGenerating sample data...")
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    df = pd.DataFrame({
        'open': np.random.normal(100, 2, 200),
        'high': np.random.normal(102, 2, 200),
        'low': np.random.normal(98, 2, 200),
        'close': np.random.normal(101, 2, 200),
        'volume': np.random.normal(1000000, 200000, 200)
    }, index=dates)
    
    for i in range(len(df)):
        values = [df.iloc[i]['open'], df.iloc[i]['close']]
        df.iloc[i, df.columns.get_loc('high')] = max(values) + np.random.normal(1, 0.2)
        df.iloc[i, df.columns.get_loc('low')] = min(values) - np.random.normal(1, 0.2)
    
    print("\nTesting Fed Whisperer...")
    fed_sentiment = fed_whisperer.get_fed_sentiment()
    fed_impact = fed_whisperer.predict_market_impact()
    
    print(f"Fed Sentiment: {fed_sentiment['sentiment']} (Confidence: {fed_sentiment['confidence']:.2f})")
    print(f"Market Impact: {fed_impact['prediction']} (Confidence: {fed_impact['confidence']:.2f})")
    
    print("\nTesting Candlestick DNA Sequencer...")
    dna_prediction = dna_sequencer.predict_next_candle(df)
    
    print(f"DNA Prediction: {dna_prediction['prediction']} (Confidence: {dna_prediction['confidence']:.2f})")
    print(f"Dominant Pattern: {dna_prediction['dominant_pattern']}")
    
    print("\nTesting Liquidity X-Ray...")
    symbol = "SPY"
    liquidity_impact = liquidity_xray.predict_price_impact(symbol)
    
    print(f"Liquidity Direction: {liquidity_impact['direction']} (Confidence: {liquidity_impact['confidence']:.2f})")
    
    print("\nTesting Retail DNA Extractor...")
    retail_fomo = retail_dna.get_retail_fomo(symbol)
    
    print(f"Retail Sentiment: {retail_fomo['sentiment']} (Confidence: {retail_fomo['confidence']:.2f})")
    print(f"FOMO Level: {retail_fomo['fomo_level']:.2f}")
    
    print("\nTesting Quantum Noise Trader...")
    quantum_signal = quantum_trader.get_quantum_signal()
    
    print(f"Quantum Signal: {quantum_signal['signal']} (Confidence: {quantum_signal['confidence']:.2f})")
    
    print("\nTesting Enhanced Indicator...")
    enhanced_signal = enhanced_indicator.get_signal(symbol, df)
    
    print(f"Enhanced Signal: {enhanced_signal['signal']} (Confidence: {enhanced_signal['confidence']:.2f})")
    print(f"Fed Bias: {enhanced_signal['fed_bias']}")
    print(f"DNA Prediction: {enhanced_signal['dna_prediction']}")
    print(f"Liquidity Direction: {enhanced_signal['liquidity_direction']}")
    
    print("\nTesting Market Warfare modules...")
    
    spoofing = ew.detect_spoofing(symbol)
    print(f"Spoofing Detected: {spoofing['bid_spoof'] or spoofing['ask_spoof']}")
    
    dark_pool = sigint.get_dark_pool_signal(symbol)
    print(f"Dark Pool Signal: {dark_pool['signal']} (Confidence: {dark_pool['confidence']:.2f})")
    
    sentiment = psyops.generate_retail_sentiment_report(symbol)
    print(f"Market Phase: {sentiment['market_phase']}")
    print(f"Contrarian Signal: {sentiment['contrarian_signal']} (Confidence: {sentiment['confidence']:.2f})")
    
    execution = commander.execute(symbol)
    print(f"Warfare Tactic: {execution['tactic']}")
    print(f"Warfare Signal: {execution['signal']} (Confidence: {execution['confidence']:.2f})")
    
    print("\nPerformance Metrics:")
    enhanced_metrics = enhanced_indicator.get_performance_metrics()
    warfare_metrics = commander.get_performance_metrics()
    
    print("\nMarket Mind Reader Metrics:")
    for module, metrics in enhanced_metrics.items():
        print(f"{module}: Win Rate Boost: {metrics['win_rate_boost']:.2f}, Drawdown Reduction: {metrics['drawdown_reduction']:.2f}")
    
    print("\nMarket Warfare Metrics:")
    for tactic, metrics in warfare_metrics.items():
        print(f"{tactic}: Win Rate: {metrics['win_rate']:.2f}, Annual ROI: {metrics['annual_roi']:.2f}, Drawdown: {metrics['drawdown']:.2f}")
    
    print("\nLegal Compliance Checklist:")
    compliance = commander.get_legal_compliance_checklist()
    
    for check, status in compliance.items():
        if check != "timestamp":
            print(f"{check}: {'✓' if status else '✗'}")
    
    print("\nIntegration Example Complete")

if __name__ == "__main__":
    main()
