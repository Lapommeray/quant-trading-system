#!/usr/bin/env python3
"""
Fast test for QMPUltraEngine with performance optimization
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
import mock_algorithm_imports
sys.modules['AlgorithmImports'] = mock_algorithm_imports

sys.path.append(os.path.dirname(__file__))
from core.qmp_engine_v3 import QMPUltraEngine

def generate_small_market_data(symbol="BTCUSD", periods=50):
    """Generate small realistic market data for fast testing"""
    dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
    
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, periods)
    base_price = 50000
    prices = base_price * np.exp(np.cumsum(returns))
    
    opens = np.roll(prices, 1)
    opens[0] = base_price
    
    highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, periods)))
    volumes = np.random.lognormal(10, 1, periods)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    history_data = {
        '1m': df,
        '5m': df.iloc[::5].copy(),  # Sample every 5th row
        '10m': df.iloc[::10].copy(),
        '15m': df.iloc[::15].copy(),
        '20m': df.iloc[::20].copy(),
        '25m': df.iloc[::25].copy()
    }
    
    return history_data

def test_fast_engine_performance():
    """Test QMPUltraEngine with optimized performance"""
    print("🚀 Starting Fast QMPUltraEngine Test")
    print("=" * 40)
    
    algo = mock_algorithm_imports.QCAlgorithm()
    engine = QMPUltraEngine(algo)
    
    print(f"✅ QMPUltraEngine loaded with {len(engine.modules)} modules")
    
    print("📊 Generating small market data...")
    history_data = generate_small_market_data()
    
    print(f"   - 1m timeframe: {len(history_data['1m'])} bars")
    print(f"   - Data range: {history_data['1m'].index[0]} to {history_data['1m'].index[-1]}")
    
    print("\n🎯 Testing single signal generation...")
    start_time = time.time()
    
    result = engine.generate_signal("BTCUSD", history_data)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"✅ Signal generated in {generation_time:.3f} seconds")
    print(f"   Final Signal: {result['final_signal']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Active Gates: {len(result['gate_scores'])}")
    
    if generation_time > 1.0:
        print(f"⚠️  WARNING: Signal generation too slow ({generation_time:.3f}s > 1.0s)")
        return False
    
    print("\n⚡ Testing rapid signal generation (10 iterations)...")
    start_time = time.time()
    
    signals = []
    for i in range(10):
        result = engine.generate_signal("BTCUSD", history_data)
        signals.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 10
    
    print(f"✅ 10 signals generated in {total_time:.3f} seconds")
    print(f"   Average: {avg_time:.4f} seconds per signal")
    print(f"   Rate: {1/avg_time:.1f} signals per second")
    
    if avg_time > 0.5:
        print(f"⚠️  WARNING: Average signal generation too slow ({avg_time:.3f}s > 0.5s)")
        return False
    
    signal_counts = {}
    confidence_sum = 0
    for result in signals:
        signal = result['final_signal']
        if signal not in signal_counts:
            signal_counts[signal] = 0
        signal_counts[signal] += 1
        confidence_sum += result['confidence']
    
    avg_confidence = confidence_sum / len(signals)
    
    print(f"\n📈 Signal Analysis:")
    for signal, count in signal_counts.items():
        percentage = (count / len(signals)) * 100
        print(f"   {signal}: {count} ({percentage:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    
    print(f"\n✅ Performance Requirements Met!")
    print(f"   ✓ Single signal: {generation_time:.3f}s < 1.0s")
    print(f"   ✓ Average signal: {avg_time:.3f}s < 0.5s")
    print(f"   ✓ Rate: {1/avg_time:.1f} signals/sec")
    
    return True

if __name__ == "__main__":
    try:
        success = test_fast_engine_performance()
        if success:
            print("\n🎉 FAST PERFORMANCE TEST PASSED!")
            print("✅ System ready for QuantConnect deployment")
        else:
            print("\n❌ Performance test failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fast test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
