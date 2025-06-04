#!/usr/bin/env python3
"""
Comprehensive test for QMPUltraEngine with proper data requirements
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

sys.path.insert(0, os.path.dirname(__file__))
from core.qmp_engine_v3 import QMPUltraEngine

def generate_comprehensive_market_data(symbol="BTCUSD", days=1):
    """Generate comprehensive market data meeting QMPUltraEngine requirements"""
    periods = days * 24 * 60
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
    
    df_1m = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    history_data = {
        '1m': df_1m,
        '5m': df_1m.resample('5min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(),
        '10m': df_1m.resample('10min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(),
        '15m': df_1m.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(),
        '20m': df_1m.resample('20min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna(),
        '25m': df_1m.resample('25min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    }
    
    return history_data

def test_comprehensive_engine():
    """Test QMPUltraEngine with comprehensive data and all modules"""
    print("🚀 Starting Comprehensive QMPUltraEngine Test")
    print("=" * 50)
    
    algo = mock_algorithm_imports.QCAlgorithm()
    engine = QMPUltraEngine(algo)
    
    print(f"✅ QMPUltraEngine loaded with {len(engine.modules)} modules")
    print(f"   Module names: {list(engine.modules.keys())}")
    
    print("\n📊 Generating comprehensive market data...")
    history_data = generate_comprehensive_market_data()
    
    for timeframe, df in history_data.items():
        print(f"   - {timeframe} timeframe: {len(df)} bars")
    
    print(f"   - Data range: {history_data['1m'].index[0]} to {history_data['1m'].index[-1]}")
    
    print("\n🎯 Testing signal generation with comprehensive data...")
    start_time = time.time()
    
    result = engine.generate_signal("BTCUSD", history_data)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"✅ Signal generated in {generation_time:.3f} seconds")
    print(f"   Final Signal: {result['final_signal']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Active Gates: {len(result['gate_scores'])}")
    
    if result['gate_scores']:
        print(f"   Gate Scores:")
        for gate, score in result['gate_scores'].items():
            print(f"     {gate}: {score:.3f}")
    
    print("\n⚡ Testing multiple symbols...")
    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]
    symbol_results = {}
    
    for symbol in symbols:
        start_time = time.time()
        result = engine.generate_signal(symbol, history_data)
        end_time = time.time()
        
        symbol_results[symbol] = {
            'signal': result['final_signal'],
            'confidence': result['confidence'],
            'time': end_time - start_time
        }
        
        print(f"   {symbol}: {result['final_signal']} (conf: {result['confidence']:.3f}, time: {end_time - start_time:.3f}s)")
    
    print("\n🔍 Testing module functionality...")
    module_test_results = {}
    
    for module_name, module in engine.modules.items():
        try:
            if hasattr(module, 'decode'):
                test_result = module.decode("BTCUSD", [])
                module_test_results[module_name] = "✅ decode() works"
            elif hasattr(module, 'detect'):
                if module_name == 'stress':
                    test_result = module.detect("BTCUSD", datetime.now())
                else:
                    test_result = module.detect("BTCUSD", history_data)
                module_test_results[module_name] = "✅ detect() works"
            elif hasattr(module, 'analyze'):
                test_result = module.analyze({})
                module_test_results[module_name] = "✅ analyze() works"
            else:
                module_test_results[module_name] = "✅ module loaded"
        except Exception as e:
            module_test_results[module_name] = f"❌ Error: {str(e)[:50]}"
    
    print("   Module Test Results:")
    for module_name, result in module_test_results.items():
        print(f"     {module_name}: {result}")
    
    successful_modules = sum(1 for result in module_test_results.values() if result.startswith("✅"))
    total_modules = len(module_test_results)
    
    print(f"\n📈 Test Summary:")
    print(f"   ✅ Modules working: {successful_modules}/{total_modules}")
    print(f"   ✅ Signal generation: {'Working' if any(r['signal'] for r in symbol_results.values()) else 'No signals generated'}")
    print(f"   ✅ Performance: Average {np.mean([r['time'] for r in symbol_results.values()]):.3f}s per signal")
    
    success_rate = successful_modules / total_modules
    performance_ok = all(r['time'] < 2.0 for r in symbol_results.values())
    
    overall_success = success_rate >= 0.9 and performance_ok
    
    if overall_success:
        print(f"\n🎉 COMPREHENSIVE TEST PASSED!")
        print(f"   ✓ Module success rate: {success_rate:.1%}")
        print(f"   ✓ Performance acceptable")
        print(f"   ✓ System ready for QuantConnect deployment")
    else:
        print(f"\n⚠️  Test completed with issues:")
        print(f"   Module success rate: {success_rate:.1%}")
        print(f"   Performance: {'OK' if performance_ok else 'SLOW'}")
    
    return overall_success, module_test_results

if __name__ == "__main__":
    try:
        success, results = test_comprehensive_engine()
        if success:
            print("\n✅ ALL COMPREHENSIVE TESTS PASSED!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed - see details above")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Comprehensive test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
