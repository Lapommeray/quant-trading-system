#!/usr/bin/env python3
"""
Isolated test for QMPUltraEngine integration with mock AlgorithmImports
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))
import mock_algorithm_imports
sys.modules['AlgorithmImports'] = mock_algorithm_imports

sys.path.append('Deco_30/QMP_Overrider_Final_20250419_203349')

def test_engine_with_all_modules():
    """Test QMPUltraEngine with all 13 new quantum modules"""
    try:
        from core.qmp_engine_v3 import QMPUltraEngine
        
        algo = mock_algorithm_imports.QCAlgorithm()
        engine = QMPUltraEngine(algo)
        
        print(f"✅ QMPUltraEngine loaded with {len(engine.modules)} modules")
        
        expected_new_modules = [
            'quantum_topology', 'nonergodic_calculus', 'meta_learning',
            'neural_pde_market', 'execution_alpha', 'hyper_topology',
            'path_signature', 'dark_pool_gan', 'neuromorphic_pde',
            'quantum_execution', 'dark_pool_dna', 'neural_holography',
            'quantum_liquidity'
        ]
        
        found_modules = []
        for module_name in expected_new_modules:
            if module_name in engine.modules:
                found_modules.append(module_name)
                print(f"  ✅ {module_name} module loaded")
            else:
                print(f"  ❌ {module_name} module missing")
        
        print(f"✅ Found {len(found_modules)}/13 new quantum modules")
        
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 100)
        prices = 50000 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 1, 100)
        
        df = pd.DataFrame({
            'Open': np.roll(prices, 1),
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        history_data = {'1m': df}
        result = engine.generate_signal("BTCUSD", history_data)
        
        print(f"✅ Signal generation successful")
        print(f"   Final Signal: {result['final_signal']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Active Modules: {len([m for m in result.get('module_results', {}).values() if m.get('signal') != 'NEUTRAL'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ QMPUltraEngine integration test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmark():
    """Test performance under load"""
    try:
        from core.qmp_engine_v3 import QMPUltraEngine
        import time
        
        algo = mock_algorithm_imports.QCAlgorithm()
        engine = QMPUltraEngine(algo)
        
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({
            'Open': [50000],
            'High': [50100],
            'Low': [49900],
            'Close': [50050],
            'Volume': [1000]
        })
        
        history_data = {'1m': df}
        
        start_time = time.time()
        for i in range(50):
            result = engine.generate_signal("BTCUSD", history_data)
        end_time = time.time()
        
        duration = end_time - start_time
        avg_time = duration / 50
        
        print(f"✅ Performance test completed")
        print(f"   50 signals in {duration:.2f} seconds")
        print(f"   Average: {avg_time:.3f} seconds per signal")
        
        if avg_time > 2.0:
            print("⚠️  Performance may be slow for live trading")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting QMPUltraEngine Integration Test")
    print("=" * 50)
    
    success1 = test_engine_with_all_modules()
    success2 = test_performance_benchmark()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 ALL INTEGRATION TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ Some integration tests failed")
        sys.exit(1)
