#!/usr/bin/env python3
"""Debug script to isolate the module testing error in comprehensive test"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import mock_algorithm_imports
sys.modules['AlgorithmImports'] = mock_algorithm_imports

def test_module_functionality_isolated():
    """Test the exact module functionality testing that's failing"""
    print("🔍 Testing module functionality in isolation...")
    
    try:
        from advanced_modules.stress_detector import StressDetector
        
        algo = mock_algorithm_imports.QCAlgorithm()
        module = StressDetector(algo)
        print("✅ StressDetector initialized")
        
        print("🔍 Testing detect() method...")
        test_result = module.detect("BTCUSD", {})  # This might be the issue - passing {} instead of proper time
        print(f"✅ detect() with dict works: {test_result}")
        
        print("🔍 Testing detect() with proper time...")
        test_result2 = module.detect("BTCUSD", algo.Time)
        print(f"✅ detect() with time works: {test_result2}")
        
        print("🔍 Testing exact pattern from comprehensive test...")
        if hasattr(module, 'detect'):
            test_result3 = module.detect("BTCUSD", {})  # This is likely the problematic call
            print(f"✅ Exact pattern works: {test_result3}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in module testing: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_module_functionality_isolated()
