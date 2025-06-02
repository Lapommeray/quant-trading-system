#!/usr/bin/env python3
"""Debug script to isolate the stress module error"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import mock_algorithm_imports
sys.modules['AlgorithmImports'] = mock_algorithm_imports

from advanced_modules.stress_detector import StressDetector

def test_stress_detector():
    """Test stress detector in isolation to find the error"""
    print("🔍 Testing StressDetector in isolation...")
    
    try:
        algo = mock_algorithm_imports.QCAlgorithm()
        stress_detector = StressDetector(algo)
        print("✅ StressDetector initialized successfully")
        
        result = stress_detector.detect("BTCUSD", algo.Time)
        print(f"✅ detect() method works: {result}")
        
        result2 = stress_detector.detect("AAPL", algo.Time)
        print(f"✅ detect() with AAPL works: {result2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in StressDetector: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_stress_detector()
