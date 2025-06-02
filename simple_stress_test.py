#!/usr/bin/env python3
"""Simple test to find the stress module error"""

import sys
import os

try:
    sys.path.insert(0, os.path.dirname(__file__))
    import mock_algorithm_imports
    sys.modules['AlgorithmImports'] = mock_algorithm_imports
    print("✅ Mock imports loaded")
    
    from advanced_modules.stress_detector import StressDetector
    print("✅ StressDetector imported")
    
    algo = mock_algorithm_imports.QCAlgorithm()
    detector = StressDetector(algo)
    print("✅ StressDetector initialized")
    
    print("🔍 Testing detect method...")
    result = detector.detect("BTCUSD", algo.Time)
    print(f"✅ Result: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
