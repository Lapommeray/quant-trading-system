"""
Test script for advanced indicators
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.indicators_integration_example import demonstrate_integration

def test_dependencies():
    """Test that all required dependencies are available"""
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import hmmlearn
        print("✓ All dependencies are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        assert False, f"Missing dependency: {e}"

def test_indicators():
    """Test that all indicators work correctly"""
    try:
        results = demonstrate_integration()
        
        assert len(results['volatility']) > 0, "Heston volatility should return values"
        assert len(results['ml_rsi']) > 0, "ML RSI should return values"
        assert len(results['order_flow']) > 0, "Order flow should return values"
        assert len(results['regimes']) > 0, "Regime detector should return values"
        
        print("✓ All indicators working correctly")
    except Exception as e:
        print(f"✗ Indicator test failed: {e}")
        assert False, f"Indicator test failed: {e}"

def main():
    """Run all tests"""
    print("Testing Advanced Indicators...")
    
    deps_ok = test_dependencies()
    indicators_ok = test_indicators()
    
    if deps_ok and indicators_ok:
        print("\n🎉 All tests passed! Advanced indicators are ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
