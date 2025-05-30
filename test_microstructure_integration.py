"""
Test script for microstructure integration
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('.')

try:
    from market_microstructure import LimitOrderBook, VPINCalculator
    from statistical_arbitrage import AdvancedCointegration
    from execution import VWAPExecution, OptimalExecution
    from enhanced_strategy import EnhancedStrategy, Strategy
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

def test_imports():
    """Test that all new modules can be imported"""
    if IMPORTS_SUCCESSFUL:
        print("✓ All imports successful")
        return True
    else:
        print(f"✗ Import failed: {IMPORT_ERROR}")
        return False

def test_limit_order_book():
    """Test LimitOrderBook functionality"""
    try:
        lob = LimitOrderBook()
        
        order = {
            'side': 'bid',
            'price': 100.0,
            'quantity': 10,
            'type': 'limit'
        }
        
        executed = lob.process_order(order)
        print(f"✓ LimitOrderBook processed order: {len(executed)} executions")
        return True
    except Exception as e:
        print(f"✗ LimitOrderBook test failed: {e}")
        return False

def test_vpin_calculator():
    """Test VPINCalculator functionality"""
    try:
        vpin = VPINCalculator()
        
        for i in range(100):
            vpin.add_trade(100 + np.random.random(), 10, 1 if i % 2 == 0 else -1)
        
        result = vpin.calculate()
        print(f"✓ VPINCalculator result: {result}")
        return True
    except Exception as e:
        print(f"✗ VPINCalculator test failed: {e}")
        return False

def test_cointegration():
    """Test AdvancedCointegration functionality"""
    try:
        coint = AdvancedCointegration()
        
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'asset1': np.cumsum(np.random.randn(100)),
            'asset2': np.cumsum(np.random.randn(100))
        }, index=dates)
        
        hedge_ratio = coint.hedge_ratio_estimation(data)
        print(f"✓ AdvancedCointegration hedge ratio: {hedge_ratio}")
        return True
    except Exception as e:
        print(f"✗ AdvancedCointegration test failed: {e}")
        return False

def test_vwap_execution():
    """Test VWAPExecution functionality"""
    try:
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        volumes = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
        
        vwap = VWAPExecution(volumes)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        schedule = vwap.get_schedule(1000, start_time, end_time)
        print(f"✓ VWAPExecution schedule: {len(schedule)} time slices")
        return True
    except Exception as e:
        print(f"✗ VWAPExecution test failed: {e}")
        return False

def test_enhanced_strategy():
    """Test EnhancedStrategy functionality"""
    try:
        strategy = EnhancedStrategy()
        
        dates = pd.date_range('2023-01-01', periods=100)
        strategy.data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        strategy.init()
        print("✓ EnhancedStrategy initialized successfully")
        return True
    except Exception as e:
        print(f"✗ EnhancedStrategy test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Microstructure Integration...")
    
    tests = [
        test_imports,
        test_limit_order_book,
        test_vpin_calculator,
        test_cointegration,
        test_vwap_execution,
        test_enhanced_strategy
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All microstructure integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
