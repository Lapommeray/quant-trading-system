"""
Test suite for institutional enhancements
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
    from execution.advanced.smart_routing import InstitutionalOptimalExecution, AdvancedVWAPExecution
    from enhanced_strategy import EnhancedStrategy, Strategy
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

def test_imports():
    """Test that all institutional modules can be imported"""
    if IMPORTS_SUCCESSFUL:
        print("✓ All institutional imports successful")
        return True
    else:
        print(f"✗ Import failed: {IMPORT_ERROR}")
        return False

def test_institutional_lob_features():
    """Test enhanced LimitOrderBook with institutional features"""
    try:
        lob = LimitOrderBook()
        
        impact = lob.process_trade_institutional(100.0, 1000, 'buy')
        assert lob.trade_imbalance == 1000, f"Expected 1000, got {lob.trade_imbalance}"
        
        impact = lob.process_trade_institutional(100.1, 500, 'sell')
        assert lob.trade_imbalance == 500, f"Expected 500, got {lob.trade_imbalance}"
        
        print("✓ LimitOrderBook institutional features working")
        return True
    except Exception as e:
        print(f"✗ LimitOrderBook institutional test failed: {e}")
        return False

def test_institutional_cointegration():
    """Test enhanced cointegration methods"""
    try:
        coint = AdvancedCointegration()
        
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'asset1': np.cumsum(np.random.randn(100)),
            'asset2': np.cumsum(np.random.randn(100))
        }, index=dates)
        
        hedge_ratio = coint.johansen_test(data)
        assert len(hedge_ratio) == 2, f"Expected length 2, got {len(hedge_ratio)}"
        
        x = np.cumsum(np.random.randn(50))
        y = x + np.random.randn(50) * 0.1  # Cointegrated series
        is_coint = coint.kernel_coint(x, y)
        
        print("✓ AdvancedCointegration institutional methods working")
        return True
    except Exception as e:
        print(f"✗ AdvancedCointegration institutional test failed: {e}")
        return False

def test_institutional_execution():
    """Test institutional execution algorithms"""
    try:
        exec_algo = InstitutionalOptimalExecution(volatility_forecast=0.2, liquidity_profile=1.0)
        schedule = exec_algo.solve_institutional(1000, 24)
        assert len(schedule) == 24, f"Expected 24 periods, got {len(schedule)}"
        
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        volumes = pd.Series(np.random.randint(1000, 10000, 100), index=dates)
        
        vwap_exec = AdvancedVWAPExecution(volumes)
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        schedule = vwap_exec.get_institutional_schedule(1000, start_time, end_time)
        
        print("✓ Institutional execution algorithms working")
        return True
    except Exception as e:
        print(f"✗ Institutional execution test failed: {e}")
        return False

def test_backwards_compatibility():
    """Ensure existing functionality still works"""
    try:
        lob = LimitOrderBook()
        order = {'side': 'bid', 'price': 100.0, 'quantity': 10, 'type': 'limit'}
        executed = lob.process_order(order)
        
        coint = AdvancedCointegration()
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'asset1': np.cumsum(np.random.randn(100)),
            'asset2': np.cumsum(np.random.randn(100))
        }, index=dates)
        hedge_ratio = coint.hedge_ratio_estimation(data)
        
        strategy = EnhancedStrategy(use_institutional=False)
        strategy.data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        strategy.init()
        
        print("✓ Backwards compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Backwards compatibility test failed: {e}")
        return False

def test_institutional_strategy_integration():
    """Test EnhancedStrategy with institutional features enabled"""
    try:
        strategy = EnhancedStrategy(use_institutional=True)
        
        dates = pd.date_range('2023-01-01', periods=100)
        strategy.data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        strategy.init()
        
        assert hasattr(strategy, 'institutional_execution'), "Missing institutional_execution"
        assert hasattr(strategy, 'advanced_vwap'), "Missing advanced_vwap"
        
        print("✓ EnhancedStrategy institutional integration working")
        return True
    except Exception as e:
        print(f"✗ EnhancedStrategy institutional test failed: {e}")
        return False

def main():
    """Run all institutional tests"""
    print("Testing Institutional Integration...")
    
    tests = [
        test_imports,
        test_institutional_lob_features,
        test_institutional_cointegration,
        test_institutional_execution,
        test_backwards_compatibility,
        test_institutional_strategy_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} institutional tests passed")
    
    if passed == len(tests):
        print("🎉 All institutional integration tests passed!")
        return 0
    else:
        print("❌ Some institutional tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
