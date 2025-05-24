import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.event_blackout import EventBlackoutManager
from core.risk_manager import RiskManager
from core.dynamic_slippage import DynamicLiquiditySlippage

class TestChaosScenarios(unittest.TestCase):
    """Test system resilience under extreme market conditions"""
    
    def setUp(self):
        """Set up test environment"""
        class MockAlgorithm:
            def __init__(self):
                self.Portfolio = MockPortfolio()
                
        class MockPortfolio:
            def __init__(self):
                self.TotalPortfolioValue = 100000
                
        self.algorithm = MockAlgorithm()
        self.risk_manager = RiskManager(self.algorithm)
        self.blackout_manager = EventBlackoutManager()
        self.slippage_model = DynamicLiquiditySlippage()
    
    def test_black_swan_resilience(self):
        """Test portfolio survives 10-sigma events"""
        returns = np.random.normal(0, 0.01, 1000)  # Normal market
        returns[-1] = -0.20  # 20% crash (black swan)
        
        results = self.blackout_manager.simulate_black_swan_events(pd.Series(returns))
        
        for result in results:
            self.assertLess(abs(result['max_drawdown']), 0.20, 
                          f"Excessive drawdown in {result['scenario']}: {result['max_drawdown']}")
            
    def test_risk_manager_under_stress(self):
        """Test risk manager performs under volatile conditions"""
        volatile_data = {
            '1m': pd.DataFrame({
                'Close': np.random.normal(100, 5, 10000),  # High volatility
                'Volume': np.random.uniform(1000, 10000, 10000)
            }, index=pd.date_range('2024-01-01', periods=10000, freq='1min'))
        }
        
        position_size = self.risk_manager.calculate_position_size('BTCUSD', 0.8, volatile_data)
        
        self.assertLess(position_size, 0.1, "Position size too aggressive under high volatility")
        
    def test_walk_forward_data_leak(self):
        """Test that walk-forward validation prevents data leakage with paranoid checks"""
        from core.walk_forward_backtest import WalkForwardBacktester
        
        dates = pd.date_range('2024-01-01', periods=300, freq='D')
        data = {
            '1d': pd.DataFrame({
                'Close': np.random.normal(100, 1, 300),
                'Volume': np.random.uniform(1000, 10000, 300)
            }, index=dates)
        }
        
        backtester = WalkForwardBacktester(train_days=180, test_days=30)
        
        current_date = dates[180]
        train_start = current_date - timedelta(days=180)
        train_end = current_date - timedelta(days=1)  # Buffer gap
        test_start = current_date
        test_end = current_date + timedelta(days=30)
        
        train_data = backtester._extract_data_range(data, train_start, train_end)
        test_data = backtester._extract_data_range(data, test_start, test_end)
        
        train_indices = set(train_data['1d'].index)
        test_indices = set(test_data['1d'].index)
        
        self.assertEqual(len(train_indices.intersection(test_indices)), 0, 
                       "CRITICAL: Data leak detected - train and test sets overlap")
        
        max_train_date = train_data['1d'].index.max()
        min_test_date = test_data['1d'].index.min()
        self.assertLess(max_train_date, min_test_date,
                       "CRITICAL: Train data contains future information relative to test data")
        
        self.assertGreaterEqual((min_test_date - max_train_date).days, 1,
                               "CRITICAL: Insufficient buffer gap between train and test data")
        
    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall calculation for fat-tail risk"""
        normal_returns = np.array(np.random.normal(0.0005, 0.01, 1000))
        
        from scipy.stats import t
        fat_tail_returns = np.array(t.rvs(df=3, loc=0.0005, scale=0.01, size=1000))
        
        crash_values = np.array([-0.05, -0.07, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.02, 0.01])
        normal_returns[-10:] = crash_values
        fat_tail_returns[-10:] = crash_values
        
        normal_var = np.percentile(normal_returns, 5)
        normal_es = normal_returns[normal_returns <= normal_var].mean()
        
        fat_var = np.percentile(fat_tail_returns, 5)
        fat_es = fat_tail_returns[fat_tail_returns <= fat_var].mean()
        
        self.assertLess(fat_es, normal_es, 
                      "Expected Shortfall not properly capturing fat-tail risk")
        
    def test_dynamic_slippage_in_crisis(self):
        """Test dynamic slippage model during market crisis"""
        normal_conditions = {
            'volatility': 0.1,  # 10% annualized volatility
            'hour': 12,  # Midday
            'news_factor': 1.0
        }
        
        crisis_conditions = {
            'volatility': 0.5,  # 50% annualized volatility
            'hour': 12,  # Midday
            'news_factor': 5.0
        }
        
        normal_slippage = self.slippage_model.get_dynamic_slippage('BTC', 10000, normal_conditions)
        crisis_slippage = self.slippage_model.get_dynamic_slippage('BTC', 10000, crisis_conditions)
        
        self.assertGreater(crisis_slippage, normal_slippage * 3, 
                         "Dynamic slippage not properly scaling during crisis")
        
        large_order_slippage = self.slippage_model.get_dynamic_slippage('BTC', 100000, crisis_conditions)
        
        self.assertGreater(large_order_slippage, crisis_slippage, 
                         "Dynamic slippage not properly scaling with order size")
        
    def test_numba_optimization(self):
        """Test numba optimization performance"""
        from core.performance_optimizer import PerformanceOptimizer, fast_volatility_calc
        
        try:
            import numba
            has_numba = True
        except ImportError:
            has_numba = False
            
        if not has_numba:
            self.skipTest("Numba not available")
            
        prices = np.random.normal(100, 1, 10000)
        
        import time
        
        start_time = time.time()
        _ = pd.Series(prices).pct_change().rolling(20).std() * np.sqrt(252)
        pandas_time = time.time() - start_time
        
        start_time = time.time()
        _ = fast_volatility_calc(prices)
        numba_time = time.time() - start_time
        
        self.assertLess(numba_time, pandas_time, 
                      "Numba optimization not providing performance improvement")

if __name__ == '__main__':
    unittest.main()
