import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import RiskManager

class MockAlgorithm:
    """Mock algorithm class for testing"""
    class Portfolio:
        TotalPortfolioValue = 100000.0
        
    def __init__(self):
        self.Portfolio = self.Portfolio()

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.algorithm = MockAlgorithm()
        self.risk_manager = RiskManager(self.algorithm)
        
        dates = pd.date_range('2024-01-01', periods=30*1440, freq='min')  # 30 days of minute data
        
        normal_returns = np.random.normal(0.0001, 0.001, 30*1440-1)
        normal_prices = 100 * np.exp(np.cumsum(normal_returns))
        normal_prices = np.insert(normal_prices, 0, 100.0)
        
        self.normal_data = {
            '1m': pd.DataFrame({
                'Close': normal_prices,
                'Open': normal_prices * 0.999,
                'High': normal_prices * 1.001,
                'Low': normal_prices * 0.998,
                'Volume': np.random.uniform(1000, 10000, 30*1440)
            }, index=dates)
        }
        
        fat_tail_returns = np.random.standard_t(3, 30*1440-1) * 0.001
        fat_tail_prices = 100 * np.exp(np.cumsum(fat_tail_returns))
        fat_tail_prices = np.insert(fat_tail_prices, 0, 100.0)
        
        self.fat_tail_data = {
            '1m': pd.DataFrame({
                'Close': fat_tail_prices,
                'Open': fat_tail_prices * 0.999,
                'High': fat_tail_prices * 1.001,
                'Low': fat_tail_prices * 0.998,
                'Volume': np.random.uniform(1000, 10000, 30*1440)
            }, index=dates)
        }
        
        crash_returns = np.random.normal(0.0001, 0.001, 30*1440-1)
        crash_indices = [5000, 15000, 25000]
        for idx in crash_indices:
            crash_returns[idx] = -0.05  # 5% sudden drop
        
        crash_prices = 100 * np.exp(np.cumsum(crash_returns))
        crash_prices = np.insert(crash_prices, 0, 100.0)
        
        self.crash_data = {
            '1m': pd.DataFrame({
                'Close': crash_prices,
                'Open': crash_prices * 0.999,
                'High': crash_prices * 1.001,
                'Low': crash_prices * 0.998,
                'Volume': np.random.uniform(1000, 10000, 30*1440)
            }, index=dates)
        }
        
    def test_normal_distribution_sizing(self):
        """Test position sizing with normal returns"""
        position_size = self.risk_manager.calculate_position_size('SPY', 0.8, self.normal_data)
        
        self.assertLess(position_size, 0.25, "Position size too large for normal distribution")
        self.assertGreater(position_size, 0.0, "Position size should be positive")
        
    def test_fat_tail_distribution_sizing(self):
        """Test position sizing with fat-tail returns"""
        position_size = self.risk_manager.calculate_position_size('SPY', 0.8, self.fat_tail_data)
        
        normal_position = self.risk_manager.calculate_position_size('SPY', 0.8, self.normal_data)
        
        self.assertLess(position_size, normal_position, 
                       "Fat-tail distribution should result in smaller position size")
        
    def test_crash_scenario_sizing(self):
        """Test position sizing during market crash"""
        position_size = self.risk_manager.calculate_position_size('SPY', 0.8, self.crash_data)
        
        self.assertLess(position_size, 0.05, 
                       "Position size should be very small during crash scenario")
        
    def test_black_swan_protection(self):
        """Test that extreme events trigger protection mechanisms"""
        dates = pd.date_range('2024-01-01', periods=30*1440, freq='min')
        
        returns = np.random.normal(0.0001, 0.001, 30*1440-1)
        
        crash_start = 20000
        for i in range(crash_start, crash_start + 7*1440):
            if i < len(returns):
                returns[i] = -0.01  # Sustained extreme negative returns
        
        prices = 100 * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, 100.0)
        
        black_swan_data = {
            '1m': pd.DataFrame({
                'Close': prices,
                'Open': prices * 0.999,
                'High': prices * 1.001,
                'Low': prices * 0.998,
                'Volume': np.random.uniform(1000, 10000, 30*1440)
            }, index=dates)
        }
        
        position_size = self.risk_manager.calculate_position_size('SPY', 0.8, black_swan_data)
        
        self.assertLess(position_size, 0.01, 
                       "Position size should be near zero during black swan event")
        
        recent_data = black_swan_data['1m'].tail(self.risk_manager.volatility_lookback * 1440)
        returns = recent_data['Close'].pct_change().dropna()
        var_95 = np.percentile(returns, 5)
        es_95 = returns[returns <= var_95].mean()
        
        self.assertLess(es_95, -0.005, 
                       "Expected Shortfall should be strongly negative during black swan")

if __name__ == '__main__':
    unittest.main()
