import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.walk_forward_backtest import WalkForwardBacktester

class TestWalkForward(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2024-01-01', periods=300, freq='D')
        self.data = {
            '1d': pd.DataFrame({
                'Close': np.random.normal(100, 1, 300),
                'Open': np.random.normal(100, 1, 300),
                'High': np.random.normal(101, 1, 300),
                'Low': np.random.normal(99, 1, 300),
                'Volume': np.random.uniform(1000, 10000, 300)
            }, index=dates)
        }
        
        self.backtester = WalkForwardBacktester(train_days=180, test_days=30)
        
    def test_no_data_leakage(self):
        """Test that walk-forward validation prevents data leakage with paranoid checks"""
        current_date = self.data['1d'].index[180]
        train_start = current_date - timedelta(days=180)
        train_end = current_date - timedelta(days=1)  # Buffer gap
        test_start = current_date
        test_end = current_date + timedelta(days=30)
        
        train_data = self.backtester._extract_data_range(self.data, train_start, train_end)
        test_data = self.backtester._extract_data_range(self.data, test_start, test_end)
        
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
                               
    def test_full_backtest_no_leakage(self):
        """Test that full backtest run maintains data separation"""
        class MockStrategy:
            def fit(self, data):
                self.train_dates = set(data['1d'].index)
                
            def predict(self, data):
                self.test_dates = set(data['1d'].index)
                assert len(self.train_dates.intersection(self.test_dates)) == 0, "Data leakage detected!"
                return [{'date': d, 'prediction': 1.0} for d in data['1d'].index]
        
        strategy = MockStrategy()
        start_date = self.data['1d'].index[0]
        end_date = self.data['1d'].index[-1]
        
        results = self.backtester.run_backtest(strategy, self.data, start_date, end_date)
        
        self.assertGreater(len(results), 0, "Backtest produced no results")

if __name__ == '__main__':
    unittest.main()
