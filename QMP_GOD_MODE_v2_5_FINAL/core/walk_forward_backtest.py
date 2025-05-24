import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WalkForwardBacktester:
    def __init__(self, train_days=180, test_days=30):
        self.train_days = train_days
        self.test_days = test_days
        
    def run_backtest(self, strategy_engine, data, start_date, end_date):
        """Run walk-forward backtest preventing look-ahead bias"""
        results = []
        current_date = start_date + timedelta(days=self.train_days)
        
        while current_date < end_date:
            train_start = current_date - timedelta(days=self.train_days)
            train_end = current_date - timedelta(days=1)  # Add 1-day buffer gap
            test_start = current_date
            test_end = min(current_date + timedelta(days=self.test_days), end_date)
            
            train_data = self._extract_data_range(data, train_start, train_end)
            test_data = self._extract_data_range(data, test_start, test_end)
            
            if len(train_data) < self.train_days // 2:  # Minimum data requirement
                current_date = test_end
                continue
                
            strategy_engine.fit(train_data)
            
            test_results = strategy_engine.predict(test_data)
            results.extend(test_results)
            
            current_date = test_end
            
        return results
        
    def _extract_data_range(self, data, start_date, end_date):
        """Extract data within date range ensuring no future leakage"""
        range_data = {}
        for timeframe, df in data.items():
            mask = (df.index >= start_date) & (df.index < end_date)
            range_data[timeframe] = df.loc[mask].copy()
        return range_data
