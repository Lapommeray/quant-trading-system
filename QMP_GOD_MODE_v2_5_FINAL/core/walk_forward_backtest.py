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
            # Define train and test windows with buffer gap
            train_start = current_date - timedelta(days=self.train_days)
            train_end = current_date - timedelta(days=1)  # Add 1-day buffer gap
            test_start = current_date
            test_end = min(current_date + timedelta(days=self.test_days), end_date)
            
            train_data = self._extract_data_range(data, train_start, train_end)
            test_data = self._extract_data_range(data, test_start, test_end)
            
            if any(df.empty for df in train_data.values()):
                current_date = test_end
                continue
                
            strategy_engine.fit(train_data)
            
            if not any(df.empty for df in test_data.values()):
                test_results = strategy_engine.predict(test_data)
                if test_results:  # Only extend if we got valid results
                    results.extend(test_results)
            
            current_date = test_end
            
        return results
        
    def _extract_data_range(self, data, start_date, end_date):
        """Extract data within date range using integer indexing to prevent leakage"""
        range_data = {}
        for timeframe, df in data.items():
            try:
                start_idx = df.index.get_indexer([start_date], method='bfill')[0]
                end_idx = df.index.get_indexer([end_date], method='ffill')[0]
                
                assert end_idx > start_idx, f"Index error: end_idx ({end_idx}) <= start_idx ({start_idx})"
                
                actual_start_date = df.index[start_idx]
                actual_end_date = df.index[end_idx-1] if end_idx > 0 else None
                
                if actual_end_date is not None:
                    assert actual_start_date <= actual_end_date, f"Date order error: start ({actual_start_date}) > end ({actual_end_date})"
                
                range_data[timeframe] = df.iloc[start_idx:end_idx].copy()
                
            except (IndexError, KeyError) as e:
                print(f"Warning: Data extraction failed for {timeframe}: {e}")
                range_data[timeframe] = pd.DataFrame(columns=df.columns)
                
        return range_data
