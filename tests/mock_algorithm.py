"""
Mock Algorithm for Testing

Provides a mock QuantConnect algorithm instance for testing purposes.
"""

from datetime import datetime
import pandas as pd
import numpy as np

class MockAlgorithm:
    """Mock algorithm for testing purposes"""
    
    def __init__(self):
        self.Time = datetime.now()
        self.Portfolio = MockPortfolio()
        self.Securities = {}
        self.debug_messages = []
        self.log_messages = []
        self.error_messages = []
        
    def Debug(self, message):
        """Mock debug logging"""
        self.debug_messages.append(message)
        
    def Log(self, message):
        """Mock info logging"""
        self.log_messages.append(message)
        
    def Error(self, message):
        """Mock error logging"""
        self.error_messages.append(message)
        
    def History(self, symbol, periods, resolution=None):
        """Mock history data"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='1min')
        data = pd.DataFrame({
            'Open': np.random.normal(100, 2, periods),
            'High': np.random.normal(102, 2, periods),
            'Low': np.random.normal(98, 2, periods),
            'Close': np.random.normal(101, 2, periods),
            'Volume': np.random.normal(1000000, 200000, periods)
        }, index=dates)
        return data

class MockPortfolio:
    """Mock portfolio for testing"""
    
    def __init__(self):
        self.TotalPortfolioValue = 100000
        self.Cash = 50000
        self.TotalMarginUsed = 0
