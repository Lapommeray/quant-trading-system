
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataFetcher:
    """Data fetcher for emergency snapshot and market data retrieval"""
    
    def __init__(self):
        self.logger = logging.getLogger("DataFetcher")
        
    def get_latest_data(self, symbols=None):
        """Get latest market data for specified symbols"""
        if symbols is None:
            symbols = ["SPY", "QQQ", "BTC"]
            
        data = []
        for symbol in symbols:
            price_data = get_latest_price(symbol)
            data.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'open': price_data['open'],
                'high': price_data['high'], 
                'low': price_data['low'],
                'close': price_data['close']
            })
            
        return pd.DataFrame(data)

def get_latest_price(asset_symbol='XAUUSD'):
    """
    Simulated real-time OHLC data.
    Replace this later with real-time API if needed.
    """
    import random

    base_price = 2000.0 if asset_symbol == 'XAUUSD' else 30000.0
    if asset_symbol in ['SPY', 'QQQ']:
        base_price = 400.0
    elif asset_symbol == 'BTC':
        base_price = 50000.0
        
    variation = random.uniform(-5, 5)

    open_price = base_price + variation
    high_price = open_price + random.uniform(1, 10)
    low_price = open_price - random.uniform(1, 10)
    close_price = open_price + random.uniform(-5, 5)

    return {
        'open': round(open_price, 2),
        'high': round(high_price, 2),
        'low': round(low_price, 2),
        'close': round(close_price, 2)
    }
