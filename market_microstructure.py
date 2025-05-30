import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

class LimitOrderBook:
    def __init__(self, tick_size=0.01):
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.tick_size = tick_size
        self.mid_price_history = []
        self.spread_history = []
    
    def process_order(self, order):
        """Handles limit/market orders with queue position tracking"""
        if order['side'] == 'bid':
            book = self.bids
            opposite_book = self.asks
        else:
            book = self.asks
            opposite_book = self.bids
        
        remaining_qty = order['quantity']
        executed = []
        
        while (remaining_qty > 0 and opposite_book and 
               ((order['side'] == 'bid' and order['price'] >= opposite_book.iloc[0]) or
                (order['side'] == 'ask' and order['price'] <= opposite_book.iloc[-1]))):
            best_opposite = opposite_book.iloc[0] if order['side'] == 'bid' else opposite_book.iloc[-1]
            match_qty = min(remaining_qty, opposite_book[best_opposite])
            
            executed.append({
                'price': best_opposite,
                'quantity': match_qty,
                'side': 'sell' if order['side'] == 'bid' else 'buy'
            })
            
            opposite_book[best_opposite] -= match_qty
            if opposite_book[best_opposite] <= 0:
                del opposite_book[best_opposite]
            
            remaining_qty -= match_qty
        
        if remaining_qty > 0 and order['type'] == 'limit':
            book[order['price']] = book.get(order['price'], 0) + remaining_qty
        
        if self.bids and self.asks:
            self.mid_price_history.append((self.bids.peekitem(-1)[0] + self.asks.peekitem(0)[0]) / 2)
            self.spread_history.append(self.asks.peekitem(0)[0] - self.bids.peekitem(-1)[0])
        
        return executed

class VPINCalculator:
    """Volume-synchronized probability of informed trading"""
    def __init__(self, bucket_volume=10000, window=50):
        self.bucket_volume = bucket_volume
        self.window = window
        self.buy_volumes = []
        self.sell_volumes = []
    
    def add_trade(self, price, volume, side):
        """side: 1 for buy, -1 for sell"""
        if side == 1:
            self.buy_volumes.append(volume)
        else:
            self.sell_volumes.append(volume)
    
    def calculate(self):
        if len(self.buy_volumes) < self.window or len(self.sell_volumes) < self.window:
            return np.nan
        
        buy_vol = np.array(self.buy_volumes[-self.window:])
        sell_vol = np.array(self.sell_volumes[-self.window:])
        return np.abs(buy_vol - sell_vol).sum() / (self.bucket_volume * self.window)
