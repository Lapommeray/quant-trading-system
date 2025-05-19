"""
Fill Engine with Slippage and Latency Simulation
"""

import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime
from .slippage_model import SlippageModel, slippage_simulator

class FillEngine:
    def __init__(self, slippage_enabled=True, latency_ms=200, order_book_simulation=True):
        """
        Initialize the fill engine with parameters
        
        Parameters:
        - slippage_enabled: Whether to enable slippage simulation
        - latency_ms: Simulated latency in milliseconds
        - order_book_simulation: Whether to simulate order book dynamics
        """
        self.slippage_enabled = slippage_enabled
        self.latency_ms = latency_ms
        self.order_book_simulation = order_book_simulation
        
        self.slippage_model = SlippageModel(
            enable_slippage=slippage_enabled,
            latency_ms=latency_ms
        )
        
        self.trades = []
    
    def execute_order(self, symbol, direction, price, volume, timestamp=None):
        """
        Execute an order with realistic fill simulation
        
        Parameters:
        - symbol: Trading symbol
        - direction: 'BUY' or 'SELL'
        - price: Requested price
        - volume: Trading volume
        - timestamp: Optional timestamp (defaults to current time)
        
        Returns:
        - Dictionary with fill details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        actual_latency = self.slippage_model.simulate_latency()
        
        current_price = self._get_current_price(symbol, price, direction, actual_latency)
        
        if self.slippage_enabled:
            fill_price = self._apply_slippage(symbol, current_price, volume, direction)
        else:
            fill_price = current_price
        
        if self.order_book_simulation:
            fill_price = self._simulate_order_book(symbol, fill_price, volume, direction)
        
        trade = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'symbol': symbol,
            'direction': direction,
            'requested_price': price,
            'current_price': current_price,
            'fill_price': fill_price,
            'volume': volume,
            'slippage_bps': (abs(fill_price - price) / price) * 10000,
            'latency_ms': actual_latency
        }
        
        self.trades.append(trade)
        
        return trade
    
    def _get_current_price(self, symbol, original_price, direction, latency_ms):
        """
        Simulate price movement during latency period
        
        Parameters:
        - symbol: Trading symbol
        - original_price: Original price at order time
        - direction: 'BUY' or 'SELL'
        - latency_ms: Latency in milliseconds
        
        Returns:
        - Updated price after latency
        """
        
        volatility_map = {
            'BTCUSD': 0.0003,  # Higher volatility for crypto
            'ETHUSD': 0.0004,
            'XAUUSD': 0.0001,  # Lower for gold
            'EURUSD': 0.00005, # Very low for forex
            'DIA': 0.0001,     # Medium for ETFs
            'QQQ': 0.00015,
            'SPY': 0.0001
        }
        
        vol_per_second = volatility_map.get(symbol, 0.0002)
        
        vol_during_latency = vol_per_second * (latency_ms / 1000.0)
        
        adverse_selection_factor = 0.2  # Slight bias
        
        if direction == 'BUY':
            mean_change = vol_during_latency * adverse_selection_factor
        else:
            mean_change = -vol_during_latency * adverse_selection_factor
        
        price_change_pct = np.random.normal(mean_change, vol_during_latency)
        
        new_price = original_price * (1 + price_change_pct)
        
        return new_price
    
    def _apply_slippage(self, symbol, price, volume, direction):
        """Apply slippage to the price"""
        return self.slippage_model.calculate_slippage(symbol, price, volume, direction)
    
    def _simulate_order_book(self, symbol, price, volume, direction):
        """
        Simulate order book dynamics for more realistic fills
        
        Parameters:
        - symbol: Trading symbol
        - price: Current price
        - volume: Trading volume
        - direction: 'BUY' or 'SELL'
        
        Returns:
        - Adjusted price after order book simulation
        """
        
        liquidity_map = {
            'BTCUSD': 100,    # Good liquidity
            'ETHUSD': 80,     # Slightly less
            'XAUUSD': 120,    # Very liquid
            'EURUSD': 200,    # Extremely liquid
            'DIA': 90,        # Good liquidity
            'QQQ': 100,
            'SPY': 150        # Very liquid
        }
        
        liquidity = liquidity_map.get(symbol, 100)
        
        market_impact_pct = (volume / liquidity) ** 0.5 * 0.0001  # Square root function to model diminishing impact
        
        if direction == 'BUY':
            adjusted_price = price * (1 + market_impact_pct)
        else:
            adjusted_price = price * (1 - market_impact_pct)
        
        return adjusted_price
    
    def get_trades(self):
        """Get all recorded trades"""
        return self.trades
    
    def save_trades_csv(self, output_file="trades.csv"):
        """Save trades to CSV file"""
        if not self.trades:
            print("No trades to save")
            return False
        
        df = pd.DataFrame(self.trades)
        df.to_csv(output_file, index=False)
        
        print(f"Trades saved to {output_file}")
        return True
    
    def generate_costs_log(self, output_file="costs.log"):
        """Generate costs log using the slippage model"""
        return self.slippage_model.generate_costs_log(output_file)


def process_fill(raw_price, volume, symbol="BTCUSD", direction="BUY", 
                slippage=True, latency_ms=200, order_book=True):
    """
    Process a fill with realistic market simulation
    
    Parameters:
    - raw_price: Original price
    - volume: Trading volume
    - symbol: Trading symbol
    - direction: 'BUY' or 'SELL'
    - slippage: Whether to enable slippage
    - latency_ms: Latency in milliseconds
    - order_book: Whether to simulate order book dynamics
    
    Returns:
    - Adjusted price with all simulations applied
    """
    engine = FillEngine(
        slippage_enabled=slippage,
        latency_ms=latency_ms,
        order_book_simulation=order_book
    )
    
    result = engine.execute_order(symbol, direction, raw_price, volume)
    
    return result['fill_price']
