"""
Slippage Model for Realistic Market Simulation
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime

class SlippageModel:
    def __init__(self, enable_slippage=True, default_spread_bps=2.0, spread_map=None, latency_ms=200):
        """
        Initialize the slippage model with parameters
        
        Parameters:
        - enable_slippage: Whether to enable slippage simulation (default: True)
        - default_spread_bps: Default spread in basis points (default: 2.0)
        - spread_map: Dictionary mapping symbols to their specific spreads in bps
        - latency_ms: Simulated latency in milliseconds (default: 200ms)
        """
        self.enable_slippage = enable_slippage
        self.default_spread_bps = default_spread_bps
        self.latency_ms = latency_ms
        
        self.default_spread_map = {
            'BTCUSD': 1.5,  # Bitcoin typically has tight spreads on major exchanges
            'ETHUSD': 2.0,  # Ethereum slightly wider
            'XAUUSD': 3.0,  # Gold can have wider spreads
            'EURUSD': 0.2,  # Forex majors have very tight spreads
            'DIA': 1.0,     # ETFs typically have modest spreads
            'QQQ': 1.0,
            'SPY': 0.5
        }
        
        self.spread_map = self.default_spread_map.copy()
        if spread_map:
            self.spread_map.update(spread_map)
            
        self.slippage_stats = {
            'total_slippage_bps': 0,
            'count': 0,
            'worst_slippage_bps': 0,
            'worst_slippage_symbol': None,
            'total_latency_ms': 0,
            'max_latency_ms': 0,
            'spreads': {},
            'timestamps': []
        }
    
    def get_spread_bps(self, symbol):
        """Get spread in basis points for a symbol"""
        return self.spread_map.get(symbol, self.default_spread_bps)
    
    def calculate_slippage(self, symbol, price, volume, direction):
        """
        Calculate slippage based on symbol, price, volume and direction
        
        Parameters:
        - symbol: Trading symbol
        - price: Raw/ideal price
        - volume: Trading volume
        - direction: 'BUY' or 'SELL'
        
        Returns:
        - Adjusted price with slippage
        """
        if not self.enable_slippage:
            return price
        
        spread_bps = self.get_spread_bps(symbol)
        
        half_spread = price * (spread_bps / 10000)
        
        volume_factor = max(1.0, 1.0 + np.log10(max(1, volume))/10)
        
        if direction == 'BUY':
            slippage = half_spread * volume_factor
            adjusted_price = price + slippage
        else:
            slippage = half_spread * volume_factor
            adjusted_price = price - slippage
        
        slippage_bps = (abs(adjusted_price - price) / price) * 10000
        self.slippage_stats['total_slippage_bps'] += slippage_bps
        self.slippage_stats['count'] += 1
        
        if slippage_bps > self.slippage_stats['worst_slippage_bps']:
            self.slippage_stats['worst_slippage_bps'] = slippage_bps
            self.slippage_stats['worst_slippage_symbol'] = symbol
        
        if symbol not in self.slippage_stats['spreads']:
            self.slippage_stats['spreads'][symbol] = []
        
        self.slippage_stats['spreads'][symbol].append(spread_bps)
        
        return adjusted_price
    
    def simulate_latency(self):
        """
        Simulate network latency for order execution
        
        Returns:
        - Actual latency in milliseconds
        """
        if not self.enable_slippage:
            return 0
        
        base_latency = self.latency_ms
        
        std_dev = base_latency * 0.3
        actual_latency = max(1, np.random.normal(base_latency, std_dev))
        
        if np.random.random() < 0.01:
            actual_latency *= 5
            
        time.sleep(actual_latency / 1000.0)
        
        self.slippage_stats['total_latency_ms'] += actual_latency
        self.slippage_stats['max_latency_ms'] = max(
            self.slippage_stats['max_latency_ms'], actual_latency
        )
        
        return actual_latency
    
    def get_stats(self):
        """Get slippage statistics"""
        stats = self.slippage_stats.copy()
        
        if stats['count'] > 0:
            stats['avg_slippage_bps'] = stats['total_slippage_bps'] / stats['count']
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['count']
        else:
            stats['avg_slippage_bps'] = 0
            stats['avg_latency_ms'] = 0
        
        avg_spreads = {}
        for symbol, spreads in stats['spreads'].items():
            if spreads:
                avg_spreads[symbol] = sum(spreads) / len(spreads)
        
        stats['avg_spreads'] = avg_spreads
        
        return stats
    
    def log_trade(self, symbol, raw_price, adjusted_price, volume, direction, latency):
        """
        Log trade details for cost analysis
        
        Parameters:
        - symbol: Trading symbol
        - raw_price: Original price before slippage
        - adjusted_price: Price after slippage
        - volume: Trading volume
        - direction: 'BUY' or 'SELL'
        - latency: Latency in milliseconds
        """
        timestamp = datetime.now().isoformat()
        
        trade_info = {
            'timestamp': timestamp,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'raw_price': raw_price,
            'adjusted_price': adjusted_price,
            'slippage_bps': (abs(adjusted_price - raw_price) / raw_price) * 10000,
            'latency_ms': latency
        }
        
        self.slippage_stats['timestamps'].append(trade_info)
        
        return trade_info
    
    def generate_costs_log(self, output_file="costs.log"):
        """Generate a costs log file with slippage and latency statistics"""
        stats = self.get_stats()
        
        with open(output_file, 'w') as f:
            f.write("========== TRADING COSTS LOG ==========\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total trades: {stats['count']}\n")
            f.write(f"Average slippage: {stats['avg_slippage_bps']:.2f} bps\n")
            f.write(f"Worst slippage: {stats['worst_slippage_bps']:.2f} bps ({stats['worst_slippage_symbol']})\n")
            f.write(f"Average latency: {stats['avg_latency_ms']:.2f} ms\n")
            f.write(f"Maximum latency: {stats['max_latency_ms']:.2f} ms\n\n")
            
            
            f.write("Average spreads by symbol:\n")
            for symbol, avg_spread in stats['avg_spreads'].items():
                f.write(f"  {symbol}: {avg_spread:.2f} bps\n")
            
            f.write("\n========== DETAILED TRADE COSTS ==========\n\n")
            for trade in stats['timestamps']:
                f.write(f"[{trade['timestamp']}] {trade['symbol']} {trade['direction']} {trade['volume']} @ "
                        f"{trade['raw_price']:.8f} -> {trade['adjusted_price']:.8f} "
                        f"(Slippage: {trade['slippage_bps']:.2f} bps, Latency: {trade['latency_ms']:.2f} ms)\n")
        
        print(f"Costs log saved to {output_file}")
        return stats


def slippage_simulator(raw_price, volume, symbol="BTCUSD", direction="BUY", spread_bps=None, latency_ms=None):
    """
    Simulate slippage and latency for a trade
    
    Parameters:
    - raw_price: Original price
    - volume: Trading volume
    - symbol: Trading symbol (default: BTCUSD)
    - direction: 'BUY' or 'SELL' (default: BUY)
    - spread_bps: Override default spread in basis points
    - latency_ms: Override default latency in milliseconds
    
    Returns:
    - Adjusted price with slippage
    """
    spread_map = None
    if spread_bps is not None:
        spread_map = {symbol: spread_bps}
    
    model = SlippageModel(
        enable_slippage=True,
        spread_map=spread_map,
        latency_ms=latency_ms if latency_ms is not None else 200
    )
    
    actual_latency = model.simulate_latency()
    
    adjusted_price = model.calculate_slippage(symbol, raw_price, volume, direction)
    
    model.log_trade(symbol, raw_price, adjusted_price, volume, direction, actual_latency)
    
    return adjusted_price
