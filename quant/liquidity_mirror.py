
import numpy as np
import pandas as pd
from datetime import datetime
import json

class LiquidityMirror:
    """
    Liquidity Mirror Scanner
    
    Detects hidden institutional order blocks by analyzing order book data
    and identifying significant imbalances in liquidity.
    """
    
    def __init__(self, min_imbalance=2.0, depth_levels=10):
        """
        Initialize the Liquidity Mirror Scanner
        
        Parameters:
        - min_imbalance: Minimum bid/ask ratio to detect imbalance (default: 2.0)
        - depth_levels: Number of price levels to analyze (default: 10)
        """
        self.min_imbalance = min_imbalance
        self.depth_levels = depth_levels
        self.historical_imbalances = []
        
    def scan_liquidity(self, bids, asks):
        """
        Identifies liquidity mirrors:
        - bid_ask_ratio > min_imbalance → Hidden bids
        - bid_ask_ratio < 1/min_imbalance → Hidden asks
        
        Parameters:
        - bids: Dictionary of bid prices and volumes {price: volume}
        - asks: Dictionary of ask prices and volumes {price: volume}
        
        Returns:
        - Tuple of (signal, ratio)
        """
        bid_vol = sum(bids.values())
        ask_vol = sum(asks.values())
        
        if ask_vol == 0:
            ratio = float('inf')
        else:
            ratio = bid_vol / ask_vol
            
        self.historical_imbalances.append({
            'timestamp': datetime.now().isoformat(),
            'ratio': ratio,
            'bid_volume': bid_vol,
            'ask_volume': ask_vol
        })
        
        if len(self.historical_imbalances) > 100:
            self.historical_imbalances = self.historical_imbalances[-100:]
        
        if ratio > self.min_imbalance:
            return "HIDDEN BIDS DETECTED", ratio
        elif ratio < 1/self.min_imbalance:
            return "HIDDEN ASKS DETECTED", ratio
        else:
            return "NO STRONG IMBALANCE", ratio
            
    def analyze_order_book(self, order_book_data):
        """
        Analyze full order book data
        
        Parameters:
        - order_book_data: Dictionary with 'bids' and 'asks' arrays of [price, volume] pairs
        
        Returns:
        - Dictionary with analysis results
        """
        if 'bids' not in order_book_data or 'asks' not in order_book_data:
            raise ValueError("Order book data must contain 'bids' and 'asks' arrays")
            
        bids = {float(bid[0]): float(bid[1]) for bid in order_book_data['bids'][:self.depth_levels]}
        asks = {float(ask[0]): float(ask[1]) for ask in order_book_data['asks'][:self.depth_levels]}
        
        signal, ratio = self.scan_liquidity(bids, asks)
        
        bid_prices = sorted(bids.keys(), reverse=True)
        ask_prices = sorted(asks.keys())
        
        spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0
        
        bid_clusters = self._find_liquidity_clusters(bids)
        ask_clusters = self._find_liquidity_clusters(asks)
        
        return {
            'signal': signal,
            'ratio': ratio,
            'spread': spread,
            'bid_clusters': bid_clusters,
            'ask_clusters': ask_clusters,
            'timestamp': datetime.now().isoformat()
        }
        
    def _find_liquidity_clusters(self, orders):
        """
        Find clusters of liquidity in the order book
        
        Parameters:
        - orders: Dictionary of prices and volumes {price: volume}
        
        Returns:
        - List of clusters with price and volume
        """
        if not orders:
            return []
            
        prices = sorted(orders.keys())
        
        clusters = []
        current_cluster = {
            'start_price': prices[0],
            'end_price': prices[0],
            'total_volume': orders[prices[0]]
        }
        
        for i in range(1, len(prices)):
            if prices[i] - prices[i-1] < 0.01 * prices[i-1]:  # Within 1%
                current_cluster['end_price'] = prices[i]
                current_cluster['total_volume'] += orders[prices[i]]
            else:
                clusters.append(current_cluster)
                current_cluster = {
                    'start_price': prices[i],
                    'end_price': prices[i],
                    'total_volume': orders[prices[i]]
                }
                
        clusters.append(current_cluster)
        
        clusters.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return clusters
