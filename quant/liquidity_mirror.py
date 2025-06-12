
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
        - bids: List of [price, size] bid orders or Dictionary of bid prices and volumes
        - asks: List of [price, size] ask orders or Dictionary of ask prices and volumes
        
        Returns:
        - Dictionary with liquidity analysis
        """
        if not bids or not asks:
            return {'imbalance': 'insufficient_data', 'ratio': 0, 'signal': None}
        
        if isinstance(bids, list):
            total_bid_size = sum([order[1] for order in bids[:self.depth_levels]])
            total_ask_size = sum([order[1] for order in asks[:self.depth_levels]])
        else:
            total_bid_size = sum(bids.values())
            total_ask_size = sum(asks.values())
        
        if total_ask_size == 0:
            return {'imbalance': 'no_asks', 'ratio': float('inf'), 'signal': 'STRONG_BUY'}
        
        bid_ask_ratio = total_bid_size / total_ask_size
        
        institutional_flow = self._detect_institutional_flow(bids, asks)
        
        self.historical_imbalances.append({
            'timestamp': datetime.now().isoformat(),
            'ratio': bid_ask_ratio,
            'bid_volume': total_bid_size,
            'ask_volume': total_ask_size
        })
        
        if len(self.historical_imbalances) > 100:
            self.historical_imbalances = self.historical_imbalances[-100:]
        
        if bid_ask_ratio > self.min_imbalance:
            return {
                'imbalance': 'hidden_bids',
                'ratio': bid_ask_ratio,
                'signal': 'BUY',
                'strength': min(1.0, bid_ask_ratio / (self.min_imbalance * 2)),
                'total_bid_size': total_bid_size,
                'total_ask_size': total_ask_size,
                'institutional_flow': institutional_flow
            }
        elif bid_ask_ratio < (1 / self.min_imbalance):
            return {
                'imbalance': 'hidden_asks',
                'ratio': bid_ask_ratio,
                'signal': 'SELL',
                'strength': min(1.0, (1/bid_ask_ratio) / (self.min_imbalance * 2)),
                'total_bid_size': total_bid_size,
                'total_ask_size': total_ask_size,
                'institutional_flow': institutional_flow
            }
        else:
            return {
                'imbalance': 'balanced',
                'ratio': bid_ask_ratio,
                'signal': None,
                'strength': 0,
                'total_bid_size': total_bid_size,
                'total_ask_size': total_ask_size,
                'institutional_flow': institutional_flow
            }
            
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
    
    def _detect_institutional_flow(self, bids, asks):
        """
        Detect institutional order flow patterns
        
        Parameters:
        - bids: List of [price, size] bid orders or Dictionary
        - asks: List of [price, size] ask orders or Dictionary
        
        Returns:
        - Dictionary with institutional flow analysis
        """
        if not bids or not asks:
            return {'detected': False, 'type': None, 'confidence': 0.0}
        
        if isinstance(bids, list):
            bid_sizes = [order[1] for order in bids[:self.depth_levels]]
            ask_sizes = [order[1] for order in asks[:self.depth_levels]]
        else:
            bid_sizes = list(bids.values())[:self.depth_levels]
            ask_sizes = list(asks.values())[:self.depth_levels]
        
        if not bid_sizes or not ask_sizes:
            return {'detected': False, 'type': None, 'confidence': 0.0}
        
        large_bid_threshold = np.percentile(bid_sizes, 90) if bid_sizes else 0
        large_ask_threshold = np.percentile(ask_sizes, 90) if ask_sizes else 0
        
        large_bids = [size for size in bid_sizes if size >= large_bid_threshold]
        large_asks = [size for size in ask_sizes if size >= large_ask_threshold]
        
        iceberg_bids = self._detect_iceberg_orders(bids)
        iceberg_asks = self._detect_iceberg_orders(asks)
        
        if len(large_bids) > len(large_asks) * 1.5 or iceberg_bids > iceberg_asks:
            return {
                'detected': True,
                'type': 'institutional_buying',
                'confidence': min(1.0, (len(large_bids) + iceberg_bids) / 10.0),
                'large_orders': len(large_bids),
                'iceberg_orders': iceberg_bids
            }
        elif len(large_asks) > len(large_bids) * 1.5 or iceberg_asks > iceberg_bids:
            return {
                'detected': True,
                'type': 'institutional_selling',
                'confidence': min(1.0, (len(large_asks) + iceberg_asks) / 10.0),
                'large_orders': len(large_asks),
                'iceberg_orders': iceberg_asks
            }
        else:
            return {
                'detected': False,
                'type': 'retail_dominated',
                'confidence': 0.5,
                'large_orders': max(len(large_bids), len(large_asks)),
                'iceberg_orders': max(iceberg_bids, iceberg_asks)
            }
    
    def _detect_iceberg_orders(self, orders):
        """
        Detect potential iceberg orders by analyzing size patterns
        
        Parameters:
        - orders: List of [price, size] orders or Dictionary
        
        Returns:
        - Number of potential iceberg orders detected
        """
        if isinstance(orders, list):
            if len(orders) < 5:
                return 0
            sizes = [order[1] for order in orders[:10]]
        else:
            if len(orders) < 5:
                return 0
            sizes = list(orders.values())[:10]
        
        round_number_count = sum(1 for size in sizes if size % 100 == 0 or size % 1000 == 0)
        
        size_clusters = {}
        for size in sizes:
            rounded_size = round(size, -2)
            size_clusters[rounded_size] = size_clusters.get(rounded_size, 0) + 1
        
        repeated_sizes = sum(1 for count in size_clusters.values() if count > 1)
        
        iceberg_score = round_number_count + repeated_sizes
        
        return min(iceberg_score, len(sizes))
    
    def analyze_order_book_depth(self, bids, asks, price_levels=20):
        """
        Advanced order book depth analysis
        
        Parameters:
        - bids: List of [price, size] bid orders
        - asks: List of [price, size] ask orders  
        - price_levels: Number of price levels to analyze
        
        Returns:
        - Dictionary with depth analysis
        """
        if not bids or not asks:
            return {'status': 'insufficient_data'}
        
        bid_depth = bids[:price_levels]
        ask_depth = asks[:price_levels]
        
        bid_prices = [order[0] for order in bid_depth]
        ask_prices = [order[0] for order in ask_depth]
        bid_sizes = [order[1] for order in bid_depth]
        ask_sizes = [order[1] for order in ask_depth]
        
        spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0
        mid_price = (bid_prices[0] + ask_prices[0]) / 2 if bid_prices and ask_prices else 0
        
        cumulative_bid_size = np.cumsum(bid_sizes)
        cumulative_ask_size = np.cumsum(ask_sizes)
        
        support_levels = []
        resistance_levels = []
        
        for i, (price, size) in enumerate(bid_depth):
            if size > np.mean(bid_sizes) * 2:
                support_levels.append({
                    'price': price,
                    'size': size,
                    'cumulative_size': cumulative_bid_size[i],
                    'distance_from_mid': abs(price - mid_price) / mid_price if mid_price > 0 else 0
                })
        
        for i, (price, size) in enumerate(ask_depth):
            if size > np.mean(ask_sizes) * 2:
                resistance_levels.append({
                    'price': price,
                    'size': size,
                    'cumulative_size': cumulative_ask_size[i],
                    'distance_from_mid': abs(price - mid_price) / mid_price if mid_price > 0 else 0
                })
        
        liquidity_score = min(1.0, (sum(bid_sizes) + sum(ask_sizes)) / 1000000)
        
        return {
            'spread': spread,
            'spread_bps': (spread / mid_price * 10000) if mid_price > 0 else 0,
            'mid_price': mid_price,
            'total_bid_liquidity': sum(bid_sizes),
            'total_ask_liquidity': sum(ask_sizes),
            'liquidity_score': liquidity_score,
            'support_levels': support_levels[:5],
            'resistance_levels': resistance_levels[:5],
            'bid_ask_imbalance': (sum(bid_sizes) - sum(ask_sizes)) / (sum(bid_sizes) + sum(ask_sizes)) if (sum(bid_sizes) + sum(ask_sizes)) > 0 else 0
        }
