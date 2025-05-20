"""
Order Book Reconstruction
Reconstructs order book from market data and simulates realistic fills
"""

import numpy as np
import pandas as pd
from datetime import datetime
import heapq
import math

class OrderBookReconstructor:
    def __init__(self, depth=10):
        """
        Initialize Order Book Reconstructor
        
        Parameters:
        - depth: Order book depth to maintain (default: 10)
        """
        self.depth = depth
        self.bids = []  # Max heap for bids (negative price for max heap)
        self.asks = []  # Min heap for asks
        self.bid_volumes = {}  # Price -> Volume mapping
        self.ask_volumes = {}  # Price -> Volume mapping
        self.last_update = None
        self.last_trade = None
        self.symbol = None
    
    def update(self, symbol, price, bid_ask_spread=0.0001, volume_profile="normal"):
        """
        Update order book based on latest price
        
        Parameters:
        - symbol: Trading symbol
        - price: Current market price
        - bid_ask_spread: Spread as a fraction of price (default: 0.0001 or 1 bps)
        - volume_profile: Volume profile type ("normal", "thin", "deep", "imbalanced_bid", "imbalanced_ask")
        
        Returns:
        - Dictionary with updated order book
        """
        self.symbol = symbol
        
        spread_amount = price * bid_ask_spread
        bid_price = price - spread_amount / 2
        ask_price = price + spread_amount / 2
        
        self.bids = []
        self.asks = []
        self.bid_volumes = {}
        self.ask_volumes = {}
        
        if volume_profile == "thin":
            base_volume = price * 0.0001  # 0.01% of price
            volume_decay = 2.0  # Faster decay
        elif volume_profile == "deep":
            base_volume = price * 0.001  # 0.1% of price
            volume_decay = 1.2  # Slower decay
        elif volume_profile == "imbalanced_bid":
            base_volume = price * 0.0005  # 0.05% of price
            volume_decay = 1.5
            bid_multiplier = 2.0
            ask_multiplier = 1.0
        elif volume_profile == "imbalanced_ask":
            base_volume = price * 0.0005  # 0.05% of price
            volume_decay = 1.5
            bid_multiplier = 1.0
            ask_multiplier = 2.0
        else:  # "normal"
            base_volume = price * 0.0005  # 0.05% of price
            volume_decay = 1.5
            bid_multiplier = 1.0
            ask_multiplier = 1.0
        
        base_volume *= np.random.normal(1, 0.1)
        
        for i in range(self.depth):
            level_price = round(bid_price * (1 - i * bid_ask_spread), 8)
            
            level_volume = base_volume * bid_multiplier * (volume_decay ** -i) * np.random.normal(1, 0.2)
            
            heapq.heappush(self.bids, -level_price)  # Negative for max heap
            self.bid_volumes[level_price] = level_volume
        
        for i in range(self.depth):
            level_price = round(ask_price * (1 + i * bid_ask_spread), 8)
            
            level_volume = base_volume * ask_multiplier * (volume_decay ** -i) * np.random.normal(1, 0.2)
            
            heapq.heappush(self.asks, level_price)
            self.ask_volumes[level_price] = level_volume
        
        self.last_update = datetime.now()
        
        return self.get_order_book()
    
    def get_order_book(self):
        """
        Get current order book state
        
        Returns:
        - Dictionary with order book state
        """
        if not self.bids or not self.asks:
            return {
                "symbol": self.symbol,
                "timestamp": datetime.now().isoformat(),
                "bids": [],
                "asks": [],
                "spread": None,
                "mid_price": None
            }
        
        best_bid = -self.bids[0]
        best_ask = self.asks[0]
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        
        bids = [{"price": -price, "volume": self.bid_volumes[-price]} for price in sorted(self.bids)]
        asks = [{"price": price, "volume": self.ask_volumes[price]} for price in sorted(self.asks)]
        
        return {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_bps": (spread / mid_price) * 10000,
            "mid_price": mid_price
        }
    
    def simulate_market_order(self, side, volume):
        """
        Simulate a market order execution
        
        Parameters:
        - side: Order side ("BUY" or "SELL")
        - volume: Order volume
        
        Returns:
        - Dictionary with execution details
        """
        if not self.bids or not self.asks:
            raise ValueError("Order book is empty")
        
        execution_price = 0
        remaining_volume = volume
        levels_used = 0
        
        if side == "BUY":
            ask_prices = sorted(self.asks)
            
            for price in ask_prices:
                available_volume = self.ask_volumes[price]
                
                if remaining_volume <= available_volume:
                    execution_price += price * remaining_volume
                    self.ask_volumes[price] -= remaining_volume
                    remaining_volume = 0
                    levels_used += 1
                    break
                else:
                    execution_price += price * available_volume
                    remaining_volume -= available_volume
                    self.ask_volumes[price] = 0
                    levels_used += 1
            
            self.asks = [price for price in self.asks if self.ask_volumes[price] > 0]
            heapq.heapify(self.asks)
            
        elif side == "SELL":
            bid_prices = [-price for price in sorted(self.bids)]
            
            for price in bid_prices:
                available_volume = self.bid_volumes[price]
                
                if remaining_volume <= available_volume:
                    execution_price += price * remaining_volume
                    self.bid_volumes[price] -= remaining_volume
                    remaining_volume = 0
                    levels_used += 1
                    break
                else:
                    execution_price += price * available_volume
                    remaining_volume -= available_volume
                    self.bid_volumes[price] = 0
                    levels_used += 1
            
            self.bids = [price for price in self.bids if self.bid_volumes[-price] > 0]
            heapq.heapify(self.bids)
            
        else:
            raise ValueError("Side must be 'BUY' or 'SELL'")
        
        filled_volume = volume - remaining_volume
        avg_price = execution_price / filled_volume if filled_volume > 0 else 0
        
        self.last_trade = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "volume": volume,
            "filled_volume": filled_volume,
            "remaining_volume": remaining_volume,
            "avg_price": avg_price,
            "levels_used": levels_used
        }
        
        return self.last_trade
    
    def simulate_limit_order(self, side, price, volume):
        """
        Simulate a limit order placement
        
        Parameters:
        - side: Order side ("BUY" or "SELL")
        - price: Limit price
        - volume: Order volume
        
        Returns:
        - Dictionary with order details
        """
        if side == "BUY":
            if self.asks and price >= self.asks[0]:
                return self.simulate_market_order(side, volume)
            else:
                heapq.heappush(self.bids, -price)
                self.bid_volumes[price] = self.bid_volumes.get(price, 0) + volume
        
        elif side == "SELL":
            if self.bids and price <= -self.bids[0]:
                return self.simulate_market_order(side, volume)
            else:
                heapq.heappush(self.asks, price)
                self.ask_volumes[price] = self.ask_volumes.get(price, 0) + volume
        
        else:
            raise ValueError("Side must be 'BUY' or 'SELL'")
        
        order = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "price": price,
            "volume": volume,
            "type": "LIMIT",
            "status": "OPEN"
        }
        
        return order
    
    def calculate_market_impact(self, side, volume):
        """
        Calculate expected market impact of an order
        
        Parameters:
        - side: Order side ("BUY" or "SELL")
        - volume: Order volume
        
        Returns:
        - Dictionary with market impact metrics
        """
        if not self.bids or not self.asks:
            raise ValueError("Order book is empty")
        
        best_bid = -self.bids[0]
        best_ask = self.asks[0]
        mid_price = (best_bid + best_ask) / 2
        
        temp_reconstructor = OrderBookReconstructor(self.depth)
        temp_reconstructor.bids = self.bids.copy()
        temp_reconstructor.asks = self.asks.copy()
        temp_reconstructor.bid_volumes = self.bid_volumes.copy()
        temp_reconstructor.ask_volumes = self.ask_volumes.copy()
        temp_reconstructor.symbol = self.symbol
        
        execution = temp_reconstructor.simulate_market_order(side, volume)
        
        impact_amount = execution["avg_price"] - mid_price if side == "BUY" else mid_price - execution["avg_price"]
        impact_bps = (impact_amount / mid_price) * 10000
        
        return {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "volume": volume,
            "mid_price": mid_price,
            "expected_avg_price": execution["avg_price"],
            "market_impact": impact_amount,
            "market_impact_bps": impact_bps,
            "levels_used": execution["levels_used"],
            "filled_volume": execution["filled_volume"],
            "remaining_volume": execution["remaining_volume"]
        }
    
    def get_liquidity_metrics(self):
        """
        Calculate liquidity metrics for the order book
        
        Returns:
        - Dictionary with liquidity metrics
        """
        if not self.bids or not self.asks:
            return {
                "symbol": self.symbol,
                "timestamp": datetime.now().isoformat(),
                "liquidity_score": 0,
                "bid_depth": 0,
                "ask_depth": 0,
                "imbalance": 0
            }
        
        best_bid = -self.bids[0]
        best_ask = self.asks[0]
        mid_price = (best_bid + best_ask) / 2
        
        total_bid_volume = sum(self.bid_volumes.values())
        total_ask_volume = sum(self.ask_volumes.values())
        
        near_bid_volume = sum(self.bid_volumes[price] for price in self.bid_volumes 
                             if price >= mid_price * 0.995)
        near_ask_volume = sum(self.ask_volumes[price] for price in self.ask_volumes 
                             if price <= mid_price * 1.005)
        
        liquidity_score = (near_bid_volume + near_ask_volume) / mid_price
        
        imbalance = (total_ask_volume - total_bid_volume) / (total_ask_volume + total_bid_volume)
        
        return {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "liquidity_score": liquidity_score,
            "bid_depth": total_bid_volume,
            "ask_depth": total_ask_volume,
            "near_bid_volume": near_bid_volume,
            "near_ask_volume": near_ask_volume,
            "imbalance": imbalance,
            "spread_bps": ((best_ask - best_bid) / mid_price) * 10000
        }
