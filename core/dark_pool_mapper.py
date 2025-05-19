"""
Dark Pool Liquidity Mapper
Maps institutional dark pool liquidity and flow to trading signals
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

class DarkPoolMapper:
    def __init__(self, api_key=None):
        """
        Initialize Dark Pool Liquidity Mapper
        
        Parameters:
        - api_key: Optional API key for FlowAlgo (can be set via environment variable)
        """
        self.api_key = api_key or os.environ.get('FLOWALGO_API_KEY')
        self.dark_pool_data = {}
        self.last_update = None
        self.update_frequency = timedelta(hours=1)
        
    def fetch_dark_pool_data(self, symbol):
        """
        Fetch dark pool data for a symbol
        
        In production, this would connect to FlowAlgo API.
        For now, we use synthetic data generation.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with dark pool data
        """
        if (symbol in self.dark_pool_data and self.last_update and 
            datetime.now() - self.last_update < self.update_frequency):
            return self.dark_pool_data[symbol]
        
        
        buys_vs_sells_ratio = np.random.normal(1.0, 0.3)  # >1 means more buys than sells
        avg_trade_size = np.random.exponential(5000) * self._get_symbol_scale(symbol)
        block_trades_count = max(0, int(np.random.normal(5, 3)))
        large_prints = []
        
        for _ in range(block_trades_count):
            side = "BUY" if np.random.random() < (buys_vs_sells_ratio / (1 + buys_vs_sells_ratio)) else "SELL"
            size = int(np.random.exponential(3) * avg_trade_size)
            time_ago = np.random.randint(1, 120)  # minutes ago
            
            large_prints.append({
                "side": side,
                "size": size,
                "time": (datetime.now() - timedelta(minutes=time_ago)).isoformat(),
                "premium": np.random.normal(0, 0.01)  # % premium/discount to visible market
            })
        
        dark_pool_metrics = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "buy_sell_ratio": buys_vs_sells_ratio,
            "avg_trade_size": avg_trade_size,
            "dark_volume_percentage": min(0.85, max(0.05, np.random.normal(0.3, 0.1))),
            "block_trades_count": block_trades_count,
            "large_prints": large_prints,
            "hidden_support_levels": self._generate_support_levels(symbol),
            "hidden_resistance_levels": self._generate_resistance_levels(symbol)
        }
        
        self.dark_pool_data[symbol] = dark_pool_metrics
        self.last_update = datetime.now()
        
        return dark_pool_metrics
    
    def _get_symbol_scale(self, symbol):
        """Get appropriate scale factor for symbol"""
        if 'BTC' in symbol:
            return 0.01  # Bitcoin has smaller trade sizes in BTC terms
        elif 'ETH' in symbol:
            return 0.1   # Ethereum also smaller but larger than BTC
        elif 'XAU' in symbol:
            return 10    # Gold is traded in larger quantities
        elif symbol in ['SPY', 'QQQ', 'DIA']:
            return 100   # ETFs have larger block trades
        else:
            return 1
    
    def _generate_support_levels(self, symbol):
        """Generate synthetic support levels"""
        levels = []
        base_price = self._get_base_price(symbol)
        
        for i in range(3):
            level = base_price * (1 - 0.01 * (i+1) * np.random.normal(1, 0.2))
            strength = np.random.normal(0.7, 0.2)
            levels.append({
                "price": level,
                "strength": min(0.99, max(0.1, strength)),
                "volume_profile": np.random.exponential(100000)
            })
        
        return sorted(levels, key=lambda x: x["price"])
    
    def _generate_resistance_levels(self, symbol):
        """Generate synthetic resistance levels"""
        levels = []
        base_price = self._get_base_price(symbol)
        
        for i in range(3):
            level = base_price * (1 + 0.01 * (i+1) * np.random.normal(1, 0.2))
            strength = np.random.normal(0.7, 0.2)
            levels.append({
                "price": level,
                "strength": min(0.99, max(0.1, strength)),
                "volume_profile": np.random.exponential(100000)
            })
        
        return sorted(levels, key=lambda x: x["price"])
    
    def _get_base_price(self, symbol):
        """Get base price for a symbol"""
        if 'BTC' in symbol:
            return 40000
        elif 'ETH' in symbol:
            return 2500
        elif 'XAU' in symbol:
            return 1800
        elif symbol == 'SPY':
            return 420
        elif symbol == 'QQQ':
            return 360
        elif symbol == 'DIA':
            return 350
        else:
            return 100
    
    def analyze_dark_pool_signal(self, symbol, current_price):
        """
        Analyze dark pool data for trading signals
        
        Parameters:
        - symbol: Trading symbol
        - current_price: Current market price
        
        Returns:
        - Dictionary with signal information
        """
        data = self.fetch_dark_pool_data(symbol)
        
        supports = data["hidden_support_levels"]
        resistances = data["hidden_resistance_levels"]
        
        closest_support = None
        closest_support_distance = float('inf')
        for level in supports:
            if level["price"] < current_price:
                distance = current_price - level["price"]
                if distance < closest_support_distance:
                    closest_support_distance = distance
                    closest_support = level
        
        closest_resistance = None
        closest_resistance_distance = float('inf')
        for level in resistances:
            if level["price"] > current_price:
                distance = level["price"] - current_price
                if distance < closest_resistance_distance:
                    closest_resistance_distance = distance
                    closest_resistance = level
        
        buy_pressure = data["buy_sell_ratio"] * data["dark_volume_percentage"]
        sell_pressure = (1 / data["buy_sell_ratio"]) * data["dark_volume_percentage"]
        
        recent_buy_volume = sum(print["size"] for print in data["large_prints"] 
                               if print["side"] == "BUY" and 
                               (datetime.now() - datetime.fromisoformat(print["time"])).total_seconds() < 3600)
        recent_sell_volume = sum(print["size"] for print in data["large_prints"] 
                                if print["side"] == "SELL" and 
                                (datetime.now() - datetime.fromisoformat(print["time"])).total_seconds() < 3600)
        
        if buy_pressure > 1.5 and recent_buy_volume > recent_sell_volume * 1.5:
            direction = "BUY"
            confidence = min(0.9, buy_pressure / 2)
        elif sell_pressure > 1.5 and recent_sell_volume > recent_buy_volume * 1.5:
            direction = "SELL"
            confidence = min(0.9, sell_pressure / 2)
        else:
            direction = None
            confidence = 0
        
        if direction == "BUY" and closest_support:
            support_factor = 1 - (closest_support_distance / current_price)
            confidence *= (1 + support_factor)
        elif direction == "SELL" and closest_resistance:
            resistance_factor = 1 - (closest_resistance_distance / current_price)
            confidence *= (1 + resistance_factor)
        
        confidence = min(0.95, confidence)
        
        return {
            "direction": direction,
            "confidence": confidence,
            "dark_pool_metrics": {
                "buy_pressure": buy_pressure,
                "sell_pressure": sell_pressure,
                "recent_buy_volume": recent_buy_volume,
                "recent_sell_volume": recent_sell_volume,
                "closest_support": closest_support["price"] if closest_support else None,
                "closest_resistance": closest_resistance["price"] if closest_resistance else None
            }
        }
    
    def get_dark_pool_report(self, symbol):
        """
        Generate detailed dark pool report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed dark pool metrics
        """
        data = self.fetch_dark_pool_data(symbol)
        
        total_volume = sum(print["size"] for print in data["large_prints"])
        buy_volume = sum(print["size"] for print in data["large_prints"] if print["side"] == "BUY")
        sell_volume = sum(print["size"] for print in data["large_prints"] if print["side"] == "SELL")
        
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "dark_pool_summary": {
                "total_volume": total_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_percentage": buy_volume / total_volume if total_volume > 0 else 0,
                "dark_pct_of_total": data["dark_volume_percentage"],
                "average_trade_size": data["avg_trade_size"],
                "support_levels": data["hidden_support_levels"],
                "resistance_levels": data["hidden_resistance_levels"]
            },
            "recent_prints": sorted(data["large_prints"], 
                                    key=lambda x: datetime.fromisoformat(x["time"]), 
                                    reverse=True)[:10]  # Only return most recent 10
        }
        
        return report
