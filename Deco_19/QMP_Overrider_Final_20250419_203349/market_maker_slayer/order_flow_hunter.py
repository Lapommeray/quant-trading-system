"""
Order Flow Hunter Module

Detects and exploits order flow imbalances by analyzing order book data
and predicting HFT reactions to liquidity gaps.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class OrderBookImbalanceScanner:
    """
    Order Book Imbalance Scanner
    
    Scans order books for imbalances that can be exploited.
    """
    
    def __init__(self):
        """Initialize Order Book Imbalance Scanner"""
        self.order_books = {}
        self.imbalance_history = {}
        
        print("Initializing Order Book Imbalance Scanner")
    
    def calculate_imbalance(self, symbol):
        """
        Calculate order book imbalance for a symbol
        
        Parameters:
        - symbol: Symbol to calculate imbalance for
        
        Returns:
        - Imbalance ratio (-1.0 to 1.0)
        """
        if symbol not in self.order_books:
            self._generate_order_book(symbol)
        
        order_book = self.order_books[symbol]
        
        bid_volume = sum([level["size"] for level in order_book["bids"]])
        ask_volume = sum([level["size"] for level in order_book["asks"]])
        
        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            imbalance = 0.0
        
        if symbol not in self.imbalance_history:
            self.imbalance_history[symbol] = []
        
        self.imbalance_history[symbol].append({
            "timestamp": datetime.now().timestamp(),
            "imbalance": imbalance
        })
        
        if len(self.imbalance_history[symbol]) > 100:
            self.imbalance_history[symbol] = self.imbalance_history[symbol][-100:]
        
        return imbalance
    
    def _generate_order_book(self, symbol):
        """
        Generate an order book for a symbol
        
        Parameters:
        - symbol: Symbol to generate order book for
        """
        
        if "BTC" in symbol:
            mid_price = random.uniform(50000, 70000)
        elif "ETH" in symbol:
            mid_price = random.uniform(2000, 4000)
        elif "XAU" in symbol:
            mid_price = random.uniform(1800, 2200)
        elif symbol in ["SPY", "QQQ", "DIA"]:
            mid_price = random.uniform(300, 500)
        else:
            mid_price = random.uniform(100, 1000)
        
        bids = []
        for i in range(10):
            price = mid_price * (1 - (i + 1) * 0.001)
            size = random.randint(1, 100) * 10
            bids.append({
                "price": price,
                "size": size
            })
        
        asks = []
        for i in range(10):
            price = mid_price * (1 + (i + 1) * 0.001)
            size = random.randint(1, 100) * 10
            asks.append({
                "price": price,
                "size": size
            })
        
        order_book = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp(),
            "bids": bids,
            "asks": asks
        }
        
        self.order_books[symbol] = order_book

class HTFBehaviorDatabase:
    """
    HFT Behavior Database
    
    Database of HFT behavior patterns for predicting reactions to order book imbalances.
    """
    
    def __init__(self):
        """Initialize HFT Behavior Database"""
        self.behavior_patterns = {
            "momentum_chase": {
                "description": "Chase momentum when imbalance exceeds threshold",
                "threshold": 0.5,
                "reaction_time_ms": 50,
                "direction": "same"
            },
            "mean_reversion": {
                "description": "Fade extreme imbalances",
                "threshold": 0.8,
                "reaction_time_ms": 20,
                "direction": "opposite"
            },
            "liquidity_provision": {
                "description": "Provide liquidity when imbalance is moderate",
                "threshold": 0.3,
                "reaction_time_ms": 10,
                "direction": "opposite"
            },
            "stop_hunt": {
                "description": "Hunt stops when imbalance indicates vulnerability",
                "threshold": 0.6,
                "reaction_time_ms": 30,
                "direction": "same"
            },
            "iceberg_detection": {
                "description": "Detect and react to hidden iceberg orders",
                "threshold": 0.4,
                "reaction_time_ms": 15,
                "direction": "opposite"
            }
        }
        
        print("Initializing HFT Behavior Database")
    
    def predict_reaction(self, imbalance):
        """
        Predict HFT reaction to order book imbalance
        
        Parameters:
        - imbalance: Order book imbalance ratio (-1.0 to 1.0)
        
        Returns:
        - HFT reaction prediction
        """
        abs_imbalance = abs(imbalance)
        
        triggered_patterns = []
        
        for pattern_name, pattern in self.behavior_patterns.items():
            if abs_imbalance >= pattern["threshold"]:
                triggered_patterns.append({
                    "pattern": pattern_name,
                    "description": pattern["description"],
                    "reaction_time_ms": pattern["reaction_time_ms"],
                    "direction": "BUY" if (imbalance > 0 and pattern["direction"] == "same") or (imbalance < 0 and pattern["direction"] == "opposite") else "SELL"
                })
        
        if not triggered_patterns:
            return None
        
        triggered_patterns.sort(key=lambda x: x["reaction_time_ms"])
        
        return triggered_patterns[0]

class OrderFlowHunter:
    """
    Order Flow Hunter
    
    Finds hidden liquidity gaps HFTs will exploit.
    """
    
    def __init__(self):
        """Initialize Order Flow Hunter"""
        self.book_analyzer = OrderBookImbalanceScanner()
        self.hft_patterns = HTFBehaviorDatabase()
        
        print("Initializing Order Flow Hunter")
    
    def detect_imbalance(self, symbol):
        """
        Finds hidden liquidity gaps HFTs will exploit
        
        Parameters:
        - symbol: Symbol to detect imbalance for
        
        Returns:
        - Imbalance detection data
        """
        imbalance = self.book_analyzer.calculate_imbalance(symbol)
        hft_reaction = self.hft_patterns.predict_reaction(imbalance)
        
        if hft_reaction:
            return {
                "symbol": symbol,
                "imbalance_ratio": imbalance,
                "expected_hft_action": hft_reaction["direction"],
                "hft_pattern": hft_reaction["pattern"],
                "pattern_description": hft_reaction["description"],
                "snipe_window_ms": hft_reaction["reaction_time_ms"],
                "confidence": abs(imbalance) * 0.8 + 0.2,  # Scale to 0.2-1.0 range
                "timestamp": datetime.now().timestamp()
            }
        
        return None
