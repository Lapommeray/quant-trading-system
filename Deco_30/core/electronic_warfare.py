"""
Electronic Warfare (EW) Tactics

This module implements electronic warfare tactics for market analysis.
It detects liquidity spoofing and other market manipulation techniques.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import time

class ElectronicWarfare:
    """
    Electronic Warfare
    
    Implements electronic warfare tactics for market analysis.
    """
    
    def __init__(self, cache_dir="data/ew_cache"):
        """
        Initialize Electronic Warfare
        
        Parameters:
        - cache_dir: Directory to cache data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.refresh_interval = 60  # Refresh data every minute
        self.spoofing_threshold = 2.5  # Threshold for detecting spoofing
        self.layering_threshold = 3  # Threshold for detecting layering
        self.min_order_size = 100  # Minimum order size to consider
        
        print("Electronic Warfare initialized")
    
    def get_order_book(self, symbol, force_refresh=False):
        """
        Get order book data
        
        Parameters:
        - symbol: Symbol to get data for
        - force_refresh: Force refresh data
        
        Returns:
        - Dictionary with order book data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_order_book.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        
        bids = []
        asks = []
        
        mid_price = 100 + np.random.normal(0, 1)
        
        for i in range(10):
            price = mid_price - (i + 1) * 0.01
            size = np.random.randint(100, 1000)
            
            if i == 3 and np.random.random() < 0.3:
                size = size * 5  # Spoof a large order
            
            bids.append({
                "price": price,
                "size": size
            })
        
        for i in range(10):
            price = mid_price + (i + 1) * 0.01
            size = np.random.randint(100, 1000)
            
            if i == 2 and np.random.random() < 0.3:
                size = size * 5  # Spoof a large order
            
            asks.append({
                "price": price,
                "size": size
            })
        
        order_book = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp(),
            "bids": bids,
            "asks": asks,
            "mid_price": mid_price
        }
        
        with open(cache_file, "w") as f:
            json.dump(order_book, f)
        
        return order_book
    
    def detect_spoofing(self, symbol, order_book=None):
        """
        Detect spoofing in order book
        
        Parameters:
        - symbol: Symbol to detect spoofing for
        - order_book: Order book data (optional)
        
        Returns:
        - Dictionary with spoofing detection
        """
        if order_book is None:
            order_book = self.get_order_book(symbol)
        
        bid_ratios = []
        
        for i in range(1, min(5, len(order_book["bids"]))):
            ratio = order_book["bids"][i]["size"] / order_book["bids"][0]["size"]
            bid_ratios.append(ratio)
        
        ask_ratios = []
        
        for i in range(1, min(5, len(order_book["asks"]))):
            ratio = order_book["asks"][i]["size"] / order_book["asks"][0]["size"]
            ask_ratios.append(ratio)
        
        bid_spoof = any(ratio > self.spoofing_threshold for ratio in bid_ratios)
        ask_spoof = any(ratio > self.spoofing_threshold for ratio in ask_ratios)
        
        bid_confidence = max(bid_ratios) / self.spoofing_threshold if bid_ratios else 0
        ask_confidence = max(ask_ratios) / self.spoofing_threshold if ask_ratios else 0
        
        return {
            "bid_spoof": bid_spoof,
            "ask_spoof": ask_spoof,
            "bid_confidence": min(bid_confidence, 1.0),
            "ask_confidence": min(ask_confidence, 1.0),
            "bid_ratios": bid_ratios,
            "ask_ratios": ask_ratios,
            "timestamp": datetime.now().timestamp()
        }
    
    def detect_layering(self, symbol, order_book=None):
        """
        Detect layering in order book
        
        Parameters:
        - symbol: Symbol to detect layering for
        - order_book: Order book data (optional)
        
        Returns:
        - Dictionary with layering detection
        """
        if order_book is None:
            order_book = self.get_order_book(symbol)
        
        bid_layering = False
        bid_layer_count = 0
        
        for i in range(1, min(5, len(order_book["bids"]))):
            if order_book["bids"][i]["size"] > self.min_order_size and \
               abs(order_book["bids"][i]["price"] - order_book["bids"][i-1]["price"]) < 0.02:
                bid_layer_count += 1
        
        bid_layering = bid_layer_count >= self.layering_threshold
        
        ask_layering = False
        ask_layer_count = 0
        
        for i in range(1, min(5, len(order_book["asks"]))):
            if order_book["asks"][i]["size"] > self.min_order_size and \
               abs(order_book["asks"][i]["price"] - order_book["asks"][i-1]["price"]) < 0.02:
                ask_layer_count += 1
        
        ask_layering = ask_layer_count >= self.layering_threshold
        
        return {
            "bid_layering": bid_layering,
            "ask_layering": ask_layering,
            "bid_layer_count": bid_layer_count,
            "ask_layer_count": ask_layer_count,
            "timestamp": datetime.now().timestamp()
        }
    
    def detect_manipulation(self, symbol):
        """
        Detect market manipulation
        
        Parameters:
        - symbol: Symbol to detect manipulation for
        
        Returns:
        - Dictionary with manipulation detection
        """
        order_book = self.get_order_book(symbol)
        
        spoofing = self.detect_spoofing(symbol, order_book)
        
        layering = self.detect_layering(symbol, order_book)
        
        manipulation = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp(),
            "spoofing": spoofing,
            "layering": layering,
            "is_manipulated": spoofing["bid_spoof"] or spoofing["ask_spoof"] or layering["bid_layering"] or layering["ask_layering"],
            "manipulation_confidence": max(
                spoofing["bid_confidence"],
                spoofing["ask_confidence"],
                layering["bid_layer_count"] / (self.layering_threshold * 2),
                layering["ask_layer_count"] / (self.layering_threshold * 2)
            )
        }
        
        return manipulation
    
    def get_counter_strategy(self, symbol):
        """
        Get counter strategy for detected manipulation
        
        Parameters:
        - symbol: Symbol to get counter strategy for
        
        Returns:
        - Dictionary with counter strategy
        """
        manipulation = self.detect_manipulation(symbol)
        
        counter_strategy = "NEUTRAL"
        confidence = 0.0
        
        if manipulation["is_manipulated"]:
            if manipulation["spoofing"]["bid_spoof"]:
                counter_strategy = "SELL"
                confidence = manipulation["spoofing"]["bid_confidence"]
            elif manipulation["spoofing"]["ask_spoof"]:
                counter_strategy = "BUY"
                confidence = manipulation["spoofing"]["ask_confidence"]
            elif manipulation["layering"]["bid_layering"]:
                counter_strategy = "BUY"
                confidence = manipulation["layering"]["bid_layer_count"] / (self.layering_threshold * 2)
            elif manipulation["layering"]["ask_layering"]:
                counter_strategy = "SELL"
                confidence = manipulation["layering"]["ask_layer_count"] / (self.layering_threshold * 2)
        
        return {
            "symbol": symbol,
            "counter_strategy": counter_strategy,
            "confidence": confidence,
            "manipulation": manipulation,
            "timestamp": datetime.now().timestamp()
        }

if __name__ == "__main__":
    ew = ElectronicWarfare()
    
    symbol = "SPY"
    
    spoofing = ew.detect_spoofing(symbol)
    
    print(f"Bid Spoofing: {spoofing['bid_spoof']}")
    print(f"Ask Spoofing: {spoofing['ask_spoof']}")
    print(f"Bid Confidence: {spoofing['bid_confidence']:.2f}")
    print(f"Ask Confidence: {spoofing['ask_confidence']:.2f}")
    
    layering = ew.detect_layering(symbol)
    
    print(f"\nBid Layering: {layering['bid_layering']}")
    print(f"Ask Layering: {layering['ask_layering']}")
    print(f"Bid Layer Count: {layering['bid_layer_count']}")
    print(f"Ask Layer Count: {layering['ask_layer_count']}")
    
    manipulation = ew.detect_manipulation(symbol)
    
    print(f"\nIs Manipulated: {manipulation['is_manipulated']}")
    print(f"Manipulation Confidence: {manipulation['manipulation_confidence']:.2f}")
    
    counter_strategy = ew.get_counter_strategy(symbol)
    
    print(f"\nCounter Strategy: {counter_strategy['counter_strategy']}")
    print(f"Confidence: {counter_strategy['confidence']:.2f}")
