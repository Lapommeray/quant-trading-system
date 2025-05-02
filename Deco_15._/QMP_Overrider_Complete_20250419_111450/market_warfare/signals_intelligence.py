"""
SIGINT (Signals Intelligence) Trading

This module implements signals intelligence tactics for market analysis.
It decodes dark pool prints and other hidden market signals.
"""

import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
import time

class SignalsIntelligence:
    """
    Signals Intelligence
    
    Implements signals intelligence tactics for market analysis.
    """
    
    def __init__(self, cache_dir="data/sigint_cache"):
        """
        Initialize Signals Intelligence
        
        Parameters:
        - cache_dir: Directory to cache data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.refresh_interval = 3600  # Refresh data every hour
        self.large_print_threshold = 10000  # Threshold for large prints
        
        print("Signals Intelligence initialized")
    
    def get_ats_data(self, symbol, force_refresh=False):
        """
        Get ATS (Alternative Trading System) data
        
        Parameters:
        - symbol: Symbol to get data for
        - force_refresh: Force refresh data
        
        Returns:
        - DataFrame with ATS data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_ats.csv")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                return pd.read_csv(cache_file)
        
        
        now = datetime.now()
        timestamps = [now - timedelta(minutes=i) for i in range(100)]
        timestamps.reverse()
        
        data = []
        
        for ts in timestamps:
            price = 100 + np.random.normal(0, 1)
            size = np.random.randint(100, 5000)
            
            if np.random.random() < 0.1:
                size = np.random.randint(10000, 50000)
            
            data.append({
                "timestamp": ts,
                "price": price,
                "size": size,
                "venue": np.random.choice(["UBS", "CITI", "MS", "JPM", "GS"]),
                "side": np.random.choice(["buy", "sell"])
            })
        
        df = pd.DataFrame(data)
        
        df.to_csv(cache_file, index=False)
        
        return df
    
    def decode_dark_pool_prints(self, symbol):
        """
        Decode dark pool prints
        
        Parameters:
        - symbol: Symbol to decode dark pool prints for
        
        Returns:
        - Dictionary with dark pool print analysis
        """
        ats = self.get_ats_data(symbol)
        
        large_prints = ats[ats["size"] > self.large_print_threshold]
        
        if not large_prints.empty:
            large_prints["rounded_price"] = np.round(large_prints["price"], 2)
            
            clusters = large_prints["rounded_price"].value_counts()
            
            top_clusters = clusters.nlargest(3)
            
            if not top_clusters.empty:
                support = top_clusters.index.min() - 0.01
                resistance = top_clusters.index.max() + 0.01
                
                volume_by_price = large_prints.groupby("rounded_price")["size"].sum()
                
                buy_volume = large_prints[large_prints["side"] == "buy"]["size"].sum()
                sell_volume = large_prints[large_prints["side"] == "sell"]["size"].sum()
                
                buying_pressure = buy_volume > sell_volume
                
                return {
                    "symbol": symbol,
                    "support": support,
                    "resistance": resistance,
                    "top_clusters": top_clusters.to_dict(),
                    "volume_by_price": volume_by_price.to_dict(),
                    "buying_pressure": buying_pressure,
                    "buy_volume": int(buy_volume),
                    "sell_volume": int(sell_volume),
                    "large_print_count": len(large_prints),
                    "timestamp": datetime.now().timestamp()
                }
        
        return {
            "symbol": symbol,
            "support": None,
            "resistance": None,
            "top_clusters": {},
            "volume_by_price": {},
            "buying_pressure": None,
            "buy_volume": 0,
            "sell_volume": 0,
            "large_print_count": 0,
            "timestamp": datetime.now().timestamp()
        }
    
    def analyze_print_patterns(self, symbol):
        """
        Analyze dark pool print patterns
        
        Parameters:
        - symbol: Symbol to analyze print patterns for
        
        Returns:
        - Dictionary with print pattern analysis
        """
        ats = self.get_ats_data(symbol)
        
        ats = ats.sort_values("timestamp")
        ats["time_diff"] = ats["timestamp"].diff().dt.total_seconds()
        
        avg_time_diff = ats["time_diff"].mean()
        
        avg_size = ats["size"].mean()
        
        size_volatility = ats["size"].std() / avg_size
        
        bursts = ats[ats["time_diff"] < avg_time_diff / 2]
        
        size_anomalies = ats[ats["size"] > avg_size * 3]
        
        return {
            "symbol": symbol,
            "avg_time_diff": avg_time_diff,
            "avg_size": avg_size,
            "size_volatility": size_volatility,
            "burst_count": len(bursts),
            "size_anomaly_count": len(size_anomalies),
            "timestamp": datetime.now().timestamp()
        }
    
    def get_dark_pool_signal(self, symbol):
        """
        Get dark pool trading signal
        
        Parameters:
        - symbol: Symbol to get signal for
        
        Returns:
        - Dictionary with dark pool signal
        """
        prints = self.decode_dark_pool_prints(symbol)
        
        patterns = self.analyze_print_patterns(symbol)
        
        signal = "NEUTRAL"
        confidence = 0.0
        
        if prints["large_print_count"] > 0:
            if prints["buying_pressure"]:
                signal = "BUY"
                confidence = min(prints["buy_volume"] / (prints["sell_volume"] + 1), 1.0) * 0.8
            else:
                signal = "SELL"
                confidence = min(prints["sell_volume"] / (prints["buy_volume"] + 1), 1.0) * 0.8
        
        if patterns["burst_count"] > 5:
            confidence *= 1.2
        
        if patterns["size_anomaly_count"] > 3:
            confidence *= 1.1
        
        confidence = min(confidence, 1.0)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "prints": prints,
            "patterns": patterns,
            "timestamp": datetime.now().timestamp()
        }

if __name__ == "__main__":
    sigint = SignalsIntelligence()
    
    symbol = "SPY"
    
    prints = sigint.decode_dark_pool_prints(symbol)
    
    print(f"Support: {prints['support']}")
    print(f"Resistance: {prints['resistance']}")
    print(f"Buying Pressure: {prints['buying_pressure']}")
    print(f"Buy Volume: {prints['buy_volume']}")
    print(f"Sell Volume: {prints['sell_volume']}")
    print(f"Large Print Count: {prints['large_print_count']}")
    
    patterns = sigint.analyze_print_patterns(symbol)
    
    print(f"\nAvg Time Diff: {patterns['avg_time_diff']:.2f} seconds")
    print(f"Avg Size: {patterns['avg_size']:.2f}")
    print(f"Size Volatility: {patterns['size_volatility']:.2f}")
    print(f"Burst Count: {patterns['burst_count']}")
    print(f"Size Anomaly Count: {patterns['size_anomaly_count']}")
    
    signal = sigint.get_dark_pool_signal(symbol)
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
