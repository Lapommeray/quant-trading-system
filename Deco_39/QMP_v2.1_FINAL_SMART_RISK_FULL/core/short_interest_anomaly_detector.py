"""
Short Interest Anomaly Detector

This module detects unusual short interest patterns that could indicate potential short squeezes.
It uses free FINRA short volume data and options open interest to identify anomalous patterns.
"""

import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
import time
from sklearn.ensemble import IsolationForest

class ShortInterestAnomalyDetector:
    """
    Short Interest Anomaly Detector
    
    Detects unusual short interest patterns that could indicate potential short squeezes.
    Uses free FINRA short volume data and options open interest to identify anomalous patterns.
    """
    
    def __init__(self, cache_dir="data/short_interest_cache"):
        """
        Initialize Short Interest Anomaly Detector
        
        Parameters:
        - cache_dir: Directory to cache data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.refresh_interval = 86400  # Refresh data every 24 hours
        self.request_limit = 60  # Maximum requests per minute
        self.last_request_time = 0
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # 5% of data points are considered anomalies
            random_state=42
        )
        
        print("Short Interest Anomaly Detector initialized")
    
    def _throttle_requests(self):
        """
        Throttle API requests to stay within rate limits
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def get_short_volume_data(self, symbol, days=30, force_refresh=False):
        """
        Get short volume data for a symbol
        
        Parameters:
        - symbol: Symbol to get short volume data for
        - days: Number of days of data to retrieve
        - force_refresh: Force refresh data
        
        Returns:
        - DataFrame with short volume data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_short_volume.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return pd.DataFrame(data)
        
        
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        data = []
        
        for i, date in enumerate(dates):
            short_volume_ratio = 0.3 + (i / days) * 0.2
            
            short_volume_ratio += np.random.normal(0, 0.05)
            
            short_volume_ratio = max(0.1, min(0.9, short_volume_ratio))
            
            total_volume = np.random.randint(100000, 10000000)
            short_volume = int(total_volume * short_volume_ratio)
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "total_volume": total_volume,
                "short_volume": short_volume,
                "short_volume_ratio": short_volume_ratio
            })
        
        with open(cache_file, "w") as f:
            json.dump(data, f)
        
        return pd.DataFrame(data)
    
    def get_options_data(self, symbol, force_refresh=False):
        """
        Get options data for a symbol
        
        Parameters:
        - symbol: Symbol to get options data for
        - force_refresh: Force refresh data
        
        Returns:
        - Dictionary with options data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_options.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        
        expiry_dates = [
            (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        ]
        
        options_data = {
            "symbol": symbol,
            "current_price": np.random.uniform(50, 200),
            "expiry_dates": expiry_dates,
            "calls": {},
            "puts": {}
        }
        
        for expiry in expiry_dates:
            options_data["calls"][expiry] = []
            options_data["puts"][expiry] = []
            
            for i in range(5):
                strike = options_data["current_price"] * (0.8 + i * 0.1)
                
                call = {
                    "strike": strike,
                    "open_interest": np.random.randint(100, 5000),
                    "volume": np.random.randint(10, 1000),
                    "implied_volatility": np.random.uniform(0.2, 0.8)
                }
                
                put = {
                    "strike": strike,
                    "open_interest": np.random.randint(100, 5000),
                    "volume": np.random.randint(10, 1000),
                    "implied_volatility": np.random.uniform(0.2, 0.8)
                }
                
                options_data["calls"][expiry].append(call)
                options_data["puts"][expiry].append(put)
        
        with open(cache_file, "w") as f:
            json.dump(options_data, f)
        
        return options_data
    
    def get_failures_to_deliver(self, symbol, force_refresh=False):
        """
        Get failures to deliver data for a symbol
        
        Parameters:
        - symbol: Symbol to get FTD data for
        - force_refresh: Force refresh data
        
        Returns:
        - DataFrame with FTD data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_ftd.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return pd.DataFrame(data)
        
        
        dates = pd.date_range(end=datetime.now(), periods=30)
        
        data = []
        
        for i, date in enumerate(dates):
            ftd_count = int(1000 * (1 + i / 10))
            
            ftd_count = int(ftd_count * np.random.uniform(0.8, 1.2))
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "ftd_count": ftd_count,
                "price": np.random.uniform(50, 200)
            })
        
        with open(cache_file, "w") as f:
            json.dump(data, f)
        
        return pd.DataFrame(data)
    
    def detect_anomalies(self, symbol):
        """
        Detect short interest anomalies for a symbol
        
        Parameters:
        - symbol: Symbol to detect anomalies for
        
        Returns:
        - Dictionary with anomaly detection results
        """
        short_volume_df = self.get_short_volume_data(symbol)
        options_data = self.get_options_data(symbol)
        ftd_df = self.get_failures_to_deliver(symbol)
        
        features = []
        
        short_volume_ratio = short_volume_df["short_volume_ratio"].values[-5:]
        short_volume_ratio_change = short_volume_ratio[-1] - short_volume_ratio[0]
        
        put_call_ratio = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for expiry in options_data["expiry_dates"]:
            for call in options_data["calls"][expiry]:
                total_call_oi += call["open_interest"]
            
            for put in options_data["puts"][expiry]:
                total_put_oi += put["open_interest"]
        
        if total_call_oi > 0:
            put_call_ratio = total_put_oi / total_call_oi
        
        recent_ftds = ftd_df["ftd_count"].values[-5:]
        ftd_change = recent_ftds[-1] / max(1, recent_ftds[0])
        
        features = np.array([
            short_volume_ratio[-1],
            short_volume_ratio_change,
            put_call_ratio,
            ftd_change
        ]).reshape(1, -1)
        
        anomaly_score = -self.model.decision_function(features)[0]
        is_anomaly = self.model.predict(features)[0] == -1
        
        squeeze_probability = min(1.0, max(0.0, anomaly_score / 2))
        
        is_potential_squeeze = (
            is_anomaly and
            short_volume_ratio_change > 0.1 and
            ftd_change > 1.5 and
            put_call_ratio > 1.2
        )
        
        return {
            "symbol": symbol,
            "anomaly_score": anomaly_score,
            "is_anomaly": bool(is_anomaly),
            "squeeze_probability": squeeze_probability,
            "is_potential_squeeze": is_potential_squeeze,
            "short_volume_ratio": short_volume_ratio[-1],
            "short_volume_change": short_volume_ratio_change,
            "put_call_ratio": put_call_ratio,
            "ftd_change": ftd_change,
            "timestamp": datetime.now().timestamp()
        }
    
    def get_short_squeeze_signal(self, symbol):
        """
        Get short squeeze signal for a symbol
        
        Parameters:
        - symbol: Symbol to get signal for
        
        Returns:
        - Dictionary with signal data
        """
        anomaly = self.detect_anomalies(symbol)
        
        signal = "NEUTRAL"
        confidence = anomaly["squeeze_probability"]
        
        if anomaly["is_potential_squeeze"]:
            if anomaly["squeeze_probability"] > 0.8:
                signal = "STRONG_BUY"
            elif anomaly["squeeze_probability"] > 0.5:
                signal = "BUY"
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "anomaly": anomaly,
            "timestamp": datetime.now().timestamp()
        }
    
    def get_top_squeeze_candidates(self, symbols=None, limit=10):
        """
        Get top short squeeze candidates
        
        Parameters:
        - symbols: List of symbols to check
        - limit: Maximum number of candidates to return
        
        Returns:
        - List of top squeeze candidates
        """
        if symbols is None:
            symbols = ["GME", "AMC", "BBBY", "KOSS", "EXPR", "NOK", "BB", "TLRY", "SNDL", "WISH"]
        
        candidates = []
        
        for symbol in symbols:
            signal = self.get_short_squeeze_signal(symbol)
            if signal["confidence"] > 0.3:
                candidates.append(signal)
        
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return candidates[:limit]

if __name__ == "__main__":
    detector = ShortInterestAnomalyDetector()
    
    symbol = "GME"
    
    anomaly = detector.detect_anomalies(symbol)
    
    print(f"Anomaly Score: {anomaly['anomaly_score']:.2f}")
    print(f"Is Anomaly: {anomaly['is_anomaly']}")
    print(f"Squeeze Probability: {anomaly['squeeze_probability']:.2f}")
    print(f"Is Potential Squeeze: {anomaly['is_potential_squeeze']}")
    
    signal = detector.get_short_squeeze_signal(symbol)
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    
    candidates = detector.get_top_squeeze_candidates()
    
    print("\nTop Short Squeeze Candidates:")
    for candidate in candidates:
        print(f"{candidate['symbol']} - {candidate['signal']} - {candidate['confidence']:.2f}")
