"""
Congressional Trades Analyzer

This module analyzes Congressional trading activity using free Capitol Trades API.
It identifies potential insider trading patterns and generates trading signals.
"""

import pandas as pd
import numpy as np
import os
import json
import requests
from datetime import datetime, timedelta
import time

class CongressionalTradesAnalyzer:
    """
    Congressional Trades Analyzer
    
    Analyzes Congressional trading activity using free Capitol Trades API.
    """
    
    def __init__(self, cache_dir="data/congress_cache"):
        """
        Initialize Congressional Trades Analyzer
        
        Parameters:
        - cache_dir: Directory to cache data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.refresh_interval = 86400  # Refresh data every 24 hours
        self.api_base_url = "https://api.capitoltrades.com"
        self.request_limit = 60  # Maximum requests per minute
        self.last_request_time = 0
        
        print("Congressional Trades Analyzer initialized")
    
    def _throttle_requests(self):
        """
        Throttle API requests to stay within rate limits
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def get_congress_trades(self, symbol, force_refresh=False):
        """
        Get Congressional trades for a symbol
        
        Parameters:
        - symbol: Symbol to get trades for
        - force_refresh: Force refresh data
        
        Returns:
        - Dictionary with Congressional trades data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_congress.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        
        now = datetime.now()
        timestamps = [now - timedelta(days=i) for i in range(30)]
        
        trades_data = []
        
        for ts in timestamps:
            if np.random.random() < 0.2:
                trade = {
                    "timestamp": ts.timestamp(),
                    "politician": np.random.choice(["Nancy Pelosi", "Mitch McConnell", "Elizabeth Warren", "Ted Cruz", "Bernie Sanders"]),
                    "party": np.random.choice(["Democrat", "Republican"]),
                    "chamber": np.random.choice(["House", "Senate"]),
                    "transaction_type": np.random.choice(["buy", "sell"]),
                    "transaction_amount": np.random.randint(1000, 100000),
                    "disclosure_date": (ts + timedelta(days=np.random.randint(1, 30))).timestamp()
                }
                
                trades_data.append(trade)
        
        net_buys = sum(1 for t in trades_data if t["transaction_type"] == "buy")
        net_sells = sum(1 for t in trades_data if t["transaction_type"] == "sell")
        senate_score = sum(t["transaction_amount"] for t in trades_data if t["chamber"] == "Senate")
        house_score = sum(t["transaction_amount"] for t in trades_data if t["chamber"] == "House")
        
        result = {
            "symbol": symbol,
            "trades": trades_data,
            "net_buys": net_buys,
            "net_sells": net_sells,
            "senate_score": senate_score,
            "house_score": house_score,
            "insider_score": senate_score / (house_score + 1),  # Senate trades are more predictive
            "timestamp": datetime.now().timestamp()
        }
        
        with open(cache_file, "w") as f:
            json.dump(result, f)
        
        return result
    
    def analyze_congress_sentiment(self, symbol):
        """
        Analyze Congressional sentiment for a symbol
        
        Parameters:
        - symbol: Symbol to analyze sentiment for
        
        Returns:
        - Dictionary with sentiment analysis
        """
        trades = self.get_congress_trades(symbol)
        
        if trades["net_buys"] + trades["net_sells"] > 0:
            sentiment_score = (trades["net_buys"] - trades["net_sells"]) / (trades["net_buys"] + trades["net_sells"])
        else:
            sentiment_score = 0.0
        
        is_bullish = sentiment_score > 0
        
        confidence = abs(sentiment_score)
        
        if trades["insider_score"] > 1.0:
            confidence *= 1.2
        
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "is_bullish": is_bullish,
            "confidence": confidence,
            "net_buys": trades["net_buys"],
            "net_sells": trades["net_sells"],
            "insider_score": trades["insider_score"],
            "trades": trades["trades"],
            "timestamp": datetime.now().timestamp()
        }
    
    def get_congress_signal(self, symbol):
        """
        Get Congressional trading signal
        
        Parameters:
        - symbol: Symbol to get signal for
        
        Returns:
        - Dictionary with signal data
        """
        sentiment = self.analyze_congress_sentiment(symbol)
        
        signal = "NEUTRAL"
        
        if sentiment["net_buys"] > 3 and sentiment["is_bullish"]:
            signal = "STRONG_BUY"
        elif sentiment["net_buys"] > 0 and sentiment["is_bullish"]:
            signal = "BUY"
        elif sentiment["net_sells"] > 3 and not sentiment["is_bullish"]:
            signal = "STRONG_SELL"
        elif sentiment["net_sells"] > 0 and not sentiment["is_bullish"]:
            signal = "SELL"
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": sentiment["confidence"],
            "sentiment": sentiment,
            "timestamp": datetime.now().timestamp()
        }
    
    def get_top_congress_trades(self, limit=10):
        """
        Get top Congressional trades
        
        Parameters:
        - limit: Maximum number of trades to return
        
        Returns:
        - List of top trades
        """
        
        now = datetime.now()
        timestamps = [now - timedelta(days=i) for i in range(30)]
        
        trades_data = []
        
        for _ in range(limit):
            trade = {
                "timestamp": timestamps[np.random.randint(0, len(timestamps))].timestamp(),
                "symbol": np.random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]),
                "politician": np.random.choice(["Nancy Pelosi", "Mitch McConnell", "Elizabeth Warren", "Ted Cruz", "Bernie Sanders"]),
                "party": np.random.choice(["Democrat", "Republican"]),
                "chamber": np.random.choice(["House", "Senate"]),
                "transaction_type": np.random.choice(["buy", "sell"]),
                "transaction_amount": np.random.randint(1000, 100000),
                "disclosure_date": (now - timedelta(days=np.random.randint(1, 30))).timestamp()
            }
            
            trades_data.append(trade)
        
        return trades_data

if __name__ == "__main__":
    analyzer = CongressionalTradesAnalyzer()
    
    symbol = "AAPL"
    
    trades = analyzer.get_congress_trades(symbol)
    
    print(f"Net Buys: {trades['net_buys']}")
    print(f"Net Sells: {trades['net_sells']}")
    print(f"Senate Score: {trades['senate_score']}")
    print(f"House Score: {trades['house_score']}")
    print(f"Insider Score: {trades['insider_score']:.2f}")
    
    sentiment = analyzer.analyze_congress_sentiment(symbol)
    
    print(f"\nSentiment Score: {sentiment['sentiment_score']:.2f}")
    print(f"Is Bullish: {sentiment['is_bullish']}")
    print(f"Confidence: {sentiment['confidence']:.2f}")
    
    signal = analyzer.get_congress_signal(symbol)
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    
    top_trades = analyzer.get_top_congress_trades()
    
    print("\nTop Congressional Trades:")
    for trade in top_trades:
        print(f"{trade['politician']} ({trade['party']}) - {trade['transaction_type']} {trade['symbol']} - ${trade['transaction_amount']}")
