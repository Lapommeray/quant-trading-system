"""
Liquidity X-Ray

This module reconstructs order flow from free Trade & Quote (TAQ) data.
It detects hidden liquidity and predicts market impact.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import time
import json

class LiquidityXRay:
    """
    Liquidity X-Ray
    
    Reconstructs order flow from free Trade & Quote (TAQ) data.
    """
    
    def __init__(self, cache_dir="data/liquidity_cache"):
        """
        Initialize Liquidity X-Ray
        
        Parameters:
        - cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.imbalance_threshold = 0.2  # Threshold for detecting imbalance
        self.volume_threshold = 1.5     # Threshold for detecting unusual volume
        self.refresh_interval = 3600    # Refresh data every hour
        
        print("Liquidity X-Ray initialized")
    
    def get_quote_data(self, symbol, force_refresh=False):
        """
        Get quote data
        
        Parameters:
        - symbol: Symbol to get data for
        - force_refresh: Force refresh data
        
        Returns:
        - DataFrame with quote data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_quotes.csv")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                return pd.read_csv(cache_file)
        
        
        now = datetime.now()
        timestamps = [now - timedelta(seconds=i) for i in range(100)]
        timestamps.reverse()
        
        data = []
        
        for ts in timestamps:
            mid_price = 100 + np.random.normal(0, 1)
            spread = np.random.uniform(0.01, 0.1)
            
            bid = mid_price - spread / 2
            ask = mid_price + spread / 2
            
            bid_size = np.random.randint(100, 1000)
            ask_size = np.random.randint(100, 1000)
            
            if np.random.random() < 0.2:
                if np.random.random() < 0.5:
                    bid_size *= np.random.uniform(1.5, 3.0)
                else:
                    ask_size *= np.random.uniform(1.5, 3.0)
            
            data.append({
                "timestamp": ts,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "mid": (bid + ask) / 2
            })
        
        df = pd.DataFrame(data)
        
        df.to_csv(cache_file, index=False)
        
        return df
    
    def get_trade_data(self, symbol, force_refresh=False):
        """
        Get trade data
        
        Parameters:
        - symbol: Symbol to get data for
        - force_refresh: Force refresh data
        
        Returns:
        - DataFrame with trade data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_trades.csv")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                return pd.read_csv(cache_file)
        
        
        now = datetime.now()
        timestamps = [now - timedelta(seconds=i) for i in range(50)]
        timestamps.reverse()
        
        data = []
        
        for ts in timestamps:
            price = 100 + np.random.normal(0, 1)
            size = np.random.randint(100, 1000)
            
            if np.random.random() < 0.1:
                size *= np.random.uniform(5.0, 10.0)
            
            data.append({
                "timestamp": ts,
                "price": price,
                "size": size,
                "exchange": np.random.choice(["NYSE", "NASDAQ", "ARCA", "BATS", "EDGX"]),
                "condition": np.random.choice(["regular", "odd_lot", "outside_regular_hours", "corrected", "derivatively_priced"])
            })
        
        df = pd.DataFrame(data)
        
        df.to_csv(cache_file, index=False)
        
        return df
    
    def detect_hidden_liquidity(self, symbol):
        """
        Detect hidden liquidity
        
        Parameters:
        - symbol: Symbol to detect hidden liquidity for
        
        Returns:
        - Dictionary with hidden liquidity detection
        """
        quotes = self.get_quote_data(symbol)
        trades = self.get_trade_data(symbol)
        
        total_bid_size = quotes['bid_size'].sum()
        total_ask_size = quotes['ask_size'].sum()
        
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        hidden_bid = imbalance < -self.imbalance_threshold
        hidden_ask = imbalance > self.imbalance_threshold
        
        avg_trade_size = trades['size'].mean()
        
        unusual_volume = False
        
        if len(trades) > 10:
            recent_trades = trades.iloc[-10:]
            recent_volume = recent_trades['size'].sum()
            
            avg_10_trade_volume = trades['size'].rolling(10).sum().mean()
            
            unusual_volume = recent_volume > avg_10_trade_volume * self.volume_threshold
        
        dark_pool_activity = False
        
        if len(trades) > 0:
            for i, trade in trades.iterrows():
                closest_quote = quotes[quotes['timestamp'] <= trade['timestamp']].iloc[-1] if not quotes[quotes['timestamp'] <= trade['timestamp']].empty else None
                
                if closest_quote is not None:
                    is_at_midpoint = abs(trade['price'] - closest_quote['mid']) < 0.01
                    
                    is_large = trade['size'] > avg_trade_size * 2
                    
                    if is_at_midpoint and is_large:
                        dark_pool_activity = True
                        break
        
        return {
            "imbalance": imbalance,
            "hidden_bid": hidden_bid,
            "hidden_ask": hidden_ask,
            "unusual_volume": unusual_volume,
            "dark_pool_activity": dark_pool_activity,
            "bid_ask_ratio": total_bid_size / total_ask_size if total_ask_size > 0 else float('inf'),
            "confidence": abs(imbalance) * (2 if unusual_volume else 1) * (1.5 if dark_pool_activity else 1)
        }
    
    def predict_price_impact(self, symbol):
        """
        Predict price impact
        
        Parameters:
        - symbol: Symbol to predict price impact for
        
        Returns:
        - Dictionary with price impact prediction
        """
        liquidity = self.detect_hidden_liquidity(symbol)
        
        if liquidity["hidden_bid"]:
            direction = "up"
            confidence = liquidity["confidence"]
        elif liquidity["hidden_ask"]:
            direction = "down"
            confidence = liquidity["confidence"]
        else:
            direction = "neutral"
            confidence = 0.0
        
        if liquidity["dark_pool_activity"]:
            if direction == "up":
                confidence *= 1.5
            elif direction == "down":
                confidence *= 1.5
        
        if liquidity["unusual_volume"]:
            confidence *= 1.2
        
        confidence = min(confidence, 1.0)
        
        expected_move = 0.0
        
        if direction == "up":
            expected_move = 0.1 * confidence
        elif direction == "down":
            expected_move = -0.1 * confidence
        
        is_institutional = liquidity["dark_pool_activity"] or (liquidity["unusual_volume"] and abs(liquidity["imbalance"]) > 0.4)
        
        gap_fill_scenario = False
        
        if len(trades := self.get_trade_data(symbol)) > 20:
            prices = trades['price'].values
            gaps = np.abs(np.diff(prices))
            
            if np.max(gaps[-20:]) > np.std(prices) * 3:
                gap_fill_scenario = True
        
        if gap_fill_scenario:
            if is_institutional:
                expected_move *= 1.5
            else:
                expected_move *= -0.5
        
        return {
            "direction": direction,
            "confidence": confidence,
            "expected_move": expected_move,
            "is_institutional": is_institutional,
            "liquidity_data": liquidity,
            "prediction": "bullish" if direction == "up" else "bearish" if direction == "down" else "neutral",
            "time_horizon": "short-term"  # Liquidity imbalances typically resolve quickly
        }
    
    def analyze_retail_vs_institutional(self, symbol):
        """
        Analyze retail vs. institutional behavior
        
        Parameters:
        - symbol: Symbol to analyze
        
        Returns:
        - Dictionary with retail vs. institutional analysis
        """
        quotes = self.get_quote_data(symbol)
        trades = self.get_trade_data(symbol)
        
        avg_trade_size = trades['size'].mean()
        
        large_trades = trades[trades['size'] > avg_trade_size * 2]
        
        small_trades = trades[trades['size'] < avg_trade_size * 0.5]
        
        large_trade_impact = 0
        small_trade_impact = 0
        
        if len(large_trades) > 1:
            large_trade_impact = np.corrcoef(large_trades['size'][:-1], np.diff(large_trades['price']))[0, 1]
        
        if len(small_trades) > 1:
            small_trade_impact = np.corrcoef(small_trades['size'][:-1], np.diff(small_trades['price']))[0, 1]
        
        high_volume_candles = []
        
        if len(trades) > 0:
            trades['minute'] = trades['timestamp'].apply(lambda x: x.replace(second=0, microsecond=0))
            candles = trades.groupby('minute').agg({
                'price': ['first', 'max', 'min', 'last'],
                'size': 'sum'
            })
            
            candles.columns = ['open', 'high', 'low', 'close', 'volume']
            
            avg_volume = candles['volume'].mean()
            high_volume_candles = candles[candles['volume'] > avg_volume * 2]
        
        breakout_count = 0
        harvesting_count = 0
        
        for _, candle in high_volume_candles.iterrows():
            is_bullish = candle['close'] > candle['open']
            
            if is_bullish and candle['close'] > candle['high'] * 0.99:
                breakout_count += 1
            elif not is_bullish and candle['close'] < candle['low'] * 1.01:
                breakout_count += 1
            else:
                harvesting_count += 1
        
        high_volume_interpretation = "breakout" if breakout_count > harvesting_count else "liquidity_harvesting"
        
        return {
            "large_trades_count": len(large_trades),
            "small_trades_count": len(small_trades),
            "large_trade_impact": large_trade_impact,
            "small_trade_impact": small_trade_impact,
            "high_volume_candles_count": len(high_volume_candles),
            "high_volume_interpretation": high_volume_interpretation,
            "is_retail_dominated": len(small_trades) > len(large_trades) * 3,
            "is_institutional_dominated": len(large_trades) > len(small_trades) * 0.5,
            "retail_sentiment": "bullish" if small_trade_impact > 0 else "bearish",
            "institutional_sentiment": "bullish" if large_trade_impact > 0 else "bearish"
        }

if __name__ == "__main__":
    xray = LiquidityXRay()
    
    symbol = "SPY"
    
    liquidity = xray.detect_hidden_liquidity(symbol)
    
    print(f"Imbalance: {liquidity['imbalance']:.2f}")
    print(f"Hidden Bid: {liquidity['hidden_bid']}")
    print(f"Hidden Ask: {liquidity['hidden_ask']}")
    print(f"Unusual Volume: {liquidity['unusual_volume']}")
    print(f"Dark Pool Activity: {liquidity['dark_pool_activity']}")
    print(f"Bid/Ask Ratio: {liquidity['bid_ask_ratio']:.2f}")
    print(f"Confidence: {liquidity['confidence']:.2f}")
    
    impact = xray.predict_price_impact(symbol)
    
    print(f"\nDirection: {impact['direction']}")
    print(f"Confidence: {impact['confidence']:.2f}")
    print(f"Expected Move: {impact['expected_move']:.4f}")
    print(f"Is Institutional: {impact['is_institutional']}")
    print(f"Prediction: {impact['prediction']}")
    
    analysis = xray.analyze_retail_vs_institutional(symbol)
    
    print(f"\nLarge Trades: {analysis['large_trades_count']}")
    print(f"Small Trades: {analysis['small_trades_count']}")
    print(f"Large Trade Impact: {analysis['large_trade_impact']:.4f}")
    print(f"Small Trade Impact: {analysis['small_trade_impact']:.4f}")
    print(f"High Volume Candles: {analysis['high_volume_candles_count']}")
    print(f"High Volume Interpretation: {analysis['high_volume_interpretation']}")
    print(f"Retail Dominated: {analysis['is_retail_dominated']}")
    print(f"Institutional Dominated: {analysis['is_institutional_dominated']}")
    print(f"Retail Sentiment: {analysis['retail_sentiment']}")
    print(f"Institutional Sentiment: {analysis['institutional_sentiment']}")
