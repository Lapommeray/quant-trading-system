"""
Retail Sentiment Analyzer
Analyzes retail trader sentiment for contrarian signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import math
import random

class RetailSentimentAnalyzer:
    def __init__(self):
        """Initialize Retail Sentiment Analyzer"""
        self.sentiment_data = {}
        self.last_update = {}
        self.update_frequency = timedelta(hours=6)
        self.sentiment_sources = ["social_media", "trading_forums", "retail_flows", "search_trends"]
        
    def fetch_sentiment_data(self, symbol):
        """
        Fetch sentiment data for a symbol
        
        In production, this would connect to social media APIs, forums, etc.
        For now, we use synthetic data generation.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with sentiment data
        """
        if (symbol in self.sentiment_data and symbol in self.last_update and 
            datetime.now() - self.last_update[symbol] < self.update_frequency):
            return self.sentiment_data[symbol]
        
        now = datetime.now()
        
        base_sentiment = np.random.normal(0, 0.3)
        
        is_trending = random.random() < 0.3
        trend_factor = 2 if is_trending else 1
        
        sentiment_by_source = {}
        for source in self.sentiment_sources:
            source_sentiment = base_sentiment + np.random.normal(0, 0.2)
            
            source_sentiment = max(-1, min(1, source_sentiment * trend_factor))
            
            sentiment_by_source[source] = source_sentiment
        
        bullish_percentage = min(0.95, max(0.05, 0.5 + base_sentiment * 0.4 * trend_factor))
        bearish_percentage = 1 - bullish_percentage
        
        base_volume = np.random.exponential(1000) * (1 + trend_factor)
        post_volume = {
            "total": int(base_volume),
            "bullish": int(base_volume * bullish_percentage),
            "bearish": int(base_volume * bearish_percentage),
            "neutral": int(base_volume * 0.1)  # Some posts are neutral
        }
        
        sentiment_magnitude = abs(base_sentiment)
        urgency_metrics = {
            "exclamation_frequency": min(0.8, 0.1 + sentiment_magnitude * 0.5),
            "caps_percentage": min(0.5, 0.05 + sentiment_magnitude * 0.3),
            "rocket_emoji_count": int(max(0, sentiment_magnitude * 20 - 5)) if base_sentiment > 0 else 0,
            "bear_emoji_count": int(max(0, sentiment_magnitude * 15 - 5)) if base_sentiment < 0 else 0,
            "urgent_terms_frequency": min(0.7, 0.1 + sentiment_magnitude * 0.4)
        }
        
        bullish_phrases = self._generate_phrases(symbol, is_bullish=True, sentiment_magnitude=sentiment_magnitude)
        bearish_phrases = self._generate_phrases(symbol, is_bullish=False, sentiment_magnitude=sentiment_magnitude)
        
        self.sentiment_data[symbol] = {
            "timestamp": now.isoformat(),
            "overall_sentiment": base_sentiment,
            "sentiment_by_source": sentiment_by_source,
            "bullish_percentage": bullish_percentage,
            "bearish_percentage": bearish_percentage,
            "post_volume": post_volume,
            "urgency_metrics": urgency_metrics,
            "common_phrases": {
                "bullish": bullish_phrases,
                "bearish": bearish_phrases
            },
            "is_trending": is_trending
        }
        
        self.last_update[symbol] = now
        
        return self.sentiment_data[symbol]
    
    def _generate_phrases(self, symbol, is_bullish, sentiment_magnitude):
        """Generate common phrases based on sentiment"""
        phrases = []
        
        base_phrases = {
            True: [  # Bullish phrases
                f"{symbol} to the moon!",
                f"HODL {symbol}",
                f"Just bought more {symbol}",
                f"{symbol} is undervalued",
                f"Bullish on {symbol}",
                f"{symbol} ready to explode",
                f"Diamond hands on {symbol}",
                f"Don't miss {symbol} rally",
                f"{symbol} short squeeze incoming",
                f"Loading up on {symbol}"
            ],
            False: [  # Bearish phrases
                f"{symbol} is overvalued",
                f"Selling my {symbol}",
                f"Bearish on {symbol}",
                f"{symbol} going to crash",
                f"Getting out of {symbol}",
                f"{symbol} is a scam",
                f"Put options on {symbol}",
                f"{symbol} bubble about to burst",
                f"Shorting {symbol}",
                f"{symbol} downtrend confirmed"
            ]
        }
        
        num_phrases = max(1, min(len(base_phrases[is_bullish]), int(sentiment_magnitude * 5)))
        selected_phrases = random.sample(base_phrases[is_bullish], num_phrases)
        
        for phrase in selected_phrases:
            count = int(np.random.exponential(50) * sentiment_magnitude)
            phrases.append({
                "text": phrase,
                "count": count
            })
        
        return sorted(phrases, key=lambda x: x["count"], reverse=True)
    
    def analyze_sentiment(self, symbol):
        """
        Analyze sentiment data for trading signals
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with signal information
        """
        data = self.fetch_sentiment_data(symbol)
        
        sentiment = data["overall_sentiment"]
        
        if sentiment > 0.7:
            direction = "SELL"
            confidence = min(0.9, (sentiment - 0.7) * 3)
            message = "Extreme bullish retail sentiment detected - contrarian sell signal"
        elif sentiment < -0.7:
            direction = "BUY"
            confidence = min(0.9, (abs(sentiment) - 0.7) * 3)
            message = "Extreme bearish retail sentiment detected - contrarian buy signal"
        elif sentiment > 0.4:
            direction = "SELL"
            confidence = min(0.6, (sentiment - 0.4) * 3)
            message = "Moderately bullish retail sentiment - weak contrarian sell signal"
        elif sentiment < -0.4:
            direction = "BUY"
            confidence = min(0.6, (abs(sentiment) - 0.4) * 3)
            message = "Moderately bearish retail sentiment - weak contrarian buy signal"
        else:
            direction = None
            confidence = 0
            message = "Neutral retail sentiment - no contrarian signal"
        
        urgency = data["urgency_metrics"]
        urgency_score = (urgency["exclamation_frequency"] + 
                         urgency["caps_percentage"] + 
                         urgency["urgent_terms_frequency"]) / 3
        
        if urgency_score > 0.3 and direction:
            confidence = min(0.95, confidence * (1 + urgency_score))
            message += f" (high urgency: {urgency_score:.2f})"
        
        volume = data["post_volume"]["total"]
        volume_factor = min(2, max(0.5, math.log10(volume) / 3))
        
        if direction:
            confidence = min(0.95, confidence * volume_factor)
            
            if volume_factor > 1.2:
                message += f" (high volume: {volume} posts)"
            elif volume_factor < 0.8:
                message += f" (low volume: {volume} posts)"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "message": message,
            "sentiment": sentiment,
            "urgency_score": urgency_score,
            "volume": volume,
            "is_trending": data["is_trending"]
        }
    
    def get_sentiment_report(self, symbol):
        """
        Generate detailed sentiment report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed sentiment metrics
        """
        data = self.fetch_sentiment_data(symbol)
        
        report = {
            "symbol": symbol,
            "timestamp": data["timestamp"],
            "overall_sentiment": data["overall_sentiment"],
            "sentiment_by_source": data["sentiment_by_source"],
            "bullish_percentage": data["bullish_percentage"],
            "bearish_percentage": data["bearish_percentage"],
            "post_volume": data["post_volume"],
            "urgency_metrics": data["urgency_metrics"],
            "top_bullish_phrases": data["common_phrases"]["bullish"][:3],
            "top_bearish_phrases": data["common_phrases"]["bearish"][:3],
            "is_trending": data["is_trending"],
            "analysis": self.analyze_sentiment(symbol)
        }
        
        return report
