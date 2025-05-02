"""
Retail Trader DNA Extraction

This module decodes Robinhood/Reddit patterns to identify retail trader sentiment.
It uses the Reddit API to analyze wallstreetbets posts and extract sentiment.
"""

import praw
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
import re
from collections import Counter

class RetailDNAExtractor:
    """
    Retail Trader DNA Extraction
    
    Decodes Robinhood/Reddit patterns to identify retail trader sentiment.
    """
    
    def __init__(self, cache_dir="data/retail_dna_cache", 
                 client_id=None, client_secret=None, user_agent="market_mind_reader"):
        """
        Initialize Retail DNA Extractor
        
        Parameters:
        - cache_dir: Directory to cache downloaded data
        - client_id: Reddit API client ID (optional)
        - client_secret: Reddit API client secret (optional)
        - user_agent: Reddit API user agent
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.reddit = None
        
        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
            except Exception as e:
                print(f"Error initializing Reddit API: {e}")
                print("Using simulated data instead")
        
        self.refresh_interval = 3600  # Refresh data every hour
        self.bullish_emojis = ["🚀", "🌙", "💎", "👐", "🦍", "📈"]
        self.bearish_emojis = ["🌈", "🐻", "💩", "📉", "🤡", "💸"]
        self.bullish_terms = ["moon", "tendies", "calls", "bull", "long", "yolo", "hold", "buy", "bullish"]
        self.bearish_terms = ["puts", "short", "bear", "sell", "crash", "dump", "bearish", "drill"]
        
        print("Retail DNA Extractor initialized")
    
    def get_wsb_sentiment(self, ticker, force_refresh=False):
        """
        Get wallstreetbets sentiment for a ticker
        
        Parameters:
        - ticker: Ticker symbol
        - force_refresh: Force refresh data
        
        Returns:
        - Dictionary with sentiment data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_wsb_sentiment.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        if self.reddit:
            try:
                subreddit = self.reddit.subreddit("wallstreetbets")
                posts = list(subreddit.search(ticker, limit=20))
                
                post_data = []
                
                for post in posts:
                    post_data.append({
                        "title": post.title,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "created_utc": post.created_utc,
                        "num_comments": post.num_comments
                    })
                
                sentiment = self._analyze_sentiment(post_data, ticker)
                
                with open(cache_file, "w") as f:
                    json.dump(sentiment, f)
                
                return sentiment
            
            except Exception as e:
                print(f"Error getting Reddit data: {e}")
                print("Using simulated data instead")
        
        return self._generate_simulated_sentiment(ticker)
    
    def _analyze_sentiment(self, post_data, ticker):
        """
        Analyze sentiment from post data
        
        Parameters:
        - post_data: List of post data dictionaries
        - ticker: Ticker symbol
        
        Returns:
        - Dictionary with sentiment data
        """
        if not post_data:
            return self._generate_simulated_sentiment(ticker)
        
        mentions = len(post_data)
        
        avg_score = sum(post["score"] for post in post_data) / mentions if mentions > 0 else 0
        avg_upvote_ratio = sum(post["upvote_ratio"] for post in post_data) / mentions if mentions > 0 else 0
        
        bullish_count = 0
        bearish_count = 0
        
        for post in post_data:
            title = post["title"].lower()
            
            for emoji in self.bullish_emojis:
                bullish_count += title.count(emoji)
            
            for term in self.bullish_terms:
                if re.search(r'\b' + term + r'\b', title):
                    bullish_count += 1
            
            for emoji in self.bearish_emojis:
                bearish_count += title.count(emoji)
            
            for term in self.bearish_terms:
                if re.search(r'\b' + term + r'\b', title):
                    bearish_count += 1
        
        total_indicators = bullish_count + bearish_count
        bull_ratio = bullish_count / total_indicators if total_indicators > 0 else 0.5
        
        sentiment_score = (bull_ratio - 0.5) * 2
        
        fomo_level = min(1.0, mentions / 20 * avg_upvote_ratio)
        
        is_meme_stock = mentions > 10 and bull_ratio > 0.7 and fomo_level > 0.7
        
        return {
            "ticker": ticker,
            "mentions": mentions,
            "bull_ratio": bull_ratio,
            "sentiment_score": sentiment_score,
            "fomo_level": fomo_level,
            "avg_score": avg_score,
            "avg_upvote_ratio": avg_upvote_ratio,
            "is_meme_stock": is_meme_stock,
            "timestamp": datetime.now().timestamp()
        }
    
    def _generate_simulated_sentiment(self, ticker):
        """
        Generate simulated sentiment data
        
        Parameters:
        - ticker: Ticker symbol
        
        Returns:
        - Dictionary with simulated sentiment data
        """
        mentions = np.random.randint(1, 30)
        bull_ratio = np.random.uniform(0.2, 0.8)
        sentiment_score = (bull_ratio - 0.5) * 2
        fomo_level = np.random.uniform(0.1, 0.9)
        avg_score = np.random.randint(10, 1000)
        avg_upvote_ratio = np.random.uniform(0.5, 0.95)
        is_meme_stock = mentions > 10 and bull_ratio > 0.7 and fomo_level > 0.7
        
        if ticker.upper() in ["GME", "AMC", "BBBY", "NOK", "BB"]:
            mentions = np.random.randint(15, 50)
            bull_ratio = np.random.uniform(0.6, 0.9)
            sentiment_score = (bull_ratio - 0.5) * 2
            fomo_level = np.random.uniform(0.7, 1.0)
            is_meme_stock = True
        
        return {
            "ticker": ticker,
            "mentions": mentions,
            "bull_ratio": bull_ratio,
            "sentiment_score": sentiment_score,
            "fomo_level": fomo_level,
            "avg_score": avg_score,
            "avg_upvote_ratio": avg_upvote_ratio,
            "is_meme_stock": is_meme_stock,
            "timestamp": datetime.now().timestamp(),
            "simulated": True
        }
    
    def get_retail_fomo(self, ticker):
        """
        Get retail FOMO level for a ticker
        
        Parameters:
        - ticker: Ticker symbol
        
        Returns:
        - Dictionary with FOMO data
        """
        sentiment = self.get_wsb_sentiment(ticker)
        
        return {
            "ticker": ticker,
            "mentions": sentiment["mentions"],
            "bull_ratio": sentiment["bull_ratio"],
            "fomo_level": sentiment["fomo_level"],
            "is_meme_stock": sentiment["is_meme_stock"],
            "sentiment": "bullish" if sentiment["sentiment_score"] > 0 else "bearish",
            "confidence": abs(sentiment["sentiment_score"])
        }
    
    def analyze_retail_vs_institutional(self, ticker):
        """
        Analyze retail vs. institutional behavior
        
        Parameters:
        - ticker: Ticker symbol
        
        Returns:
        - Dictionary with retail vs. institutional analysis
        """
        retail = self.get_retail_fomo(ticker)
        
        is_contrarian = retail["bull_ratio"] < 0.3 or retail["bull_ratio"] > 0.7
        
        is_good_contrarian = (retail["bull_ratio"] < 0.3 and retail["mentions"] > 5) or \
                             (retail["bull_ratio"] > 0.7 and retail["mentions"] > 15)
        
        contrarian_score = 0.0
        
        if retail["bull_ratio"] < 0.3:
            contrarian_score = 1.0 - retail["bull_ratio"]
        elif retail["bull_ratio"] > 0.7:
            contrarian_score = -(retail["bull_ratio"] - 0.5) * 2
        
        return {
            "ticker": ticker,
            "retail_sentiment": retail["sentiment"],
            "retail_confidence": retail["confidence"],
            "is_contrarian_signal": is_contrarian,
            "is_good_contrarian": is_good_contrarian,
            "contrarian_score": contrarian_score,
            "contrarian_direction": "bullish" if contrarian_score > 0 else "bearish",
            "fomo_level": retail["fomo_level"],
            "is_meme_stock": retail["is_meme_stock"]
        }

if __name__ == "__main__":
    extractor = RetailDNAExtractor()
    
    ticker = "SPY"
    
    sentiment = extractor.get_wsb_sentiment(ticker)
    
    print(f"Ticker: {sentiment['ticker']}")
    print(f"Mentions: {sentiment['mentions']}")
    print(f"Bull Ratio: {sentiment['bull_ratio']:.2f}")
    print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
    print(f"FOMO Level: {sentiment['fomo_level']:.2f}")
    print(f"Is Meme Stock: {sentiment['is_meme_stock']}")
    
    fomo = extractor.get_retail_fomo(ticker)
    
    print(f"\nFOMO Level: {fomo['fomo_level']:.2f}")
    print(f"Sentiment: {fomo['sentiment']}")
    print(f"Confidence: {fomo['confidence']:.2f}")
    
    analysis = extractor.analyze_retail_vs_institutional(ticker)
    
    print(f"\nRetail Sentiment: {analysis['retail_sentiment']}")
    print(f"Is Contrarian Signal: {analysis['is_contrarian_signal']}")
    print(f"Is Good Contrarian: {analysis['is_good_contrarian']}")
    print(f"Contrarian Score: {analysis['contrarian_score']:.2f}")
    print(f"Contrarian Direction: {analysis['contrarian_direction']}")
