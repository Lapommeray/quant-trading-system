"""
PSYOPS (Psychological Operations)

This module implements psychological operations tactics for market analysis.
It analyzes retail sentiment and social media trends (legally).
"""

import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime, timedelta
import time
from collections import Counter

class PsychologicalOperations:
    """
    Psychological Operations
    
    Implements psychological operations tactics for market analysis.
    """
    
    def __init__(self, cache_dir="data/psyops_cache", 
                 reddit_client_id=None, reddit_client_secret=None, 
                 reddit_user_agent="market_warfare_psyops"):
        """
        Initialize Psychological Operations
        
        Parameters:
        - cache_dir: Directory to cache data
        - reddit_client_id: Reddit API client ID (optional)
        - reddit_client_secret: Reddit API client secret (optional)
        - reddit_user_agent: Reddit API user agent
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.reddit = None
        
        if reddit_client_id and reddit_client_secret:
            try:
                import praw
                
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                
                print("Reddit API initialized")
            except Exception as e:
                print(f"Error initializing Reddit API: {e}")
                print("Using simulated data instead")
        
        self.refresh_interval = 3600  # Refresh data every hour
        self.request_limit = 60  # Maximum requests per minute (Reddit API limit)
        self.last_request_time = 0
        
        self.bullish_terms = [
            "moon", "rocket", "bull", "long", "calls", "buy", "hold", "bullish",
            "tendies", "yolo", "diamond hands", "squeeze", "breakout", "rally"
        ]
        
        self.bearish_terms = [
            "crash", "bear", "short", "puts", "sell", "dump", "bearish", "drill",
            "rug pull", "tank", "collapse", "bubble", "overvalued", "bagholders"
        ]
        
        self.bullish_emojis = ["🚀", "🌙", "💎", "👐", "🦍", "📈", "🤑", "💰"]
        self.bearish_emojis = ["🌈", "🐻", "💩", "📉", "🤡", "💸", "😱", "🔥"]
        
        print("Psychological Operations initialized")
    
    def _throttle_requests(self):
        """
        Throttle API requests to stay within rate limits
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def get_wsb_data(self, symbol, force_refresh=False):
        """
        Get wallstreetbets data for a symbol
        
        Parameters:
        - symbol: Symbol to get data for
        - force_refresh: Force refresh data
        
        Returns:
        - Dictionary with wallstreetbets data
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_wsb.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < self.refresh_interval:
                with open(cache_file, "r") as f:
                    return json.load(f)
        
        if self.reddit:
            try:
                self._throttle_requests()
                
                subreddit = self.reddit.subreddit("wallstreetbets")
                posts = list(subreddit.search(symbol, limit=20))
                
                post_data = []
                
                for post in posts:
                    post_data.append({
                        "title": post.title,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "created_utc": post.created_utc,
                        "num_comments": post.num_comments
                    })
                
                with open(cache_file, "w") as f:
                    json.dump(post_data, f)
                
                return post_data
            
            except Exception as e:
                print(f"Error getting Reddit data: {e}")
                print("Using simulated data instead")
        
        return self._generate_simulated_wsb_data(symbol)
    
    def _generate_simulated_wsb_data(self, symbol):
        """
        Generate simulated wallstreetbets data
        
        Parameters:
        - symbol: Symbol to generate data for
        
        Returns:
        - List of dictionaries with simulated post data
        """
        post_data = []
        
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(20)]
        
        bullish_titles = [
            f"{symbol} to the moon! 🚀🚀🚀",
            f"Just YOLO'd my life savings into {symbol} calls",
            f"{symbol} is going to squeeze, get in now!",
            f"Diamond hands on {symbol} 💎👐",
            f"DD: Why {symbol} is undervalued and ready to pop",
            f"The {symbol} dip is a buying opportunity",
            f"{symbol} technical analysis: bullish pattern forming",
            f"Institutions are loading up on {symbol}, we should too",
            f"{symbol} earnings will blow expectations away",
            f"I'm not selling my {symbol} until it hits $1000"
        ]
        
        bearish_titles = [
            f"{symbol} is overvalued, change my mind",
            f"Put your life savings into {symbol} puts",
            f"{symbol} is a bubble about to pop",
            f"Why I'm shorting {symbol} with everything I have",
            f"Technical analysis: {symbol} showing bearish divergence",
            f"{symbol} insiders are dumping, time to get out",
            f"The {symbol} rally is fake, here's why",
            f"DD: {symbol} earnings will disappoint",
            f"The bear case for {symbol} that no one is talking about",
            f"{symbol} bagholders anonymous thread"
        ]
        
        for i in range(20):
            is_bullish = np.random.random() < 0.6  # 60% chance of bullish
            
            if is_bullish:
                title = np.random.choice(bullish_titles)
            else:
                title = np.random.choice(bearish_titles)
            
            score = np.random.randint(10, 5000)
            upvote_ratio = np.random.uniform(0.5, 0.99)
            
            num_comments = np.random.randint(5, 500)
            
            post_data.append({
                "title": title,
                "score": score,
                "upvote_ratio": upvote_ratio,
                "created_utc": timestamps[i].timestamp(),
                "num_comments": num_comments
            })
        
        return post_data
    
    def analyze_sentiment(self, symbol):
        """
        Analyze sentiment for a symbol
        
        Parameters:
        - symbol: Symbol to analyze sentiment for
        
        Returns:
        - Dictionary with sentiment analysis
        """
        wsb_data = self.get_wsb_data(symbol)
        
        bullish_count = 0
        bearish_count = 0
        
        for post in wsb_data:
            title = post["title"].lower()
            
            for term in self.bullish_terms:
                if term.lower() in title:
                    bullish_count += 1
            
            for emoji in self.bullish_emojis:
                bullish_count += title.count(emoji)
            
            for term in self.bearish_terms:
                if term.lower() in title:
                    bearish_count += 1
            
            for emoji in self.bearish_emojis:
                bearish_count += title.count(emoji)
        
        total_count = bullish_count + bearish_count
        
        if total_count == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (bullish_count - bearish_count) / total_count
        
        total_posts = len(wsb_data)
        total_score = sum(post["score"] for post in wsb_data)
        total_comments = sum(post["num_comments"] for post in wsb_data)
        avg_upvote_ratio = np.mean([post["upvote_ratio"] for post in wsb_data]) if wsb_data else 0.0
        
        fomo_level = 0.0
        
        if total_posts > 0:
            normalized_posts = min(total_posts / 20, 1.0)
            normalized_score = min(total_score / 10000, 1.0)
            normalized_comments = min(total_comments / 1000, 1.0)
            
            fomo_level = (normalized_posts + normalized_score + normalized_comments + sentiment_score) / 4
            
            fomo_level *= avg_upvote_ratio
        
        return {
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "total_posts": total_posts,
            "total_score": total_score,
            "total_comments": total_comments,
            "avg_upvote_ratio": avg_upvote_ratio,
            "fomo_level": fomo_level,
            "is_bullish": sentiment_score > 0,
            "timestamp": datetime.now().timestamp()
        }
    
    def detect_retail_fomo(self, symbol):
        """
        Detect retail FOMO for a symbol
        
        Parameters:
        - symbol: Symbol to detect FOMO for
        
        Returns:
        - Dictionary with FOMO detection
        """
        sentiment = self.analyze_sentiment(symbol)
        
        fomo_level = sentiment["fomo_level"]
        
        is_fomo = fomo_level > 0.7
        
        is_panic = sentiment["sentiment_score"] < -0.5 and sentiment["total_posts"] > 10
        
        return {
            "symbol": symbol,
            "fomo_level": fomo_level,
            "is_fomo": is_fomo,
            "is_panic": is_panic,
            "sentiment": sentiment,
            "timestamp": datetime.now().timestamp()
        }
    
    def generate_retail_sentiment_report(self, symbol):
        """
        Generate retail sentiment report for a symbol
        
        Parameters:
        - symbol: Symbol to generate report for
        
        Returns:
        - Dictionary with sentiment report
        """
        fomo = self.detect_retail_fomo(symbol)
        
        market_phase = "neutral"
        
        if fomo["is_fomo"]:
            market_phase = "euphoria"
        elif fomo["is_panic"]:
            market_phase = "capitulation"
        elif fomo["sentiment"]["sentiment_score"] > 0.3:
            market_phase = "optimism"
        elif fomo["sentiment"]["sentiment_score"] < -0.3:
            market_phase = "fear"
        
        contrarian_signal = "NEUTRAL"
        
        if market_phase == "euphoria":
            contrarian_signal = "SELL"
        elif market_phase == "capitulation":
            contrarian_signal = "BUY"
        
        confidence = 0.0
        
        if contrarian_signal == "SELL":
            confidence = fomo["fomo_level"]
        elif contrarian_signal == "BUY":
            confidence = abs(fomo["sentiment"]["sentiment_score"])
        
        return {
            "symbol": symbol,
            "market_phase": market_phase,
            "contrarian_signal": contrarian_signal,
            "confidence": confidence,
            "fomo": fomo,
            "timestamp": datetime.now().timestamp()
        }
    
    def trigger_retail_fomo(self, symbol, fake_news_strength=0.0):
        """
        Generate synthetic social media trends (legally)
        
        Parameters:
        - symbol: Symbol to generate trends for
        - fake_news_strength: Strength of fake news (0.0 = no fake news, legal)
        
        Returns:
        - Dictionary with generated trends
        """
        if fake_news_strength > 0.0:
            print("WARNING: Generating fake news is illegal. Setting fake_news_strength to 0.0.")
            fake_news_strength = 0.0
        
        sentiment = self.analyze_sentiment(symbol)
        
        if sentiment["is_bullish"]:
            message = f"🚀 {symbol} showing bullish sentiment on social media. Current FOMO level: {sentiment['fomo_level']:.2f}"
        else:
            message = f"📉 {symbol} showing bearish sentiment on social media. Current fear level: {abs(sentiment['sentiment_score']):.2f}"
        
        return {
            "symbol": symbol,
            "message": message,
            "sentiment": sentiment,
            "is_legal": True,
            "timestamp": datetime.now().timestamp()
        }

if __name__ == "__main__":
    psyops = PsychologicalOperations()
    
    symbol = "GME"
    
    sentiment = psyops.analyze_sentiment(symbol)
    
    print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
    print(f"Bullish Count: {sentiment['bullish_count']}")
    print(f"Bearish Count: {sentiment['bearish_count']}")
    print(f"Total Posts: {sentiment['total_posts']}")
    print(f"FOMO Level: {sentiment['fomo_level']:.2f}")
    
    fomo = psyops.detect_retail_fomo(symbol)
    
    print(f"\nFOMO Level: {fomo['fomo_level']:.2f}")
    print(f"Is FOMO: {fomo['is_fomo']}")
    print(f"Is Panic: {fomo['is_panic']}")
    
    report = psyops.generate_retail_sentiment_report(symbol)
    
    print(f"\nMarket Phase: {report['market_phase']}")
    print(f"Contrarian Signal: {report['contrarian_signal']}")
    print(f"Confidence: {report['confidence']:.2f}")
    
    trends = psyops.trigger_retail_fomo(symbol)
    
    print(f"\nMessage: {trends['message']}")
    print(f"Is Legal: {trends['is_legal']}")
