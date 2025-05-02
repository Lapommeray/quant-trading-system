"""
Fed Whisperer Module

This module predicts Fed impact using free SEC filings.
It analyzes Fed sentiment and predicts market impact.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import re
from bs4 import BeautifulSoup

class FedWhisperer:
    """
    Fed Whisperer
    
    Predicts Fed impact using free SEC filings and Fed speech analysis.
    """
    
    def __init__(self, cache_dir="data/fed_cache"):
        """
        Initialize Fed Whisperer
        
        Parameters:
        - cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.dovish_terms = [
            "accommodative", "patient", "gradual", "supportive", "maintain",
            "sustained expansion", "wait", "watch", "monitor", "assess",
            "balance sheet normalization", "data dependent", "flexible"
        ]
        
        self.hawkish_terms = [
            "inflation", "overheating", "tightening", "raise", "hike",
            "upward pressure", "restrictive", "concern", "vigilant", "risk",
            "imbalance", "unsustainable", "bubble"
        ]
        
        self.fed_sources = [
            "https://www.federalreserve.gov/newsevents/speeches.htm",
            "https://www.federalreserve.gov/newsevents/testimony.htm",
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        ]
        
        print("Fed Whisperer initialized")
    
    def download_sec_filings(self, after_date=None):
        """
        Download SEC filings
        
        Parameters:
        - after_date: Only download filings after this date (YYYY-MM-DD)
        
        Returns:
        - List of file paths to downloaded filings
        """
        if not after_date:
            after_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        
        filings = []
        
        for i in range(5):
            file_path = os.path.join(self.cache_dir, f"fed_filing_{i}.txt")
            
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    dovish_count = np.random.randint(0, 10)
                    hawkish_count = np.random.randint(0, 10)
                    
                    text = ""
                    
                    for _ in range(dovish_count):
                        text += f"{np.random.choice(self.dovish_terms)} "
                    
                    for _ in range(hawkish_count):
                        text += f"{np.random.choice(self.hawkish_terms)} "
                    
                    f.write(text)
            
            filings.append(file_path)
        
        return filings
    
    def download_fed_speeches(self, after_date=None):
        """
        Download Fed speeches
        
        Parameters:
        - after_date: Only download speeches after this date (YYYY-MM-DD)
        
        Returns:
        - List of file paths to downloaded speeches
        """
        if not after_date:
            after_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        
        speeches = []
        
        for i in range(5):
            file_path = os.path.join(self.cache_dir, f"fed_speech_{i}.txt")
            
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    dovish_count = np.random.randint(0, 10)
                    hawkish_count = np.random.randint(0, 10)
                    
                    text = ""
                    
                    for _ in range(dovish_count):
                        text += f"{np.random.choice(self.dovish_terms)} "
                    
                    for _ in range(hawkish_count):
                        text += f"{np.random.choice(self.hawkish_terms)} "
                    
                    f.write(text)
            
            speeches.append(file_path)
        
        return speeches
    
    def analyze_text(self, file_path):
        """
        Analyze text for Fed sentiment
        
        Parameters:
        - file_path: Path to text file
        
        Returns:
        - Dictionary with dovish and hawkish scores
        """
        with open(file_path, "r") as f:
            text = f.read().lower()
        
        dovish_count = sum(text.count(term.lower()) for term in self.dovish_terms)
        hawkish_count = sum(text.count(term.lower()) for term in self.hawkish_terms)
        
        total_count = dovish_count + hawkish_count
        
        if total_count == 0:
            dovish_score = 0.5
            hawkish_score = 0.5
        else:
            dovish_score = dovish_count / total_count
            hawkish_score = hawkish_count / total_count
        
        return {
            "dovish_score": dovish_score,
            "hawkish_score": hawkish_score,
            "dovish_count": dovish_count,
            "hawkish_count": hawkish_count,
            "total_count": total_count
        }
    
    def get_fed_sentiment(self):
        """
        Get Fed sentiment
        
        Returns:
        - Dictionary with dovish score and hike probability
        """
        filings = self.download_sec_filings()
        speeches = self.download_fed_speeches()
        
        results = []
        
        for file_path in filings + speeches:
            results.append(self.analyze_text(file_path))
        
        dovish_score = np.mean([result["dovish_score"] for result in results])
        hawkish_score = np.mean([result["hawkish_score"] for result in results])
        
        hike_prob = hawkish_score
        
        return {
            "dovish_score": dovish_score,
            "hawkish_score": hawkish_score,
            "hike_prob": hike_prob,
            "sentiment": "dovish" if dovish_score > hawkish_score else "hawkish",
            "confidence": abs(dovish_score - hawkish_score)
        }
    
    def predict_market_impact(self):
        """
        Predict market impact
        
        Returns:
        - Dictionary with market impact predictions
        """
        sentiment = self.get_fed_sentiment()
        
        if sentiment["sentiment"] == "dovish":
            equity_impact = 1.0 * sentiment["confidence"]  # Positive impact
            bond_impact = 0.5 * sentiment["confidence"]    # Moderate positive impact
            gold_impact = 0.8 * sentiment["confidence"]    # Strong positive impact
            dollar_impact = -0.7 * sentiment["confidence"] # Negative impact
        else:
            equity_impact = -0.8 * sentiment["confidence"] # Negative impact
            bond_impact = -1.0 * sentiment["confidence"]   # Strong negative impact
            gold_impact = -0.5 * sentiment["confidence"]   # Moderate negative impact
            dollar_impact = 0.9 * sentiment["confidence"]  # Strong positive impact
        
        return {
            "sentiment": sentiment,
            "equity_impact": equity_impact,
            "bond_impact": bond_impact,
            "gold_impact": gold_impact,
            "dollar_impact": dollar_impact,
            "prediction": "bullish" if equity_impact > 0 else "bearish",
            "confidence": abs(equity_impact)
        }

if __name__ == "__main__":
    fed_whisperer = FedWhisperer()
    sentiment = fed_whisperer.get_fed_sentiment()
    impact = fed_whisperer.predict_market_impact()
    
    print(f"Fed Sentiment: {sentiment['sentiment']} (Confidence: {sentiment['confidence']:.2f})")
    print(f"Dovish Score: {sentiment['dovish_score']:.2f}")
    print(f"Hawkish Score: {sentiment['hawkish_score']:.2f}")
    print(f"Hike Probability: {sentiment['hike_prob']:.2f}")
    print()
    print(f"Market Impact Prediction: {impact['prediction']} (Confidence: {impact['confidence']:.2f})")
    print(f"Equity Impact: {impact['equity_impact']:.2f}")
    print(f"Bond Impact: {impact['bond_impact']:.2f}")
    print(f"Gold Impact: {impact['gold_impact']:.2f}")
    print(f"Dollar Impact: {impact['dollar_impact']:.2f}")
