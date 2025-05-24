#!/usr/bin/env python3
"""
News Filter for Quantum Finance Modules

Prevents trading during news events and only allows trading 30 minutes after any news release.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time
import calendar
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('NewsFilter')

class NewsFilter:
    """
    News Filter for Quantum Finance Modules
    
    Prevents trading during news events and only allows trading 30 minutes after any news release.
    """
    
    def __init__(self, news_data_path=None, market_news_path=None, min_wait_minutes=30):
        """
        Initialize the News Filter
        
        Parameters:
        - news_data_path: Path to specific news events data file (default: None)
        - market_news_path: Path to market news categories data file (default: None)
        - min_wait_minutes: Minimum wait time after news in minutes (default: 30)
        """
        self.news_data_path = news_data_path
        self.market_news_path = market_news_path
        self.min_wait_minutes = min_wait_minutes
        self.news_events = []
        self.market_news_categories = {}
        self.last_news_time = None
        self.last_check_time = None
        self.safe_to_trade = True
        
        if news_data_path and os.path.exists(news_data_path):
            self.load_news_data(news_data_path)
        else:
            logger.warning(f"News data file not found at {news_data_path}")
            
        if market_news_path and os.path.exists(market_news_path):
            self.load_market_news_categories(market_news_path)
        else:
            logger.warning(f"Market news categories file not found at {market_news_path}")
            
    def load_news_data(self, news_data_path):
        """
        Load specific news events data from file
        
        Parameters:
        - news_data_path: Path to news data file
        """
        try:
            with open(news_data_path, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and "events" in data:
                    self.news_events = data["events"]
                elif isinstance(data, list):
                    self.news_events = data
                else:
                    self.news_events = []
                    
                logger.info(f"Loaded {len(self.news_events)} news events from {news_data_path}")
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            self.news_events = []
            
    def load_market_news_categories(self, market_news_path):
        """
        Load market news categories from file
        
        Parameters:
        - market_news_path: Path to market news categories file
        """
        try:
            with open(market_news_path, 'r') as f:
                self.market_news_categories = json.load(f)
                
                category_count = sum(len(subcategories) for category in self.market_news_categories.values() 
                                    for subcategories in category.values())
                logger.info(f"Loaded {category_count} market news categories from {market_news_path}")
        except Exception as e:
            logger.error(f"Error loading market news categories: {e}")
            self.market_news_categories = {}
            
    def add_news_event(self, event_time, event_type, event_impact, event_description, assets=None):
        """
        Add a news event to the filter
        
        Parameters:
        - event_time: Time of the news event (datetime or string)
        - event_type: Type of news event (e.g., "Economic", "Earnings", "Fed")
        - event_impact: Impact level of the news (e.g., "High", "Medium", "Low")
        - event_description: Description of the news event
        - assets: List of assets affected by the news event (default: None, affects all assets)
        """
        if isinstance(event_time, str):
            try:
                event_time = datetime.fromisoformat(event_time)
            except ValueError:
                try:
                    event_time = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.error(f"Invalid event time format: {event_time}")
                    return
                    
        event = {
            "time": event_time.isoformat(),
            "type": event_type,
            "impact": event_impact,
            "description": event_description
        }
        
        if assets:
            event["assets"] = assets
            
        self.news_events.append(event)
        logger.info(f"Added news event: {event_description} at {event_time}")
        
        if self.last_news_time is None or event_time > datetime.fromisoformat(self.last_news_time):
            self.last_news_time = event_time.isoformat()
            
    def is_scheduled_news_event(self, current_time, asset=None):
        """
        Check if there's a scheduled news event at the current time
        
        Parameters:
        - current_time: Current time
        - asset: Asset to check (default: None, checks all assets)
        
        Returns:
        - Tuple (bool, str): Whether there's a scheduled news event and the event description
        """
        if not self.market_news_categories:
            return False, None
            
        day_of_week = calendar.day_name[current_time.weekday()]
        day_of_month = current_time.day
        current_hour_minute = current_time.strftime("%H:%M")
        
        for category_name, category in self.market_news_categories.items():
            for subcategory_name, subcategory in category.items():
                for event_code, event in subcategory.items():
                    if asset and "assets" in event and asset not in event["assets"]:
                        continue
                        
                    release_time = event.get("release_time", "")
                    release_day = event.get("release_day", "")
                    
                    if not release_time or not release_day:
                        continue
                        
                    if release_time != "Varies" and release_time != current_hour_minute:
                        continue
                        
                    if release_day == "Varies":
                        pass
                    elif "Friday" in release_day and day_of_week != "Friday":
                        continue
                    elif "Thursday" in release_day and day_of_week != "Thursday":
                        continue
                    elif "Wednesday" in release_day and day_of_week != "Wednesday":
                        continue
                    elif "Monthly" in release_day and day_of_month not in [1, 15, 30, 31]:
                        continue
                    elif "Quarterly" in release_day and current_time.month not in [1, 4, 7, 10]:
                        continue
                    elif release_day not in ["Varies", "Monthly", "Quarterly"] and release_day != day_of_week:
                        continue
                        
                    return True, f"{event.get('name', 'Unknown event')} ({event.get('description', '')})"
                    
        return False, None
            
    def is_safe_to_trade(self, current_time=None, asset=None):
        """
        Check if it's safe to trade based on news events
        
        Parameters:
        - current_time: Current time (default: now)
        - asset: Asset to check (default: None, checks all assets)
        
        Returns:
        - Boolean indicating whether it's safe to trade
        """
        if current_time is None:
            current_time = datetime.now()
        elif isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                try:
                    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.error(f"Invalid current time format: {current_time}")
                    return False
                    
        self.last_check_time = current_time.isoformat()
        
        # Check specific news events
        if self.news_events:
            for event in self.news_events:
                event_time = datetime.fromisoformat(event["time"])
                time_since_event = (current_time - event_time).total_seconds() / 60
                
                if asset and "assets" in event and asset not in event["assets"]:
                    continue
                    
                if 0 <= time_since_event < self.min_wait_minutes:
                    logger.info(f"Not safe to trade: {event['description']} occurred {time_since_event:.1f} minutes ago")
                    self.safe_to_trade = False
                    return False
        
        # Check scheduled news events
        is_scheduled, event_description = self.is_scheduled_news_event(current_time, asset)
        if is_scheduled:
            logger.info(f"Not safe to trade: Scheduled news event {event_description} at {current_time}")
            self.safe_to_trade = False
            return False
            
        self.safe_to_trade = True
        return True
        
    def get_next_safe_time(self, current_time=None, asset=None):
        """
        Get the next time when it's safe to trade
        
        Parameters:
        - current_time: Current time (default: now)
        - asset: Asset to check (default: None, checks all assets)
        
        Returns:
        - Datetime indicating the next safe time to trade
        """
        if current_time is None:
            current_time = datetime.now()
        elif isinstance(current_time, str):
            try:
                current_time = datetime.fromisoformat(current_time)
            except ValueError:
                try:
                    current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.error(f"Invalid current time format: {current_time}")
                    return current_time + timedelta(minutes=self.min_wait_minutes)
        
        is_scheduled, _ = self.is_scheduled_news_event(current_time, asset)
        if is_scheduled:
            return current_time + timedelta(minutes=self.min_wait_minutes)
                    
        # Check specific news events
        if not self.news_events:
            return current_time
            
        most_recent_event = None
        most_recent_time = None
        
        for event in self.news_events:
            event_time = datetime.fromisoformat(event["time"])
            
            if asset and "assets" in event and asset not in event["assets"]:
                continue
                
            if event_time <= current_time and (most_recent_time is None or event_time > most_recent_time):
                most_recent_event = event
                most_recent_time = event_time
                
        if most_recent_event is None:
            return current_time
            
        next_safe_time = most_recent_time + timedelta(minutes=self.min_wait_minutes)
        
        if next_safe_time <= current_time:
            return current_time
            
        return next_safe_time
        
    def save_news_data(self, news_data_path=None):
        """
        Save news data to file
        
        Parameters:
        - news_data_path: Path to news data file (default: self.news_data_path)
        """
        if news_data_path is None:
            news_data_path = self.news_data_path
            
        if news_data_path is None:
            logger.error("No news data path specified")
            return
            
        try:
            with open(news_data_path, 'w') as f:
                json.dump({"events": self.news_events}, f, indent=2)
                logger.info(f"Saved {len(self.news_events)} news events to {news_data_path}")
        except Exception as e:
            logger.error(f"Error saving news data: {e}")
            
    def get_status(self):
        """
        Get the current status of the news filter
        
        Returns:
        - Dictionary with news filter status
        """
        return {
            "safe_to_trade": self.safe_to_trade,
            "last_news_time": self.last_news_time,
            "last_check_time": self.last_check_time,
            "min_wait_minutes": self.min_wait_minutes,
            "news_events_count": len(self.news_events),
            "market_news_categories_count": sum(len(subcategories) for category in self.market_news_categories.values() 
                                              for subcategories in category.values())
        }
        
    def fetch_news_from_api(self, api_key, query="stock market", source="newsapi"):
        """
        Fetch news from an API
        
        Parameters:
        - api_key: API key for the news service
        - query: Query string (default: "stock market")
        - source: News API source (default: "newsapi")
        
        Returns:
        - List of news articles
        """
        try:
            import requests
            
            if source.lower() == "newsapi":
                url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    
                    for article in articles:
                        published_at = article.get("publishedAt")
                        title = article.get("title")
                        
                        if published_at and title:
                            try:
                                event_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                self.add_news_event(
                                    event_time=event_time,
                                    event_type="API",
                                    event_impact="Medium",
                                    event_description=title
                                )
                            except ValueError:
                                logger.error(f"Invalid published_at format: {published_at}")
                                
                    logger.info(f"Added {len(articles)} news events from NewsAPI")
                    return articles
                else:
                    logger.error(f"Error fetching news from NewsAPI: {response.status_code}")
                    return []
            elif source.lower() == "cryptopanic":
                url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    for result in results:
                        published_at = result.get("published_at")
                        title = result.get("title")
                        
                        if published_at and title:
                            try:
                                event_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                self.add_news_event(
                                    event_time=event_time,
                                    event_type="Crypto",
                                    event_impact="Medium",
                                    event_description=title
                                )
                            except ValueError:
                                logger.error(f"Invalid published_at format: {published_at}")
                                
                    logger.info(f"Added {len(results)} news events from CryptoPanic")
                    return results
                else:
                    logger.error(f"Error fetching news from CryptoPanic: {response.status_code}")
                    return []
            else:
                logger.error(f"Unsupported news source: {source}")
                return []
        except ImportError:
            logger.error("requests module not installed")
            return []
        except Exception as e:
            logger.error(f"Error fetching news from API: {e}")
            return []
