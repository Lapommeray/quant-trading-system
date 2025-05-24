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
    
    def __init__(self, news_data_path=None, min_wait_minutes=30):
        """
        Initialize the News Filter
        
        Parameters:
        - news_data_path: Path to news data file (default: None)
        - min_wait_minutes: Minimum wait time after news in minutes (default: 30)
        """
        self.news_data_path = news_data_path
        self.min_wait_minutes = min_wait_minutes
        self.news_events = []
        self.last_news_time = None
        self.last_check_time = None
        self.safe_to_trade = True
        
        if news_data_path and os.path.exists(news_data_path):
            self.load_news_data(news_data_path)
        else:
            logger.warning(f"News data file not found at {news_data_path}")
            
    def load_news_data(self, news_data_path):
        """
        Load news data from file
        
        Parameters:
        - news_data_path: Path to news data file
        """
        try:
            with open(news_data_path, 'r') as f:
                self.news_events = json.load(f)
                logger.info(f"Loaded {len(self.news_events)} news events from {news_data_path}")
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            self.news_events = []
            
    def add_news_event(self, event_time, event_type, event_impact, event_description):
        """
        Add a news event to the filter
        
        Parameters:
        - event_time: Time of the news event (datetime or string)
        - event_type: Type of news event (e.g., "Economic", "Earnings", "Fed")
        - event_impact: Impact level of the news (e.g., "High", "Medium", "Low")
        - event_description: Description of the news event
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
        
        self.news_events.append(event)
        logger.info(f"Added news event: {event_description} at {event_time}")
        
        if self.last_news_time is None or event_time > datetime.fromisoformat(self.last_news_time):
            self.last_news_time = event_time.isoformat()
            
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
        
        if not self.news_events:
            self.safe_to_trade = True
            return True
            
        for event in self.news_events:
            event_time = datetime.fromisoformat(event["time"])
            time_since_event = (current_time - event_time).total_seconds() / 60
            
            if asset and "assets" in event and asset not in event["assets"]:
                continue
                
            if 0 <= time_since_event < self.min_wait_minutes:
                logger.info(f"Not safe to trade: {event['description']} occurred {time_since_event:.1f} minutes ago")
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
                json.dump(self.news_events, f, indent=2)
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
            "news_events_count": len(self.news_events)
        }
