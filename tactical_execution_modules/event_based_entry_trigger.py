"""
Event-Based Entry Trigger

Uses news AI to anticipate CPI/FOMC surprises for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import re

class EventBasedEntryTrigger:
    """
    Uses news AI to anticipate CPI/FOMC surprises and trigger entries.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Event-Based Entry Trigger.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("EventBasedEntryTrigger")
        self.logger.setLevel(logging.INFO)
        
        self.event_calendar = {}
        
        self.event_types = {
            "FOMC": 1.0,           # Federal Open Market Committee
            "CPI": 0.9,            # Consumer Price Index
            "NFP": 0.9,            # Non-Farm Payrolls
            "GDP": 0.8,            # Gross Domestic Product
            "RETAIL_SALES": 0.7,   # Retail Sales
            "ISM": 0.7,            # Institute for Supply Management
            "JOBLESS_CLAIMS": 0.6, # Initial Jobless Claims
            "PPI": 0.6,            # Producer Price Index
            "DURABLE_GOODS": 0.5,  # Durable Goods Orders
            "CONSUMER_SENTIMENT": 0.5  # Consumer Sentiment
        }
        
        self.sentiment_data = {}
        
        self.surprise_predictions = {}
        
        self.active_triggers = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
    def update(self, current_time):
        """
        Update the entry trigger with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing trigger results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "active_triggers": self.active_triggers,
                "upcoming_events": self._get_upcoming_events(current_time),
                "surprise_predictions": self.surprise_predictions
            }
            
        self._update_event_calendar()
        
        self._update_sentiment_data()
        
        self._predict_surprises()
        
        self._generate_triggers(current_time)
        
        self.last_update = current_time
        
        return {
            "active_triggers": self.active_triggers,
            "upcoming_events": self._get_upcoming_events(current_time),
            "surprise_predictions": self.surprise_predictions
        }
        
    def _update_event_calendar(self):
        """
        Update the economic event calendar.
        """
        
        self.event_calendar = {
            "2023-05-15": [
                {
                    "time": "08:30",
                    "type": "RETAIL_SALES",
                    "description": "US Retail Sales MoM",
                    "forecast": 0.8,
                    "previous": 0.6
                }
            ],
            "2023-05-16": [
                {
                    "time": "08:30",
                    "type": "JOBLESS_CLAIMS",
                    "description": "US Initial Jobless Claims",
                    "forecast": 240,
                    "previous": 232
                }
            ],
            "2023-05-17": [
                {
                    "time": "08:30",
                    "type": "DURABLE_GOODS",
                    "description": "US Durable Goods Orders",
                    "forecast": 0.5,
                    "previous": -0.2
                }
            ],
            "2023-05-24": [
                {
                    "time": "14:00",
                    "type": "FOMC",
                    "description": "FOMC Minutes",
                    "forecast": None,
                    "previous": None
                }
            ],
            "2023-06-02": [
                {
                    "time": "08:30",
                    "type": "NFP",
                    "description": "US Non-Farm Payrolls",
                    "forecast": 180,
                    "previous": 236
                }
            ],
            "2023-06-13": [
                {
                    "time": "08:30",
                    "type": "CPI",
                    "description": "US CPI YoY",
                    "forecast": 4.1,
                    "previous": 4.9
                }
            ],
            "2023-06-14": [
                {
                    "time": "14:00",
                    "type": "FOMC",
                    "description": "FOMC Rate Decision",
                    "forecast": 5.25,
                    "previous": 5.00
                }
            ]
        }
        
    def _update_sentiment_data(self):
        """
        Update sentiment data from news sources.
        """
        
        self.sentiment_data = {
            "FOMC": {
                "recent_articles": [
                    {
                        "title": "Fed officials signal potential pause in rate hikes",
                        "source": "Financial Times",
                        "date": "2023-05-10",
                        "sentiment_score": 0.65  # Positive sentiment (dovish)
                    },
                    {
                        "title": "Fed's Bullard sees two more rate hikes in 2023",
                        "source": "Reuters",
                        "date": "2023-05-08",
                        "sentiment_score": 0.35  # Negative sentiment (hawkish)
                    }
                ],
                "average_sentiment": 0.50,
                "sentiment_change": 0.05
            },
            "CPI": {
                "recent_articles": [
                    {
                        "title": "Analysts expect further cooling in May inflation data",
                        "source": "Bloomberg",
                        "date": "2023-05-11",
                        "sentiment_score": 0.70  # Positive sentiment (lower inflation)
                    },
                    {
                        "title": "Core inflation remains sticky despite headline improvement",
                        "source": "Wall Street Journal",
                        "date": "2023-05-09",
                        "sentiment_score": 0.40  # Slightly negative sentiment
                    }
                ],
                "average_sentiment": 0.55,
                "sentiment_change": 0.10
            },
            "NFP": {
                "recent_articles": [
                    {
                        "title": "Job market showing signs of cooling",
                        "source": "CNBC",
                        "date": "2023-05-12",
                        "sentiment_score": 0.45  # Slightly negative sentiment
                    },
                    {
                        "title": "Labor market remains resilient despite rate hikes",
                        "source": "Bloomberg",
                        "date": "2023-05-10",
                        "sentiment_score": 0.60  # Positive sentiment
                    }
                ],
                "average_sentiment": 0.525,
                "sentiment_change": -0.05
            }
        }
        
    def _predict_surprises(self):
        """
        Predict economic data surprises based on sentiment analysis.
        """
        self.surprise_predictions = {}
        
        for event_type, sentiment in self.sentiment_data.items():
            avg_sentiment = sentiment.get("average_sentiment", 0.5)
            sentiment_change = sentiment.get("sentiment_change", 0.0)
            
            surprise_direction = "POSITIVE" if avg_sentiment > 0.55 else "NEGATIVE" if avg_sentiment < 0.45 else "NEUTRAL"
            
            surprise_magnitude = abs(avg_sentiment - 0.5) * 2.0
            
            num_articles = len(sentiment.get("recent_articles", []))
            confidence = min(1.0, (0.5 + abs(sentiment_change) * 2.0) * (num_articles / 5.0))
            
            self.surprise_predictions[event_type] = {
                "direction": surprise_direction,
                "magnitude": surprise_magnitude,
                "confidence": confidence
            }
        
    def _generate_triggers(self, current_time):
        """
        Generate entry triggers based on upcoming events and surprise predictions.
        
        Parameters:
        - current_time: Current datetime
        """
        self._clear_expired_triggers(current_time)
        
        upcoming_events = self._get_upcoming_events(current_time)
        
        for event in upcoming_events:
            event_type = event["type"]
            event_date = event["date"]
            event_time = event["time"]
            event_importance = self.event_types.get(event_type, 0.0)
            
            if event_type in self.surprise_predictions:
                prediction = self.surprise_predictions[event_type]
                
                if prediction["confidence"] > 0.6 and event_importance > 0.6:
                    trigger_id = f"{event_type}_{event_date}_{event_time}"
                    
                    if trigger_id not in self.active_triggers:
                        event_datetime = datetime.strptime(f"{event_date} {event_time}", "%Y-%m-%d %H:%M")
                        trigger_time = event_datetime - timedelta(hours=24)
                        
                        expiration_time = event_datetime
                        
                        self.active_triggers[trigger_id] = {
                            "event_type": event_type,
                            "event_description": event["description"],
                            "event_date": event_date,
                            "event_time": event_time,
                            "trigger_time": trigger_time,
                            "expiration_time": expiration_time,
                            "direction": prediction["direction"],
                            "magnitude": prediction["magnitude"],
                            "confidence": prediction["confidence"],
                            "importance": event_importance,
                            "status": "PENDING" if trigger_time > current_time else "ACTIVE"
                        }
        
    def _clear_expired_triggers(self, current_time):
        """
        Clear expired triggers.
        
        Parameters:
        - current_time: Current datetime
        """
        expired_triggers = []
        
        for trigger_id, trigger in self.active_triggers.items():
            expiration_time = trigger["expiration_time"]
            if expiration_time < current_time:
                expired_triggers.append(trigger_id)
        
        for trigger_id in expired_triggers:
            del self.active_triggers[trigger_id]
        
    def _get_upcoming_events(self, current_time):
        """
        Get upcoming economic events.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - List of upcoming events
        """
        upcoming_events = []
        
        current_date = current_time.strftime("%Y-%m-%d")
        
        for date, events in self.event_calendar.items():
            if date >= current_date:
                for event in events:
                    event_with_date = event.copy()
                    event_with_date["date"] = date
                    
                    event_date = datetime.strptime(date, "%Y-%m-%d")
                    days_until_event = (event_date.date() - current_time.date()).days
                    event_with_date["days_until"] = days_until_event
                    
                    upcoming_events.append(event_with_date)
        
        upcoming_events.sort(key=lambda x: (x["date"], x["time"]))
        
        return upcoming_events
        
    def get_active_triggers(self):
        """
        Get active entry triggers.
        
        Returns:
        - Dictionary of active triggers
        """
        return self.active_triggers
        
    def get_trigger_signals(self, current_time):
        """
        Get trading signals from active triggers.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - List of trigger signals
        """
        trigger_signals = []
        
        for trigger_id, trigger in self.active_triggers.items():
            if trigger["status"] == "ACTIVE":
                trigger_time = trigger["trigger_time"]
                if trigger_time <= current_time:
                    signal = {
                        "trigger_id": trigger_id,
                        "event_type": trigger["event_type"],
                        "event_description": trigger["event_description"],
                        "event_date": trigger["event_date"],
                        "event_time": trigger["event_time"],
                        "direction": self._convert_direction_to_signal(trigger["direction"]),
                        "confidence": trigger["confidence"] * trigger["importance"],
                        "reason": f"Anticipated {trigger['direction']} surprise in {trigger['event_description']}"
                    }
                    
                    trigger_signals.append(signal)
        
        return trigger_signals
        
    def _convert_direction_to_signal(self, direction):
        """
        Convert surprise direction to trading signal direction.
        
        Parameters:
        - direction: Surprise direction
        
        Returns:
        - Signal direction
        """
        if direction == "POSITIVE":
            return "BUY"
        elif direction == "NEGATIVE":
            return "SELL"
        else:
            return "NEUTRAL"
