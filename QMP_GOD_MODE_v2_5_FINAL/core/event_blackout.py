import pandas as pd
from datetime import datetime, timedelta

class EventBlackoutManager:
    def __init__(self):
        self.blackout_events = {
            "NFP": {"time": "08:30", "duration": 30, "days": [4]},  # Friday 8:30 AM EST
            "FOMC": {"time": "14:00", "duration": 120, "days": [2]}, # Wednesday 2:00 PM EST  
            "CPI": {"time": "08:30", "duration": 60, "days": [1, 2, 3, 4]}, # Various weekdays
            "GDP": {"time": "08:30", "duration": 45, "days": [1, 2, 3, 4]}
        }
        
    def is_blackout_period(self, current_time):
        """Check if current time is during a news blackout period"""
        for event_name, config in self.blackout_events.items():
            if current_time.weekday() in config["days"]:
                event_time = current_time.replace(
                    hour=int(config["time"].split(":")[0]),
                    minute=int(config["time"].split(":")[1]),
                    second=0,
                    microsecond=0
                )
                
                end_time = event_time + timedelta(minutes=config["duration"])
                
                if event_time <= current_time <= end_time:
                    return True, event_name
                    
        return False, None
        
    def check_weekend_market(self, current_time):
        """Prevent trading during weekends"""
        return current_time.weekday() >= 5  # Saturday = 5, Sunday = 6
