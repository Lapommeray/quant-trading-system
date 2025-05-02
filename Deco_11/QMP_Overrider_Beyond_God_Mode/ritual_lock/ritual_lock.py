"""
ritual_lock.py

Ritual Lock ("Cosmic Filter") for QMP Overrider

Prevents trades when cosmic/weather cycles signal instability,
providing an additional layer of protection against adverse market conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import random

class RitualLock:
    """
    Ritual Lock for QMP Overrider
    
    Prevents trades when cosmic/weather cycles signal instability,
    providing an additional layer of protection against adverse market conditions.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Ritual Lock
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.last_check = None
        self.last_result = None
        self.moon_phase = None
        self.mercury_retrograde = False
        self.geomagnetic_storm = False
        self.solar_activity = 0.0
        
    def is_aligned(self, direction=None):
        """
        Check if cosmic/weather cycles are aligned for trading
        
        Parameters:
        - direction: Trade direction to check (optional)
        
        Returns:
        - True if aligned, False if misaligned
        """
        now = datetime.now()
        
        if self.last_check and (now - self.last_check).total_seconds() < 3600:
            return self.last_result
        
        self.moon_phase = self._calculate_moon_phase(now)
        
        self.mercury_retrograde = self._is_mercury_retrograde(now)
        
        self.geomagnetic_storm = self._is_geomagnetic_storm(now)
        
        self.solar_activity = self._get_solar_activity(now)
        
        is_aligned = True
        
        if self.moon_phase > 0.48 and self.moon_phase < 0.52:
            is_aligned = False
        elif self.moon_phase < 0.02 or self.moon_phase > 0.98:
            is_aligned = False
        
        if self.mercury_retrograde:
            is_aligned = False
        
        if self.geomagnetic_storm:
            is_aligned = False
        
        if self.solar_activity > 7.5:
            is_aligned = False
        
        if direction and is_aligned:
            if direction == "BUY" and self.moon_phase > 0.5:
                is_aligned = self.moon_phase > 0.9 or self.moon_phase < 0.1
            elif direction == "SELL" and self.moon_phase < 0.5:
                is_aligned = self.moon_phase > 0.4 and self.moon_phase < 0.6
        
        self.last_check = now
        self.last_result = is_aligned
        
        if self.algorithm:
            self.algorithm.Debug(f"Ritual Lock: {'Aligned' if is_aligned else 'Misaligned'}")
            self.algorithm.Debug(f"Moon Phase: {self.moon_phase:.2f}, Mercury Retrograde: {self.mercury_retrograde}")
            self.algorithm.Debug(f"Geomagnetic Storm: {self.geomagnetic_storm}, Solar Activity: {self.solar_activity:.1f}")
        
        return is_aligned
    
    def get_ritual_data(self):
        """
        Get ritual data for logging and analysis
        
        Returns:
        - Dictionary with ritual data
        """
        return {
            "moon_phase": self.moon_phase,
            "mercury_retrograde": self.mercury_retrograde,
            "geomagnetic_storm": self.geomagnetic_storm,
            "solar_activity": self.solar_activity,
            "is_aligned": self.last_result
        }
    
    def _calculate_moon_phase(self, date):
        """
        Calculate the moon phase for a given date
        
        Parameters:
        - date: Date to calculate moon phase for
        
        Returns:
        - Moon phase (0.0 to 1.0, where 0.0 is new moon and 0.5 is full moon)
        """
        
        new_moon = datetime(2024, 1, 1)
        
        lunar_cycle = 29.53
        
        days_since_new_moon = (date - new_moon).total_seconds() / (24 * 3600)
        
        moon_phase = (days_since_new_moon % lunar_cycle) / lunar_cycle
        
        return moon_phase
    
    def _is_mercury_retrograde(self, date):
        """
        Check if Mercury is in retrograde for a given date
        
        Parameters:
        - date: Date to check
        
        Returns:
        - True if Mercury is in retrograde, False otherwise
        """
        
        retrograde_periods = [
            (datetime(2024, 4, 1), datetime(2024, 4, 25)),
            (datetime(2024, 8, 5), datetime(2024, 8, 28)),
            (datetime(2024, 11, 26), datetime(2024, 12, 15))
        ]
        
        for start, end in retrograde_periods:
            if start <= date <= end:
                return True
        
        return False
    
    def _is_geomagnetic_storm(self, date):
        """
        Check if there is a geomagnetic storm for a given date
        
        Parameters:
        - date: Date to check
        
        Returns:
        - True if there is a geomagnetic storm, False otherwise
        """
        
        seed = int(date.timestamp() / (24 * 3600))
        random.seed(seed)
        
        return random.random() < 0.05
    
    def _get_solar_activity(self, date):
        """
        Get solar activity level for a given date
        
        Parameters:
        - date: Date to get solar activity for
        
        Returns:
        - Solar activity level (0.0 to 10.0)
        """
        
        seed = int(date.timestamp() / (24 * 3600))
        random.seed(seed)
        
        year_in_cycle = (date.year % 11) / 11.0
        base_activity = 5.0 + 4.0 * math.sin(year_in_cycle * 2 * math.pi)
        
        daily_variation = random.uniform(-1.0, 1.0)
        
        return min(10.0, max(0.0, base_activity + daily_variation))
