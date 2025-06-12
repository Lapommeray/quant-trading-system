"""
Interest Rate Pulse Integrator

Tracks yield curve tremors (e.g., 2s10s inversion) for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class InterestRatePulseIntegrator:
    """
    Tracks yield curve tremors and integrates interest rate signals.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Interest Rate Pulse Integrator.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("InterestRatePulseIntegrator")
        self.logger.setLevel(logging.INFO)
        
        self.last_update = datetime.now()
        self.update_frequency = timedelta(hours=1)
        
        self.yield_curve_data = {}
        self.rate_expectations_data = {}
        self.inversion_score = 0.0
        self.pulse_score = 0.0
        
        self.spreads = {
            "2s10s": 0.0,  # 10-year minus 2-year
            "3m10y": 0.0,  # 10-year minus 3-month
            "5s30s": 0.0,  # 30-year minus 5-year
            "2s5s": 0.0,   # 5-year minus 2-year
            "10s30s": 0.0  # 30-year minus 10-year
        }
        
        self.inversions = {
            "2s10s": False,
            "3m10y": False,
            "5s30s": False,
            "2s5s": False,
            "10s30s": False
        }
        
        self.inversion_durations = {
            "2s10s": 0,
            "3m10y": 0,
            "5s30s": 0,
            "2s5s": 0,
            "10s30s": 0
        }
        
    def update(self, current_time):
        """
        Update the integrator with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing integrator results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "inversion_score": self.inversion_score,
                "pulse_score": self.pulse_score,
                "combined_score": (self.inversion_score + self.pulse_score) / 2,
                "spreads": self.spreads,
                "inversions": self.inversions,
                "inversion_durations": self.inversion_durations,
                "signal": self._generate_signal()
            }
            
        self._fetch_yield_curve_data()
        
        self._fetch_rate_expectations_data()
        
        self._calculate_spreads()
        
        self._update_inversion_status()
        
        self.inversion_score = self._calculate_inversion_score()
        
        self.pulse_score = self._calculate_pulse_score()
        
        self.last_update = current_time
        
        return {
            "inversion_score": self.inversion_score,
            "pulse_score": self.pulse_score,
            "combined_score": (self.inversion_score + self.pulse_score) / 2,
            "spreads": self.spreads,
            "inversions": self.inversions,
            "inversion_durations": self.inversion_durations,
            "signal": self._generate_signal()
        }
        
    def _fetch_yield_curve_data(self):
        """
        Fetch latest yield curve data from Treasury API.
        """
        self.yield_curve_data = {
            "3m": 0.0520,  # 3-month yield
            "2y": 0.0480,  # 2-year yield
            "5y": 0.0460,  # 5-year yield
            "10y": 0.0440, # 10-year yield
            "30y": 0.0430  # 30-year yield
        }
        
    def _fetch_rate_expectations_data(self):
        """
        Fetch latest rate expectations data from Fed Funds Futures.
        """
        self.rate_expectations_data = {
            "current_rate": 0.0525,  # Current Fed Funds Rate
            "next_meeting": 0.0500,  # Expected rate after next meeting
            "6m_forward": 0.0450,    # Expected rate 6 months forward
            "12m_forward": 0.0400    # Expected rate 12 months forward
        }
        
    def _calculate_spreads(self):
        """
        Calculate key yield curve spreads.
        """
        if not self.yield_curve_data:
            return
            
        self.spreads["2s10s"] = self.yield_curve_data["10y"] - self.yield_curve_data["2y"]
        self.spreads["3m10y"] = self.yield_curve_data["10y"] - self.yield_curve_data["3m"]
        self.spreads["5s30s"] = self.yield_curve_data["30y"] - self.yield_curve_data["5y"]
        self.spreads["2s5s"] = self.yield_curve_data["5y"] - self.yield_curve_data["2y"]
        self.spreads["10s30s"] = self.yield_curve_data["30y"] - self.yield_curve_data["10y"]
        
    def _update_inversion_status(self):
        """
        Update inversion status and durations.
        """
        if not self.spreads:
            return
            
        self.inversions["2s10s"] = self.spreads["2s10s"] < 0
        self.inversions["3m10y"] = self.spreads["3m10y"] < 0
        self.inversions["5s30s"] = self.spreads["5s30s"] < 0
        self.inversions["2s5s"] = self.spreads["2s5s"] < 0
        self.inversions["10s30s"] = self.spreads["10s30s"] < 0
        
        for spread in self.inversions:
            if self.inversions[spread]:
                self.inversion_durations[spread] += 1
            else:
                self.inversion_durations[spread] = 0
        
    def _calculate_inversion_score(self):
        """
        Calculate yield curve inversion score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.inversions:
            return 0.0
            
        inversion_count = sum(1 for inv in self.inversions.values() if inv)
        
        weighted_duration = 0.0
        for spread in self.inversion_durations:
            if self.inversions[spread]:
                if spread == "2s10s":
                    weight = 0.4  # 2s10s is most important
                elif spread == "3m10y":
                    weight = 0.3  # 3m10y is second most important
                else:
                    weight = 0.1  # Others are less important
                    
                duration_effect = 2.0 / (1.0 + np.exp(-0.05 * self.inversion_durations[spread])) - 1.0
                weighted_duration += weight * duration_effect
        
        inversion_score = min(1.0, max(0.0, (inversion_count / len(self.inversions) * 0.5 + weighted_duration * 0.5)))
        
        return inversion_score
        
    def _calculate_pulse_score(self):
        """
        Calculate interest rate pulse score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.rate_expectations_data:
            return 0.0
            
        current_rate = self.rate_expectations_data["current_rate"]
        next_meeting_change = self.rate_expectations_data["next_meeting"] - current_rate
        six_month_change = self.rate_expectations_data["6m_forward"] - current_rate
        twelve_month_change = self.rate_expectations_data["12m_forward"] - current_rate
        
        raw_pulse_score = 0.5 - (next_meeting_change * 5.0 + six_month_change * 3.0 + twelve_month_change * 2.0) / 10.0
        
        pulse_score = min(1.0, max(0.0, raw_pulse_score))
        
        return pulse_score
        
    def _generate_signal(self):
        """
        Generate trading signal based on inversion and pulse scores.
        
        Returns:
        - Dictionary containing signal information
        """
        
        if self.inversion_score > 0.7 and self.pulse_score > 0.7:
            return {
                "direction": "STRONG_SELL",
                "confidence": max(self.inversion_score, self.pulse_score),
                "reason": "Severe yield curve inversion with dovish rate expectations (recession signal)"
            }
        elif self.inversion_score > 0.5:
            return {
                "direction": "SELL",
                "confidence": self.inversion_score,
                "reason": "Significant yield curve inversion"
            }
        elif self.pulse_score > 0.7 and self.inversion_score < 0.3:
            return {
                "direction": "STRONG_BUY",
                "confidence": self.pulse_score,
                "reason": "Strongly dovish rate expectations without yield curve inversion"
            }
        elif self.pulse_score > 0.5:
            return {
                "direction": "BUY",
                "confidence": self.pulse_score,
                "reason": "Dovish rate expectations"
            }
        elif self.pulse_score < 0.3:
            return {
                "direction": "SELL",
                "confidence": 1.0 - self.pulse_score,
                "reason": "Hawkish rate expectations"
            }
        else:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.5,
                "reason": "Balanced yield curve and rate expectations"
            }
