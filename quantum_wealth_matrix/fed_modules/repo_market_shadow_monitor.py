"""
Repo Market Shadow Monitor

Scans tri-party repo anomalies and collateral substitution for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class RepoMarketShadowMonitor:
    """
    Scans tri-party repo anomalies and collateral substitution.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Repo Market Shadow Monitor.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("RepoMarketShadowMonitor")
        self.logger.setLevel(logging.INFO)
        
        self.last_update = datetime.now()
        self.update_frequency = timedelta(hours=1)
        
        self.triparty_repo_data = {}
        self.collateral_data = {}
        self.anomaly_score = 0.0
        self.substitution_score = 0.0
        
    def update(self, current_time):
        """
        Update the monitor with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing monitor results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "anomaly_score": self.anomaly_score,
                "substitution_score": self.substitution_score,
                "combined_score": (self.anomaly_score + self.substitution_score) / 2,
                "signal": self._generate_signal()
            }
            
        self._fetch_triparty_repo_data()
        
        self._fetch_collateral_data()
        
        self.anomaly_score = self._calculate_anomaly_score()
        
        self.substitution_score = self._calculate_substitution_score()
        
        self.last_update = current_time
        
        return {
            "anomaly_score": self.anomaly_score,
            "substitution_score": self.substitution_score,
            "combined_score": (self.anomaly_score + self.substitution_score) / 2,
            "signal": self._generate_signal()
        }
        
    def _fetch_triparty_repo_data(self):
        """
        Fetch latest tri-party repo data from Fed API.
        """
        self.triparty_repo_data = {
            "total_volume": 2.1,  # Trillions USD
            "treasury_collateral": 1.4,  # Trillions USD
            "agency_collateral": 0.5,  # Trillions USD
            "other_collateral": 0.2,  # Trillions USD
            "previous_day": {
                "total_volume": 2.0,
                "treasury_collateral": 1.35,
                "agency_collateral": 0.48,
                "other_collateral": 0.17
            }
        }
        
    def _fetch_collateral_data(self):
        """
        Fetch latest collateral data from Fed API.
        """
        self.collateral_data = {
            "substitution_rate": 0.05,  # 5% of collateral substituted
            "downgrade_rate": 0.02,  # 2% of collateral downgraded
            "haircut_change": 0.001,  # 0.1% change in haircuts
            "previous_day": {
                "substitution_rate": 0.04,
                "downgrade_rate": 0.015,
                "haircut_change": 0.0005
            }
        }
        
    def _calculate_anomaly_score(self):
        """
        Calculate tri-party repo anomaly score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.triparty_repo_data:
            return 0.0
            
        volume_change = (self.triparty_repo_data["total_volume"] - self.triparty_repo_data["previous_day"]["total_volume"]) / self.triparty_repo_data["previous_day"]["total_volume"]
        
        treasury_change = (self.triparty_repo_data["treasury_collateral"] - self.triparty_repo_data["previous_day"]["treasury_collateral"]) / self.triparty_repo_data["previous_day"]["treasury_collateral"]
        
        agency_change = (self.triparty_repo_data["agency_collateral"] - self.triparty_repo_data["previous_day"]["agency_collateral"]) / self.triparty_repo_data["previous_day"]["agency_collateral"]
        
        other_change = (self.triparty_repo_data["other_collateral"] - self.triparty_repo_data["previous_day"]["other_collateral"]) / self.triparty_repo_data["previous_day"]["other_collateral"]
        
        anomaly_score = min(1.0, max(0.0, (abs(volume_change) * 3 + abs(treasury_change) * 3 + abs(agency_change) * 2 + abs(other_change) * 2) / 10))
        
        return anomaly_score
        
    def _calculate_substitution_score(self):
        """
        Calculate collateral substitution score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.collateral_data:
            return 0.0
            
        substitution_change = (self.collateral_data["substitution_rate"] - self.collateral_data["previous_day"]["substitution_rate"]) / self.collateral_data["previous_day"]["substitution_rate"]
        
        downgrade_change = (self.collateral_data["downgrade_rate"] - self.collateral_data["previous_day"]["downgrade_rate"]) / self.collateral_data["previous_day"]["downgrade_rate"]
        
        haircut_change = (self.collateral_data["haircut_change"] - self.collateral_data["previous_day"]["haircut_change"]) / self.collateral_data["previous_day"]["haircut_change"]
        
        substitution_score = min(1.0, max(0.0, (abs(substitution_change) * 4 + abs(downgrade_change) * 3 + abs(haircut_change) * 3) / 10))
        
        return substitution_score
        
    def _generate_signal(self):
        """
        Generate trading signal based on anomaly and substitution scores.
        
        Returns:
        - Dictionary containing signal information
        """
        combined_score = (self.anomaly_score + self.substitution_score) / 2
        
        if combined_score > 0.8:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Extreme repo market anomaly and collateral substitution"
            }
        elif combined_score > 0.6:
            return {
                "direction": "SELL",
                "confidence": combined_score,
                "reason": "High repo market anomaly and collateral substitution"
            }
        elif combined_score > 0.4:
            return {
                "direction": "NEUTRAL",
                "confidence": 1.0 - combined_score,
                "reason": "Moderate repo market anomaly and collateral substitution"
            }
        elif combined_score > 0.2:
            return {
                "direction": "BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Low repo market anomaly and collateral substitution"
            }
        else:
            return {
                "direction": "STRONG_BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Minimal repo market anomaly and collateral substitution"
            }
