"""
Overnight Reserve Drift Tracker

Detects Fed balance sheet reserve leakage for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class OvernightReserveDriftTracker:
    """
    Detects Fed balance sheet reserve leakage.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Overnight Reserve Drift Tracker.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("OvernightReserveDriftTracker")
        self.logger.setLevel(logging.INFO)
        
        self.last_update = datetime.now()
        self.update_frequency = timedelta(hours=12)  # Less frequent updates for balance sheet data
        
        self.reserve_data = {}
        self.balance_sheet_data = {}
        self.drift_score = 0.0
        self.leakage_score = 0.0
        
        self.reserve_trends = {
            "1d": 0.0,
            "1w": 0.0,
            "1m": 0.0,
            "3m": 0.0
        }
        
        self.balance_sheet_trends = {
            "1d": 0.0,
            "1w": 0.0,
            "1m": 0.0,
            "3m": 0.0
        }
        
    def update(self, current_time):
        """
        Update the tracker with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing tracker results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "drift_score": self.drift_score,
                "leakage_score": self.leakage_score,
                "combined_score": (self.drift_score + self.leakage_score) / 2,
                "reserve_trends": self.reserve_trends,
                "balance_sheet_trends": self.balance_sheet_trends,
                "signal": self._generate_signal()
            }
            
        self._fetch_reserve_data()
        
        self._fetch_balance_sheet_data()
        
        self._calculate_reserve_trends()
        
        self._calculate_balance_sheet_trends()
        
        self.drift_score = self._calculate_drift_score()
        
        self.leakage_score = self._calculate_leakage_score()
        
        self.last_update = current_time
        
        return {
            "drift_score": self.drift_score,
            "leakage_score": self.leakage_score,
            "combined_score": (self.drift_score + self.leakage_score) / 2,
            "reserve_trends": self.reserve_trends,
            "balance_sheet_trends": self.balance_sheet_trends,
            "signal": self._generate_signal()
        }
        
    def _fetch_reserve_data(self):
        """
        Fetch latest reserve data from Fed API.
        """
        self.reserve_data = {
            "current": {
                "date": "2023-05-10",
                "total_reserves": 3.2,  # Trillions USD
                "excess_reserves": 2.8,  # Trillions USD
                "required_reserves": 0.4,  # Trillions USD
                "overnight_repo": 0.15,  # Trillions USD
                "reverse_repo": 1.8  # Trillions USD
            },
            "historical": {
                "1d_ago": {
                    "date": "2023-05-09",
                    "total_reserves": 3.22,
                    "excess_reserves": 2.82,
                    "required_reserves": 0.4,
                    "overnight_repo": 0.14,
                    "reverse_repo": 1.78
                },
                "1w_ago": {
                    "date": "2023-05-03",
                    "total_reserves": 3.25,
                    "excess_reserves": 2.85,
                    "required_reserves": 0.4,
                    "overnight_repo": 0.13,
                    "reverse_repo": 1.75
                },
                "1m_ago": {
                    "date": "2023-04-10",
                    "total_reserves": 3.3,
                    "excess_reserves": 2.9,
                    "required_reserves": 0.4,
                    "overnight_repo": 0.12,
                    "reverse_repo": 1.7
                },
                "3m_ago": {
                    "date": "2023-02-10",
                    "total_reserves": 3.4,
                    "excess_reserves": 3.0,
                    "required_reserves": 0.4,
                    "overnight_repo": 0.1,
                    "reverse_repo": 1.6
                }
            }
        }
        
    def _fetch_balance_sheet_data(self):
        """
        Fetch latest Fed balance sheet data.
        """
        self.balance_sheet_data = {
            "current": {
                "date": "2023-05-10",
                "total_assets": 8.5,  # Trillions USD
                "securities_held": 7.2,  # Trillions USD
                "loans": 0.3,  # Trillions USD
                "other_assets": 1.0,  # Trillions USD
                "total_liabilities": 8.5,  # Trillions USD
                "currency_in_circulation": 2.3,  # Trillions USD
                "reserve_balances": 3.2,  # Trillions USD
                "other_liabilities": 3.0  # Trillions USD
            },
            "historical": {
                "1d_ago": {
                    "date": "2023-05-09",
                    "total_assets": 8.51,
                    "securities_held": 7.21,
                    "loans": 0.3,
                    "other_assets": 1.0,
                    "total_liabilities": 8.51,
                    "currency_in_circulation": 2.3,
                    "reserve_balances": 3.22,
                    "other_liabilities": 2.99
                },
                "1w_ago": {
                    "date": "2023-05-03",
                    "total_assets": 8.55,
                    "securities_held": 7.25,
                    "loans": 0.3,
                    "other_assets": 1.0,
                    "total_liabilities": 8.55,
                    "currency_in_circulation": 2.29,
                    "reserve_balances": 3.25,
                    "other_liabilities": 3.01
                },
                "1m_ago": {
                    "date": "2023-04-10",
                    "total_assets": 8.6,
                    "securities_held": 7.3,
                    "loans": 0.3,
                    "other_assets": 1.0,
                    "total_liabilities": 8.6,
                    "currency_in_circulation": 2.28,
                    "reserve_balances": 3.3,
                    "other_liabilities": 3.02
                },
                "3m_ago": {
                    "date": "2023-02-10",
                    "total_assets": 8.7,
                    "securities_held": 7.4,
                    "loans": 0.3,
                    "other_assets": 1.0,
                    "total_liabilities": 8.7,
                    "currency_in_circulation": 2.25,
                    "reserve_balances": 3.4,
                    "other_liabilities": 3.05
                }
            }
        }
        
    def _calculate_reserve_trends(self):
        """
        Calculate reserve trends over different time periods.
        """
        if not self.reserve_data or "current" not in self.reserve_data or "historical" not in self.reserve_data:
            return
            
        current = self.reserve_data["current"]
        historical = self.reserve_data["historical"]
        
        if "1d_ago" in historical:
            self.reserve_trends["1d"] = (current["total_reserves"] - historical["1d_ago"]["total_reserves"]) / historical["1d_ago"]["total_reserves"]
        
        if "1w_ago" in historical:
            self.reserve_trends["1w"] = (current["total_reserves"] - historical["1w_ago"]["total_reserves"]) / historical["1w_ago"]["total_reserves"]
        
        if "1m_ago" in historical:
            self.reserve_trends["1m"] = (current["total_reserves"] - historical["1m_ago"]["total_reserves"]) / historical["1m_ago"]["total_reserves"]
        
        if "3m_ago" in historical:
            self.reserve_trends["3m"] = (current["total_reserves"] - historical["3m_ago"]["total_reserves"]) / historical["3m_ago"]["total_reserves"]
        
    def _calculate_balance_sheet_trends(self):
        """
        Calculate balance sheet trends over different time periods.
        """
        if not self.balance_sheet_data or "current" not in self.balance_sheet_data or "historical" not in self.balance_sheet_data:
            return
            
        current = self.balance_sheet_data["current"]
        historical = self.balance_sheet_data["historical"]
        
        if "1d_ago" in historical:
            self.balance_sheet_trends["1d"] = (current["total_assets"] - historical["1d_ago"]["total_assets"]) / historical["1d_ago"]["total_assets"]
        
        if "1w_ago" in historical:
            self.balance_sheet_trends["1w"] = (current["total_assets"] - historical["1w_ago"]["total_assets"]) / historical["1w_ago"]["total_assets"]
        
        if "1m_ago" in historical:
            self.balance_sheet_trends["1m"] = (current["total_assets"] - historical["1m_ago"]["total_assets"]) / historical["1m_ago"]["total_assets"]
        
        if "3m_ago" in historical:
            self.balance_sheet_trends["3m"] = (current["total_assets"] - historical["3m_ago"]["total_assets"]) / historical["3m_ago"]["total_assets"]
        
    def _calculate_drift_score(self):
        """
        Calculate reserve drift score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.reserve_trends:
            return 0.0
            
        weighted_trend = (
            self.reserve_trends["1d"] * 0.4 +
            self.reserve_trends["1w"] * 0.3 +
            self.reserve_trends["1m"] * 0.2 +
            self.reserve_trends["3m"] * 0.1
        )
        
        drift_score = min(1.0, max(0.0, 0.5 - weighted_trend * 10.0))
        
        return drift_score
        
    def _calculate_leakage_score(self):
        """
        Calculate balance sheet leakage score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.balance_sheet_trends or not self.reserve_data or "current" not in self.reserve_data:
            return 0.0
            
        weighted_trend = (
            self.balance_sheet_trends["1d"] * 0.4 +
            self.balance_sheet_trends["1w"] * 0.3 +
            self.balance_sheet_trends["1m"] * 0.2 +
            self.balance_sheet_trends["3m"] * 0.1
        )
        
        reverse_repo_ratio = self.reserve_data["current"]["reverse_repo"] / self.reserve_data["current"]["total_reserves"]
        
        leakage_score = min(1.0, max(0.0, (0.5 - weighted_trend * 10.0) * 0.7 + reverse_repo_ratio * 0.3))
        
        return leakage_score
        
    def _generate_signal(self):
        """
        Generate trading signal based on drift and leakage scores.
        
        Returns:
        - Dictionary containing signal information
        """
        combined_score = (self.drift_score + self.leakage_score) / 2
        
        rapid_reserve_change = abs(self.reserve_trends["1d"]) > 0.01  # 1% daily change
        rapid_balance_sheet_change = abs(self.balance_sheet_trends["1d"]) > 0.005  # 0.5% daily change
        
        if rapid_reserve_change and rapid_balance_sheet_change and combined_score > 0.7:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Rapid Fed balance sheet contraction and reserve leakage"
            }
        elif combined_score > 0.8:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Extreme Fed balance sheet contraction and reserve leakage"
            }
        elif combined_score > 0.6:
            return {
                "direction": "SELL",
                "confidence": combined_score,
                "reason": "Significant Fed balance sheet contraction and reserve leakage"
            }
        elif combined_score > 0.4:
            return {
                "direction": "NEUTRAL",
                "confidence": 1.0 - combined_score,
                "reason": "Moderate Fed balance sheet contraction and reserve leakage"
            }
        elif combined_score > 0.2:
            return {
                "direction": "BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Minimal Fed balance sheet contraction and reserve leakage"
            }
        else:
            return {
                "direction": "STRONG_BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Fed balance sheet expansion and reserve growth"
            }
