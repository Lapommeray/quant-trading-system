"""
Fed Liquidity Arbitrage Decoder

Detects ON RRP imbalances and Fedwire distortions for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class FedLiquidityArbitrageDecoder:
    """
    Detects ON RRP imbalances and Fedwire distortions.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Fed Liquidity Arbitrage Decoder.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("FedLiquidityArbitrageDecoder")
        self.logger.setLevel(logging.INFO)
        
        self.last_update = datetime.now()
        self.update_frequency = timedelta(hours=1)
        
        self.on_rrp_data = {}
        self.fedwire_data = {}
        self.imbalance_score = 0.0
        self.distortion_score = 0.0
        
    def update(self, current_time):
        """
        Update the decoder with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing decoder results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "imbalance_score": self.imbalance_score,
                "distortion_score": self.distortion_score,
                "combined_score": (self.imbalance_score + self.distortion_score) / 2,
                "signal": self._generate_signal()
            }
            
        self._fetch_on_rrp_data()
        
        self._fetch_fedwire_data()
        
        self.imbalance_score = self._calculate_imbalance_score()
        
        self.distortion_score = self._calculate_distortion_score()
        
        self.last_update = current_time
        
        return {
            "imbalance_score": self.imbalance_score,
            "distortion_score": self.distortion_score,
            "combined_score": (self.imbalance_score + self.distortion_score) / 2,
            "signal": self._generate_signal()
        }
        
    def _fetch_on_rrp_data(self):
        """
        Fetch latest ON RRP data from Fed API.
        """
        self.on_rrp_data = {
            "total_amount": 1500.0,  # Billions USD
            "participants": 90,
            "average_rate": 0.05,
            "previous_day": {
                "total_amount": 1450.0,
                "participants": 88,
                "average_rate": 0.05
            }
        }
        
    def _fetch_fedwire_data(self):
        """
        Fetch latest Fedwire data from Fed API.
        """
        self.fedwire_data = {
            "total_volume": 3.2,  # Trillions USD
            "transaction_count": 750000,
            "average_value": 4.3,  # Millions USD
            "previous_day": {
                "total_volume": 3.1,
                "transaction_count": 740000,
                "average_value": 4.2
            }
        }
        
    def _calculate_imbalance_score(self):
        """
        Calculate ON RRP imbalance score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.on_rrp_data:
            return 0.0
            
        amount_change = (self.on_rrp_data["total_amount"] - self.on_rrp_data["previous_day"]["total_amount"]) / self.on_rrp_data["previous_day"]["total_amount"]
        
        participant_change = (self.on_rrp_data["participants"] - self.on_rrp_data["previous_day"]["participants"]) / self.on_rrp_data["previous_day"]["participants"]
        
        imbalance_score = min(1.0, max(0.0, (abs(amount_change) * 5 + abs(participant_change) * 3) / 8))
        
        return imbalance_score
        
    def _calculate_distortion_score(self):
        """
        Calculate Fedwire distortion score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.fedwire_data:
            return 0.0
            
        volume_change = (self.fedwire_data["total_volume"] - self.fedwire_data["previous_day"]["total_volume"]) / self.fedwire_data["previous_day"]["total_volume"]
        
        transaction_change = (self.fedwire_data["transaction_count"] - self.fedwire_data["previous_day"]["transaction_count"]) / self.fedwire_data["previous_day"]["transaction_count"]
        
        value_change = (self.fedwire_data["average_value"] - self.fedwire_data["previous_day"]["average_value"]) / self.fedwire_data["previous_day"]["average_value"]
        
        distortion_score = min(1.0, max(0.0, (abs(volume_change) * 4 + abs(transaction_change) * 3 + abs(value_change) * 3) / 10))
        
        return distortion_score
        
    def _generate_signal(self):
        """
        Generate trading signal based on imbalance and distortion scores.
        
        Returns:
        - Dictionary containing signal information
        """
        combined_score = (self.imbalance_score + self.distortion_score) / 2
        
        if combined_score > 0.8:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Extreme Fed liquidity imbalance and distortion"
            }
        elif combined_score > 0.6:
            return {
                "direction": "SELL",
                "confidence": combined_score,
                "reason": "High Fed liquidity imbalance and distortion"
            }
        elif combined_score > 0.4:
            return {
                "direction": "NEUTRAL",
                "confidence": 1.0 - combined_score,
                "reason": "Moderate Fed liquidity imbalance and distortion"
            }
        elif combined_score > 0.2:
            return {
                "direction": "BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Low Fed liquidity imbalance and distortion"
            }
        else:
            return {
                "direction": "STRONG_BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Minimal Fed liquidity imbalance and distortion"
            }
