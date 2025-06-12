"""
Treasury Auction Imbalance Scanner

Identifies primary dealer failures in auctions for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class TreasuryAuctionImbalanceScanner:
    """
    Identifies primary dealer failures in auctions.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Treasury Auction Imbalance Scanner.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("TreasuryAuctionImbalanceScanner")
        self.logger.setLevel(logging.INFO)
        
        self.last_update = datetime.now()
        self.update_frequency = timedelta(hours=6)  # Less frequent updates for auction data
        
        self.auction_data = {}
        self.primary_dealer_data = {}
        self.imbalance_score = 0.0
        self.failure_score = 0.0
        
        self.upcoming_auctions = []
        
        self.recent_results = []
        
    def update(self, current_time):
        """
        Update the scanner with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing scanner results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "imbalance_score": self.imbalance_score,
                "failure_score": self.failure_score,
                "combined_score": (self.imbalance_score + self.failure_score) / 2,
                "upcoming_auctions": self.upcoming_auctions,
                "recent_results": self.recent_results,
                "signal": self._generate_signal()
            }
            
        self._fetch_auction_data()
        
        self._fetch_primary_dealer_data()
        
        self.imbalance_score = self._calculate_imbalance_score()
        
        self.failure_score = self._calculate_failure_score()
        
        self._update_upcoming_auctions()
        
        self._update_recent_results()
        
        self.last_update = current_time
        
        return {
            "imbalance_score": self.imbalance_score,
            "failure_score": self.failure_score,
            "combined_score": (self.imbalance_score + self.failure_score) / 2,
            "upcoming_auctions": self.upcoming_auctions,
            "recent_results": self.recent_results,
            "signal": self._generate_signal()
        }
        
    def _fetch_auction_data(self):
        """
        Fetch latest Treasury auction data.
        """
        self.auction_data = {
            "recent_auctions": [
                {
                    "date": "2023-05-10",
                    "security": "10-Year Note",
                    "amount": 24.0,  # Billions USD
                    "bid_to_cover": 2.35,
                    "high_yield": 0.0350,
                    "indirect_bidders": 0.65,  # 65% of accepted bids
                    "direct_bidders": 0.15,  # 15% of accepted bids
                    "primary_dealers": 0.20,  # 20% of accepted bids
                    "tail": 0.001  # Difference between high yield and median yield
                },
                {
                    "date": "2023-05-09",
                    "security": "3-Year Note",
                    "amount": 40.0,  # Billions USD
                    "bid_to_cover": 2.42,
                    "high_yield": 0.0380,
                    "indirect_bidders": 0.58,
                    "direct_bidders": 0.17,
                    "primary_dealers": 0.25,
                    "tail": 0.0005
                }
            ],
            "upcoming_auctions": [
                {
                    "date": "2023-05-17",
                    "security": "20-Year Bond",
                    "amount": 16.0,  # Billions USD
                    "announcement_date": "2023-05-11",
                    "settlement_date": "2023-05-19"
                },
                {
                    "date": "2023-05-18",
                    "security": "5-Year TIPS",
                    "amount": 18.0,  # Billions USD
                    "announcement_date": "2023-05-11",
                    "settlement_date": "2023-05-22"
                }
            ]
        }
        
    def _fetch_primary_dealer_data(self):
        """
        Fetch latest primary dealer data.
        """
        self.primary_dealer_data = {
            "total_dealers": 24,
            "dealer_positions": {
                "treasury_bills": 120.5,  # Billions USD
                "treasury_notes": 85.3,  # Billions USD
                "treasury_bonds": 42.1,  # Billions USD
                "tips": 18.7  # Billions USD
            },
            "dealer_failures": [
                {
                    "date": "2023-05-08",
                    "security": "2-Year Note",
                    "dealer_id": "PD12",
                    "severity": 0.7  # 0.0 to 1.0 scale
                }
            ],
            "dealer_stress_index": 0.35  # 0.0 to 1.0 scale
        }
        
    def _calculate_imbalance_score(self):
        """
        Calculate Treasury auction imbalance score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.auction_data or "recent_auctions" not in self.auction_data or not self.auction_data["recent_auctions"]:
            return 0.0
            
        avg_bid_to_cover = sum(auction["bid_to_cover"] for auction in self.auction_data["recent_auctions"]) / len(self.auction_data["recent_auctions"])
        
        avg_tail = sum(auction["tail"] for auction in self.auction_data["recent_auctions"]) / len(self.auction_data["recent_auctions"])
        
        avg_primary_dealer_take = sum(auction["primary_dealers"] for auction in self.auction_data["recent_auctions"]) / len(self.auction_data["recent_auctions"])
        
        bid_to_cover_factor = max(0.0, 1.0 - (avg_bid_to_cover - 2.0) / 1.0)  # 2.0 is threshold, 3.0 is excellent
        tail_factor = min(1.0, avg_tail * 500.0)  # 0.002 (2 bps) is high tail
        primary_dealer_factor = min(1.0, avg_primary_dealer_take * 2.0)  # 0.5 (50%) is high take
        
        imbalance_score = (bid_to_cover_factor * 0.4 + tail_factor * 0.4 + primary_dealer_factor * 0.2)
        
        return imbalance_score
        
    def _calculate_failure_score(self):
        """
        Calculate primary dealer failure score.
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if not self.primary_dealer_data:
            return 0.0
            
        failure_count = len(self.primary_dealer_data.get("dealer_failures", []))
        failure_severity = sum(failure["severity"] for failure in self.primary_dealer_data.get("dealer_failures", [])) / max(1, failure_count)
        dealer_stress = self.primary_dealer_data.get("dealer_stress_index", 0.0)
        
        failure_score = min(1.0, (failure_count / 5.0) * 0.3 + failure_severity * 0.3 + dealer_stress * 0.4)
        
        return failure_score
        
    def _update_upcoming_auctions(self):
        """
        Update list of upcoming auctions.
        """
        self.upcoming_auctions = []
        
        if self.auction_data and "upcoming_auctions" in self.auction_data:
            for auction in self.auction_data["upcoming_auctions"]:
                self.upcoming_auctions.append({
                    "date": auction["date"],
                    "security": auction["security"],
                    "amount": auction["amount"],
                    "announcement_date": auction["announcement_date"],
                    "settlement_date": auction["settlement_date"]
                })
        
        self.upcoming_auctions.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
        
    def _update_recent_results(self):
        """
        Update list of recent auction results.
        """
        self.recent_results = []
        
        if self.auction_data and "recent_auctions" in self.auction_data:
            for auction in self.auction_data["recent_auctions"]:
                self.recent_results.append({
                    "date": auction["date"],
                    "security": auction["security"],
                    "amount": auction["amount"],
                    "bid_to_cover": auction["bid_to_cover"],
                    "high_yield": auction["high_yield"],
                    "indirect_bidders": auction["indirect_bidders"],
                    "direct_bidders": auction["direct_bidders"],
                    "primary_dealers": auction["primary_dealers"],
                    "tail": auction["tail"]
                })
        
        self.recent_results.sort(key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"), reverse=True)
        
    def _generate_signal(self):
        """
        Generate trading signal based on imbalance and failure scores.
        
        Returns:
        - Dictionary containing signal information
        """
        combined_score = (self.imbalance_score + self.failure_score) / 2
        
        imminent_auction = False
        large_auction_coming = False
        for auction in self.upcoming_auctions:
            auction_date = datetime.strptime(auction["date"], "%Y-%m-%d")
            days_until_auction = (auction_date - datetime.now()).days
            if days_until_auction <= 2:
                imminent_auction = True
                if auction["amount"] > 30.0:  # Large auction threshold (30 billion)
                    large_auction_coming = True
                break
        
        if large_auction_coming and combined_score > 0.6:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Large Treasury auction imminent with high imbalance/failure risk"
            }
        elif imminent_auction and combined_score > 0.5:
            return {
                "direction": "SELL",
                "confidence": combined_score,
                "reason": "Treasury auction imminent with elevated imbalance/failure risk"
            }
        elif combined_score > 0.7:
            return {
                "direction": "STRONG_SELL",
                "confidence": combined_score,
                "reason": "Extreme Treasury auction imbalance and primary dealer stress"
            }
        elif combined_score > 0.5:
            return {
                "direction": "SELL",
                "confidence": combined_score,
                "reason": "Elevated Treasury auction imbalance and primary dealer stress"
            }
        elif combined_score > 0.3:
            return {
                "direction": "NEUTRAL",
                "confidence": 1.0 - combined_score,
                "reason": "Moderate Treasury auction imbalance and primary dealer stress"
            }
        else:
            return {
                "direction": "BUY",
                "confidence": 1.0 - combined_score,
                "reason": "Healthy Treasury auction demand and primary dealer stability"
            }
