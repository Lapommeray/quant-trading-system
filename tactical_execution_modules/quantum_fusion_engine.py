"""
Quantum Fusion Engine

Fuses all core Fed modules into one decision engine for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_wealth_matrix.fed_modules.liquidity_arbitrage_decoder import FedLiquidityArbitrageDecoder
from quantum_wealth_matrix.fed_modules.repo_market_shadow_monitor import RepoMarketShadowMonitor
from quantum_wealth_matrix.fed_modules.interest_rate_pulse_integrator import InterestRatePulseIntegrator
from quantum_wealth_matrix.fed_modules.federal_docket_forecast_ai import FederalDocketForecastAI
from quantum_wealth_matrix.fed_modules.treasury_auction_imbalance_scanner import TreasuryAuctionImbalanceScanner
from quantum_wealth_matrix.fed_modules.overnight_reserve_drift_tracker import OvernightReserveDriftTracker

class QuantumFusionEngine:
    """
    Fuses all core Fed modules into one decision engine.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Quantum Fusion Engine.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("QuantumFusionEngine")
        self.logger.setLevel(logging.INFO)
        
        self.fed_modules = {
            "liquidity_arbitrage": FedLiquidityArbitrageDecoder(algorithm),
            "repo_market": RepoMarketShadowMonitor(algorithm),
            "interest_rate": InterestRatePulseIntegrator(algorithm),
            "federal_docket": FederalDocketForecastAI(algorithm),
            "treasury_auction": TreasuryAuctionImbalanceScanner(algorithm),
            "overnight_reserve": OvernightReserveDriftTracker(algorithm)
        }
        
        self.module_weights = {
            "liquidity_arbitrage": 0.20,
            "repo_market": 0.15,
            "interest_rate": 0.25,
            "federal_docket": 0.15,
            "treasury_auction": 0.15,
            "overnight_reserve": 0.10
        }
        
        self.fusion_score = 0.0
        self.fusion_direction = "NEUTRAL"
        self.fusion_confidence = 0.0
        self.fusion_reason = ""
        
        self.module_results = {}
        
        self.last_update = datetime.now()
        
    def update(self, current_time):
        """
        Update the fusion engine with latest data from all modules.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing fusion results
        """
        self.module_results = {}
        for module_name, module in self.fed_modules.items():
            self.module_results[module_name] = module.update(current_time)
        
        self._fuse_module_results()
        
        self._generate_fusion_signal()
        
        self.last_update = current_time
        
        return {
            "fusion_score": self.fusion_score,
            "fusion_direction": self.fusion_direction,
            "fusion_confidence": self.fusion_confidence,
            "fusion_reason": self.fusion_reason,
            "module_results": self.module_results,
            "module_weights": self.module_weights
        }
        
    def _fuse_module_results(self):
        """
        Fuse results from all Fed modules.
        """
        if not self.module_results:
            return
            
        weighted_score = 0.0
        total_weight = 0.0
        
        for module_name, result in self.module_results.items():
            if module_name in self.module_weights:
                if "combined_score" in result:
                    score = result["combined_score"]
                elif "imbalance_score" in result and "distortion_score" in result:
                    score = (result["imbalance_score"] + result["distortion_score"]) / 2
                elif "volatility_score" in result and "market_impact_score" in result:
                    score = (result["volatility_score"] + result["market_impact_score"]) / 2
                elif "inversion_score" in result and "pulse_score" in result:
                    score = (result["inversion_score"] + result["pulse_score"]) / 2
                elif "drift_score" in result and "leakage_score" in result:
                    score = (result["drift_score"] + result["leakage_score"]) / 2
                else:
                    score = 0.5
                
                weight = self.module_weights[module_name]
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            self.fusion_score = weighted_score / total_weight
        else:
            self.fusion_score = 0.5
        
    def _generate_fusion_signal(self):
        """
        Generate fusion signal based on fused module results.
        """
        if self.fusion_score > 0.8:
            self.fusion_direction = "STRONG_SELL"
            self.fusion_confidence = self.fusion_score
            self.fusion_reason = "Extreme Fed risk signals across multiple modules"
        elif self.fusion_score > 0.65:
            self.fusion_direction = "SELL"
            self.fusion_confidence = self.fusion_score
            self.fusion_reason = "Elevated Fed risk signals across multiple modules"
        elif self.fusion_score > 0.45:
            self.fusion_direction = "NEUTRAL"
            self.fusion_confidence = 1.0 - abs(self.fusion_score - 0.5) * 2
            self.fusion_reason = "Mixed Fed signals with no clear direction"
        elif self.fusion_score > 0.3:
            self.fusion_direction = "BUY"
            self.fusion_confidence = 1.0 - self.fusion_score
            self.fusion_reason = "Positive Fed signals across multiple modules"
        else:
            self.fusion_direction = "STRONG_BUY"
            self.fusion_confidence = 1.0 - self.fusion_score
            self.fusion_reason = "Extremely positive Fed signals across multiple modules"
        
    def adjust_module_weights(self, new_weights):
        """
        Adjust the weights of Fed modules.
        
        Parameters:
        - new_weights: Dictionary containing new weights for modules
        
        Returns:
        - Boolean indicating success
        """
        if not isinstance(new_weights, dict):
            return False
            
        total_weight = sum(weight for module, weight in new_weights.items() if module in self.module_weights)
        if abs(total_weight - 1.0) > 0.001:  # Allow small rounding errors
            for module in new_weights:
                if module in self.module_weights:
                    new_weights[module] /= total_weight
        
        for module, weight in new_weights.items():
            if module in self.module_weights:
                self.module_weights[module] = weight
        
        return True
        
    def get_module_signals(self):
        """
        Get signals from all Fed modules.
        
        Returns:
        - Dictionary containing signals from all modules
        """
        signals = {}
        
        for module_name, result in self.module_results.items():
            if "signal" in result:
                signals[module_name] = result["signal"]
        
        return signals
        
    def get_module_scores(self):
        """
        Get scores from all Fed modules.
        
        Returns:
        - Dictionary containing scores from all modules
        """
        scores = {}
        
        for module_name, result in self.module_results.items():
            module_score = {}
            
            if "imbalance_score" in result:
                module_score["imbalance_score"] = result["imbalance_score"]
            if "distortion_score" in result:
                module_score["distortion_score"] = result["distortion_score"]
            if "volatility_score" in result:
                module_score["volatility_score"] = result["volatility_score"]
            if "market_impact_score" in result:
                module_score["market_impact_score"] = result["market_impact_score"]
            if "inversion_score" in result:
                module_score["inversion_score"] = result["inversion_score"]
            if "pulse_score" in result:
                module_score["pulse_score"] = result["pulse_score"]
            if "drift_score" in result:
                module_score["drift_score"] = result["drift_score"]
            if "leakage_score" in result:
                module_score["leakage_score"] = result["leakage_score"]
            if "combined_score" in result:
                module_score["combined_score"] = result["combined_score"]
                
            scores[module_name] = module_score
        
        return scores
