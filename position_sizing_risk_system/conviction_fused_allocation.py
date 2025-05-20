"""
Conviction-Fused Allocation

Integrates Fed + technical signals for capital assignment for the QMP Overrider system.
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

class ConvictionFusedAllocation:
    """
    Integrates Fed + technical signals for capital assignment.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Conviction-Fused Allocation.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("ConvictionFusedAllocation")
        self.logger.setLevel(logging.INFO)
        
        self.fed_modules = {
            "liquidity_arbitrage": FedLiquidityArbitrageDecoder(algorithm),
            "repo_market": RepoMarketShadowMonitor(algorithm),
            "interest_rate": InterestRatePulseIntegrator(algorithm),
            "federal_docket": FederalDocketForecastAI(algorithm),
            "treasury_auction": TreasuryAuctionImbalanceScanner(algorithm),
            "overnight_reserve": OvernightReserveDriftTracker(algorithm)
        }
        
        self.technical_sources = {
            "trend": 0.0,
            "momentum": 0.0,
            "volatility": 0.0,
            "volume": 0.0,
            "sentiment": 0.0
        }
        
        self.conviction_weights = {
            "fed": 0.6,
            "technical": 0.4
        }
        
        self.fed_weights = {
            "liquidity_arbitrage": 0.2,
            "repo_market": 0.15,
            "interest_rate": 0.25,
            "federal_docket": 0.15,
            "treasury_auction": 0.15,
            "overnight_reserve": 0.1
        }
        
        self.technical_weights = {
            "trend": 0.3,
            "momentum": 0.3,
            "volatility": 0.2,
            "volume": 0.1,
            "sentiment": 0.1
        }
        
        self.conviction_scores = {
            "fed": 0.0,
            "technical": 0.0,
            "combined": 0.0
        }
        
        self.allocation_data = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
    def update(self, current_time, technical_data=None):
        """
        Update the conviction-fused allocation with latest data.
        
        Parameters:
        - current_time: Current datetime
        - technical_data: Technical signal data (optional)
        
        Returns:
        - Dictionary containing allocation results
        """
        if current_time - self.last_update < self.update_frequency and technical_data is None:
            return {
                "conviction_scores": self.conviction_scores,
                "allocation_data": self.allocation_data
            }
            
        self._update_fed_modules(current_time)
        
        if technical_data is not None:
            self._update_technical_data(technical_data)
        else:
            self._update_technical_data_internal()
        
        self._calculate_conviction_scores()
        
        self._calculate_allocations()
        
        self.last_update = current_time
        
        return {
            "conviction_scores": self.conviction_scores,
            "allocation_data": self.allocation_data
        }
        
    def _update_fed_modules(self, current_time):
        """
        Update Fed module data.
        
        Parameters:
        - current_time: Current datetime
        """
        for module_name, module in self.fed_modules.items():
            module.update(current_time)
        
    def _update_technical_data(self, technical_data):
        """
        Update technical signal data.
        
        Parameters:
        - technical_data: Technical signal data
        """
        for source, value in technical_data.items():
            if source in self.technical_sources:
                self.technical_sources[source] = value
        
    def _update_technical_data_internal(self):
        """
        Update technical signal data internally.
        """
        
        self.technical_sources = {
            "trend": 0.7,       # Bullish trend
            "momentum": 0.65,   # Positive momentum
            "volatility": 0.4,  # Low volatility
            "volume": 0.6,      # Above average volume
            "sentiment": 0.55   # Slightly positive sentiment
        }
        
    def _calculate_conviction_scores(self):
        """
        Calculate conviction scores.
        """
        fed_score = 0.0
        for module_name, weight in self.fed_weights.items():
            module = self.fed_modules.get(module_name)
            if module is not None:
                module_result = module.update(datetime.now())
                
                if "combined_score" in module_result:
                    module_score = module_result["combined_score"]
                elif "imbalance_score" in module_result and "distortion_score" in module_result:
                    module_score = (module_result["imbalance_score"] + module_result["distortion_score"]) / 2
                elif "volatility_score" in module_result and "market_impact_score" in module_result:
                    module_score = (module_result["volatility_score"] + module_result["market_impact_score"]) / 2
                elif "inversion_score" in module_result and "pulse_score" in module_result:
                    module_score = (module_result["inversion_score"] + module_result["pulse_score"]) / 2
                elif "drift_score" in module_result and "leakage_score" in module_result:
                    module_score = (module_result["drift_score"] + module_result["leakage_score"]) / 2
                else:
                    module_score = 0.5
                
                fed_score += module_score * weight
        
        technical_score = 0.0
        for source, weight in self.technical_weights.items():
            source_value = self.technical_sources.get(source, 0.0)
            technical_score += source_value * weight
        
        self.conviction_scores["fed"] = fed_score
        self.conviction_scores["technical"] = technical_score
        
        self.conviction_scores["combined"] = (
            self.conviction_scores["fed"] * self.conviction_weights["fed"] +
            self.conviction_scores["technical"] * self.conviction_weights["technical"]
        )
        
    def _calculate_allocations(self):
        """
        Calculate allocations based on conviction scores.
        """
        conviction = self.conviction_scores["combined"]
        
        if conviction > 0.8:
            base_allocation = 1.0  # Full allocation
        elif conviction > 0.7:
            base_allocation = 0.8  # 80% allocation
        elif conviction > 0.6:
            base_allocation = 0.6  # 60% allocation
        elif conviction > 0.5:
            base_allocation = 0.4  # 40% allocation
        elif conviction > 0.4:
            base_allocation = 0.2  # 20% allocation
        else:
            base_allocation = 0.0  # No allocation
        
        if conviction > 0.5:
            direction = "LONG"
        elif conviction < 0.5:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        if direction == "NEUTRAL":
            confidence = 0.0
        else:
            confidence = abs(conviction - 0.5) * 2.0
        
        self.allocation_data = {
            "base_allocation": base_allocation,
            "direction": direction,
            "confidence": confidence,
            "conviction": conviction
        }
        
    def get_conviction_scores(self):
        """
        Get conviction scores.
        
        Returns:
        - Conviction scores
        """
        return self.conviction_scores
        
    def get_allocation_data(self):
        """
        Get allocation data.
        
        Returns:
        - Allocation data
        """
        return self.allocation_data
        
    def calculate_position_size(self, base_size, asset_conviction=None):
        """
        Calculate position size based on conviction.
        
        Parameters:
        - base_size: Base position size
        - asset_conviction: Asset-specific conviction (optional)
        
        Returns:
        - Adjusted position size
        """
        allocation = self.allocation_data.get("base_allocation", 0.0)
        direction = self.allocation_data.get("direction", "NEUTRAL")
        
        if direction == "NEUTRAL":
            return 0.0
        
        if asset_conviction is not None:
            conviction_blend = 0.7 * allocation + 0.3 * asset_conviction
            adjusted_size = base_size * conviction_blend
        else:
            adjusted_size = base_size * allocation
        
        if direction == "SHORT":
            adjusted_size = -adjusted_size
        
        return adjusted_size
        
    def adjust_conviction_weights(self, new_weights):
        """
        Adjust conviction weights.
        
        Parameters:
        - new_weights: Dictionary containing new weights
        
        Returns:
        - Boolean indicating success
        """
        if not isinstance(new_weights, dict):
            return False
            
        if "fed" in new_weights and "technical" in new_weights:
            total_weight = new_weights["fed"] + new_weights["technical"]
            if abs(total_weight - 1.0) > 0.001:  # Allow small rounding errors
                new_weights["fed"] /= total_weight
                new_weights["technical"] /= total_weight
            
            self.conviction_weights = new_weights
            return True
        
        return False
