"""
Margin-Weighted Alpha Tiers

Allocates more capital to high-Sharpe plays for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class MarginWeightedAlphaTiers:
    """
    Allocates more capital to high-Sharpe plays.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Margin-Weighted Alpha Tiers.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("MarginWeightedAlphaTiers")
        self.logger.setLevel(logging.INFO)
        
        self.alpha_tiers = {
            "S": {  # Superior
                "min_sharpe": 3.0,
                "min_win_rate": 0.7,
                "min_profit_factor": 3.0,
                "allocation_multiplier": 2.0,
                "margin_requirement_discount": 0.2  # 20% discount on margin requirements
            },
            "A": {  # Excellent
                "min_sharpe": 2.5,
                "min_win_rate": 0.65,
                "min_profit_factor": 2.5,
                "allocation_multiplier": 1.5,
                "margin_requirement_discount": 0.1  # 10% discount on margin requirements
            },
            "B": {  # Good
                "min_sharpe": 2.0,
                "min_win_rate": 0.6,
                "min_profit_factor": 2.0,
                "allocation_multiplier": 1.0,
                "margin_requirement_discount": 0.0  # No discount on margin requirements
            },
            "C": {  # Average
                "min_sharpe": 1.5,
                "min_win_rate": 0.55,
                "min_profit_factor": 1.5,
                "allocation_multiplier": 0.75,
                "margin_requirement_discount": 0.0  # No discount on margin requirements
            },
            "D": {  # Below Average
                "min_sharpe": 1.0,
                "min_win_rate": 0.5,
                "min_profit_factor": 1.0,
                "allocation_multiplier": 0.5,
                "margin_requirement_discount": 0.0  # No discount on margin requirements
            },
            "F": {  # Poor
                "min_sharpe": 0.0,
                "min_win_rate": 0.0,
                "min_profit_factor": 0.0,
                "allocation_multiplier": 0.25,
                "margin_requirement_discount": 0.0  # No discount on margin requirements
            }
        }
        
        self.strategy_performance = {}
        
        self.strategy_tiers = {}
        
        self.allocation_data = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(days=1)
        
    def update(self, current_time):
        """
        Update the alpha tiers with latest performance data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing tier results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "strategy_tiers": self.strategy_tiers,
                "allocation_data": self.allocation_data
            }
            
        self._update_strategy_performance()
        
        self._assign_strategy_tiers()
        
        self._calculate_allocations()
        
        self.last_update = current_time
        
        return {
            "strategy_tiers": self.strategy_tiers,
            "allocation_data": self.allocation_data
        }
        
    def _update_strategy_performance(self):
        """
        Update strategy performance data.
        """
        
        self.strategy_performance = {
            "momentum_strategy": {
                "sharpe_ratio": 2.8,
                "win_rate": 0.68,
                "profit_factor": 2.7,
                "max_drawdown": 0.15,
                "avg_trade": 0.02,
                "trades_per_day": 3.5,
                "margin_requirement": 0.5  # 50% margin requirement
            },
            "mean_reversion_strategy": {
                "sharpe_ratio": 2.2,
                "win_rate": 0.62,
                "profit_factor": 2.3,
                "max_drawdown": 0.18,
                "avg_trade": 0.015,
                "trades_per_day": 5.0,
                "margin_requirement": 0.4  # 40% margin requirement
            },
            "breakout_strategy": {
                "sharpe_ratio": 1.8,
                "win_rate": 0.58,
                "profit_factor": 1.9,
                "max_drawdown": 0.22,
                "avg_trade": 0.025,
                "trades_per_day": 2.0,
                "margin_requirement": 0.6  # 60% margin requirement
            },
            "trend_following_strategy": {
                "sharpe_ratio": 1.5,
                "win_rate": 0.55,
                "profit_factor": 1.6,
                "max_drawdown": 0.25,
                "avg_trade": 0.03,
                "trades_per_day": 1.5,
                "margin_requirement": 0.5  # 50% margin requirement
            },
            "arbitrage_strategy": {
                "sharpe_ratio": 3.2,
                "win_rate": 0.75,
                "profit_factor": 3.5,
                "max_drawdown": 0.1,
                "avg_trade": 0.01,
                "trades_per_day": 10.0,
                "margin_requirement": 0.7  # 70% margin requirement
            }
        }
        
    def _assign_strategy_tiers(self):
        """
        Assign strategies to alpha tiers.
        """
        self.strategy_tiers = {}
        
        for strategy_name, performance in self.strategy_performance.items():
            sharpe_ratio = performance.get("sharpe_ratio", 0.0)
            win_rate = performance.get("win_rate", 0.0)
            profit_factor = performance.get("profit_factor", 0.0)
            
            assigned_tier = "F"  # Default tier
            
            for tier, criteria in self.alpha_tiers.items():
                if (sharpe_ratio >= criteria["min_sharpe"] and
                    win_rate >= criteria["min_win_rate"] and
                    profit_factor >= criteria["min_profit_factor"]):
                    assigned_tier = tier
                    break
            
            self.strategy_tiers[strategy_name] = {
                "tier": assigned_tier,
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "allocation_multiplier": self.alpha_tiers[assigned_tier]["allocation_multiplier"],
                "margin_requirement_discount": self.alpha_tiers[assigned_tier]["margin_requirement_discount"]
            }
        
    def _calculate_allocations(self):
        """
        Calculate allocations based on alpha tiers.
        """
        self.allocation_data = {}
        
        total_points = 0.0
        for strategy_name, tier_data in self.strategy_tiers.items():
            total_points += tier_data["allocation_multiplier"]
        
        for strategy_name, tier_data in self.strategy_tiers.items():
            performance = self.strategy_performance.get(strategy_name, {})
            
            base_allocation = tier_data["allocation_multiplier"] / total_points if total_points > 0 else 0.0
            
            margin_requirement = performance.get("margin_requirement", 0.5)
            effective_margin = margin_requirement * (1.0 - tier_data["margin_requirement_discount"])
            
            margin_adjusted_allocation = base_allocation / effective_margin if effective_margin > 0 else 0.0
            
            self.allocation_data[strategy_name] = {
                "tier": tier_data["tier"],
                "base_allocation": base_allocation,
                "margin_requirement": margin_requirement,
                "effective_margin": effective_margin,
                "margin_adjusted_allocation": margin_adjusted_allocation
            }
        
        total_margin_adjusted = sum(data["margin_adjusted_allocation"] for data in self.allocation_data.values())
        
        if total_margin_adjusted > 0:
            for strategy_name in self.allocation_data:
                self.allocation_data[strategy_name]["normalized_allocation"] = (
                    self.allocation_data[strategy_name]["margin_adjusted_allocation"] / total_margin_adjusted
                )
        else:
            equal_allocation = 1.0 / len(self.allocation_data) if len(self.allocation_data) > 0 else 0.0
            for strategy_name in self.allocation_data:
                self.allocation_data[strategy_name]["normalized_allocation"] = equal_allocation
        
    def get_strategy_tier(self, strategy_name):
        """
        Get tier assignment for a strategy.
        
        Parameters:
        - strategy_name: Strategy name
        
        Returns:
        - Tier assignment data
        """
        return self.strategy_tiers.get(strategy_name, None)
        
    def get_strategy_allocation(self, strategy_name):
        """
        Get allocation for a strategy.
        
        Parameters:
        - strategy_name: Strategy name
        
        Returns:
        - Allocation data
        """
        return self.allocation_data.get(strategy_name, None)
        
    def get_all_allocations(self):
        """
        Get all strategy allocations.
        
        Returns:
        - Dictionary of all allocations
        """
        return self.allocation_data
        
    def calculate_position_size(self, strategy_name, base_size, account_size=None):
        """
        Calculate position size for a strategy.
        
        Parameters:
        - strategy_name: Strategy name
        - base_size: Base position size
        - account_size: Account size (optional)
        
        Returns:
        - Adjusted position size
        """
        if strategy_name not in self.allocation_data:
            return base_size
            
        allocation = self.allocation_data[strategy_name]
        
        if account_size is not None:
            adjusted_size = account_size * allocation["normalized_allocation"]
        else:
            tier = self.strategy_tiers.get(strategy_name, {})
            multiplier = tier.get("allocation_multiplier", 1.0)
            adjusted_size = base_size * multiplier
        
        return adjusted_size
