"""
VIX Shock Dampener

Cuts risk 50% if VIX > 30 and inversion occurs for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class VIXShockDampener:
    """
    Cuts risk 50% if VIX > 30 and inversion occurs.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the VIX Shock Dampener.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("VIXShockDampener")
        self.logger.setLevel(logging.INFO)
        
        self.vix_threshold = 30.0
        self.vix_extreme_threshold = 40.0
        
        self.inversion_threshold = -0.1  # 10bps inversion
        self.deep_inversion_threshold = -0.25  # 25bps inversion
        
        self.risk_reduction_levels = {
            "normal": 1.0,       # No reduction
            "elevated": 0.75,    # 25% reduction
            "high": 0.5,         # 50% reduction
            "extreme": 0.25,     # 75% reduction
            "crisis": 0.1        # 90% reduction
        }
        
        self.current_risk_level = "normal"
        self.current_risk_reduction = self.risk_reduction_levels[self.current_risk_level]
        
        self.market_data = {
            "vix": 0.0,
            "vix_5d_avg": 0.0,
            "vix_20d_avg": 0.0,
            "vix_percentile": 0.0,
            "yield_10y2y": 0.0,
            "yield_10y2y_5d_avg": 0.0,
            "yield_10y2y_20d_avg": 0.0
        }
        
        self.risk_level_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
    def update(self, current_time, market_data=None):
        """
        Update the VIX shock dampener with latest data.
        
        Parameters:
        - current_time: Current datetime
        - market_data: Market data (optional)
        
        Returns:
        - Dictionary containing dampener results
        """
        if current_time - self.last_update < self.update_frequency and market_data is None:
            return {
                "current_risk_level": self.current_risk_level,
                "current_risk_reduction": self.current_risk_reduction,
                "market_data": self.market_data
            }
            
        if market_data is not None:
            self._update_market_data(market_data)
        else:
            self._update_market_data_internal()
        
        self._determine_risk_level()
        
        self.last_update = current_time
        
        return {
            "current_risk_level": self.current_risk_level,
            "current_risk_reduction": self.current_risk_reduction,
            "market_data": self.market_data
        }
        
    def _update_market_data(self, market_data):
        """
        Update market data.
        
        Parameters:
        - market_data: Market data
        """
        for key, value in market_data.items():
            if key in self.market_data:
                self.market_data[key] = value
        
    def _update_market_data_internal(self):
        """
        Update market data internally.
        """
        
        self.market_data = {
            "vix": 22.5,
            "vix_5d_avg": 21.8,
            "vix_20d_avg": 19.5,
            "vix_percentile": 0.65,
            "yield_10y2y": 0.15,
            "yield_10y2y_5d_avg": 0.18,
            "yield_10y2y_20d_avg": 0.22
        }
        
    def _determine_risk_level(self):
        """
        Determine the appropriate risk level.
        """
        vix = self.market_data["vix"]
        vix_5d_avg = self.market_data["vix_5d_avg"]
        vix_percentile = self.market_data["vix_percentile"]
        yield_10y2y = self.market_data["yield_10y2y"]
        yield_10y2y_5d_avg = self.market_data["yield_10y2y_5d_avg"]
        
        if vix > self.vix_extreme_threshold and yield_10y2y < self.deep_inversion_threshold:
            new_risk_level = "crisis"
        elif vix > self.vix_threshold and yield_10y2y < self.inversion_threshold:
            new_risk_level = "extreme"
        elif vix > self.vix_threshold or yield_10y2y < self.inversion_threshold:
            new_risk_level = "high"
        elif vix > 25.0 or yield_10y2y < 0.0:
            new_risk_level = "elevated"
        else:
            new_risk_level = "normal"
        
        if vix > vix_5d_avg * 1.5:
            risk_levels = list(self.risk_reduction_levels.keys())
            current_index = risk_levels.index(new_risk_level)
            if current_index < len(risk_levels) - 1:
                new_risk_level = risk_levels[current_index + 1]
        
        if yield_10y2y < yield_10y2y_5d_avg - 0.2:
            risk_levels = list(self.risk_reduction_levels.keys())
            current_index = risk_levels.index(new_risk_level)
            if current_index < len(risk_levels) - 1:
                new_risk_level = risk_levels[current_index + 1]
        
        if new_risk_level != self.current_risk_level:
            self.risk_level_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "from_level": self.current_risk_level,
                "to_level": new_risk_level,
                "vix": vix,
                "yield_10y2y": yield_10y2y
            })
            
            self.current_risk_level = new_risk_level
            self.current_risk_reduction = self.risk_reduction_levels[self.current_risk_level]
            
            self.logger.info(f"Risk level changed from {self.risk_level_history[-1]['from_level']} to {self.current_risk_level}")
            self.logger.info(f"Risk reduction: {self.current_risk_reduction}")
        
    def get_current_risk_reduction(self):
        """
        Get current risk reduction.
        
        Returns:
        - Current risk reduction
        """
        return self.current_risk_reduction
        
    def get_current_risk_level(self):
        """
        Get current risk level.
        
        Returns:
        - Current risk level
        """
        return self.current_risk_level
        
    def get_risk_level_history(self):
        """
        Get risk level history.
        
        Returns:
        - Risk level history
        """
        return self.risk_level_history
        
    def calculate_position_size(self, base_size, asset_beta=None):
        """
        Calculate position size based on current risk level.
        
        Parameters:
        - base_size: Base position size
        - asset_beta: Asset beta to market (optional)
        
        Returns:
        - Adjusted position size
        """
        reduction = self.current_risk_reduction
        
        if asset_beta is not None:
            beta_adjustment = 1.0 / max(1.0, asset_beta)
            reduction = min(1.0, reduction * beta_adjustment)
        
        adjusted_size = base_size * reduction
        
        return adjusted_size
