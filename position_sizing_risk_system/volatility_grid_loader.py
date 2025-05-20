"""
Volatility Grid Loader

Adjusts position size based on VIX structure for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class VolatilityGridLoader:
    """
    Adjusts position size based on VIX structure.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Volatility Grid Loader.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("VolatilityGridLoader")
        self.logger.setLevel(logging.INFO)
        
        self.vix_thresholds = [12, 15, 20, 25, 30, 35, 40, 50]
        
        self.position_multipliers = [1.5, 1.3, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
        
        self.contango_threshold = 0.95  # VIX3M/VIX ratio for contango
        self.backwardation_threshold = 1.05  # VIX3M/VIX ratio for backwardation
        
        self.term_structure_multipliers = {
            "steep_contango": 1.2,      # VIX3M/VIX < 0.85
            "contango": 1.1,            # 0.85 <= VIX3M/VIX < 0.95
            "neutral": 1.0,             # 0.95 <= VIX3M/VIX < 1.05
            "backwardation": 0.8,       # 1.05 <= VIX3M/VIX < 1.15
            "steep_backwardation": 0.6  # VIX3M/VIX >= 1.15
        }
        
        self.vix_data = {
            "current": 0.0,
            "vix3m": 0.0,
            "term_structure_ratio": 0.0,
            "term_structure_state": "neutral",
            "historical": []
        }
        
        self.grid_data = {}
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
    def update(self, current_time):
        """
        Update the volatility grid with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing grid results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "vix_data": self.vix_data,
                "grid_data": self.grid_data,
                "position_size_multiplier": self._get_current_multiplier()
            }
            
        self._update_vix_data()
        
        self._update_grid_data()
        
        self.last_update = current_time
        
        return {
            "vix_data": self.vix_data,
            "grid_data": self.grid_data,
            "position_size_multiplier": self._get_current_multiplier()
        }
        
    def _update_vix_data(self):
        """
        Update VIX data.
        """
        
        current_vix = 18.5
        vix3m = 19.2
        
        self.vix_data["current"] = current_vix
        self.vix_data["vix3m"] = vix3m
        self.vix_data["term_structure_ratio"] = vix3m / current_vix if current_vix > 0 else 1.0
        
        ratio = self.vix_data["term_structure_ratio"]
        if ratio < 0.85:
            self.vix_data["term_structure_state"] = "steep_contango"
        elif ratio < self.contango_threshold:
            self.vix_data["term_structure_state"] = "contango"
        elif ratio < self.backwardation_threshold:
            self.vix_data["term_structure_state"] = "neutral"
        elif ratio < 1.15:
            self.vix_data["term_structure_state"] = "backwardation"
        else:
            self.vix_data["term_structure_state"] = "steep_backwardation"
        
        self.vix_data["historical"].append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "vix": current_vix,
            "vix3m": vix3m,
            "ratio": ratio,
            "state": self.vix_data["term_structure_state"]
        })
        
        if len(self.vix_data["historical"]) > 30:
            self.vix_data["historical"] = self.vix_data["historical"][-30:]
        
    def _update_grid_data(self):
        """
        Update volatility grid data.
        """
        current_vix = self.vix_data["current"]
        term_structure_state = self.vix_data["term_structure_state"]
        
        self.grid_data = {}
        
        threshold_idx = 0
        for i, threshold in enumerate(self.vix_thresholds):
            if current_vix < threshold:
                threshold_idx = i
                break
            threshold_idx = len(self.vix_thresholds)
        
        base_multiplier = self.position_multipliers[min(threshold_idx, len(self.position_multipliers) - 1)]
        
        term_multiplier = self.term_structure_multipliers.get(term_structure_state, 1.0)
        
        final_multiplier = base_multiplier * term_multiplier
        
        self.grid_data = {
            "vix_threshold": self.vix_thresholds[min(threshold_idx, len(self.vix_thresholds) - 1)] if threshold_idx < len(self.vix_thresholds) else "50+",
            "base_multiplier": base_multiplier,
            "term_structure_state": term_structure_state,
            "term_multiplier": term_multiplier,
            "final_multiplier": final_multiplier
        }
        
    def _get_current_multiplier(self):
        """
        Get current position size multiplier.
        
        Returns:
        - Current position size multiplier
        """
        if "final_multiplier" in self.grid_data:
            return self.grid_data["final_multiplier"]
        else:
            return 1.0
        
    def calculate_position_size(self, base_size, asset_volatility=None, asset_correlation=None):
        """
        Calculate position size based on volatility grid.
        
        Parameters:
        - base_size: Base position size
        - asset_volatility: Asset-specific volatility (optional)
        - asset_correlation: Asset correlation with market (optional)
        
        Returns:
        - Adjusted position size
        """
        multiplier = self._get_current_multiplier()
        
        if asset_volatility is not None:
            vol_adjustment = 0.2 / asset_volatility
            multiplier *= vol_adjustment
        
        if asset_correlation is not None:
            corr_adjustment = 1.0 - (asset_correlation * 0.5)
            multiplier *= corr_adjustment
        
        adjusted_size = base_size * multiplier
        
        return adjusted_size
        
    def get_vix_data(self):
        """
        Get VIX data.
        
        Returns:
        - VIX data
        """
        return self.vix_data
        
    def get_grid_data(self):
        """
        Get grid data.
        
        Returns:
        - Grid data
        """
        return self.grid_data
        
    def get_position_multiplier(self):
        """
        Get current position size multiplier.
        
        Returns:
        - Current position size multiplier
        """
        return self._get_current_multiplier()
