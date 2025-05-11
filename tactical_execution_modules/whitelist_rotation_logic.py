"""
Whitelist Rotation Logic

Rotates sectors based on macro regime shifts for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class WhitelistRotationLogic:
    """
    Rotates sectors and assets based on macro regime shifts.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Whitelist Rotation Logic.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("WhitelistRotationLogic")
        self.logger.setLevel(logging.INFO)
        
        self.macro_regimes = {
            "EXPANSION": {
                "description": "Economic growth with low inflation",
                "sectors": ["Technology", "Consumer Discretionary", "Industrials"],
                "assets": ["QQQ", "XLY", "XLI", "BTCUSD", "ETHUSD"]
            },
            "INFLATION": {
                "description": "Economic growth with high inflation",
                "sectors": ["Energy", "Materials", "Real Estate"],
                "assets": ["XLE", "XLB", "IYR", "XAUUSD"]
            },
            "SLOWDOWN": {
                "description": "Economic slowdown with high inflation",
                "sectors": ["Utilities", "Consumer Staples", "Healthcare"],
                "assets": ["XLU", "XLP", "XLV", "XAUUSD"]
            },
            "CONTRACTION": {
                "description": "Economic contraction with low inflation",
                "sectors": ["Utilities", "Consumer Staples", "Healthcare"],
                "assets": ["XLU", "XLP", "XLV", "TLT", "XAUUSD"]
            },
            "RECOVERY": {
                "description": "Economic recovery from contraction",
                "sectors": ["Financials", "Industrials", "Materials"],
                "assets": ["XLF", "XLI", "XLB", "BTCUSD"]
            }
        }
        
        self.regime_indicators = {
            "gdp_growth": 0.0,
            "inflation": 0.0,
            "unemployment": 0.0,
            "yield_curve": 0.0,
            "vix": 0.0
        }
        
        self.current_regime = "EXPANSION"
        self.regime_probability = {regime: 0.0 for regime in self.macro_regimes}
        self.regime_probability[self.current_regime] = 1.0
        
        self.current_whitelist = self.macro_regimes[self.current_regime]["assets"]
        
        self.rotation_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(days=1)
        
    def update(self, current_time):
        """
        Update the whitelist rotation logic with latest data.
        
        Parameters:
        - current_time: Current datetime
        
        Returns:
        - Dictionary containing rotation results
        """
        if current_time - self.last_update < self.update_frequency:
            return {
                "current_regime": self.current_regime,
                "regime_probability": self.regime_probability,
                "current_whitelist": self.current_whitelist,
                "rotation_history": self.rotation_history
            }
            
        self._update_regime_indicators()
        
        self._calculate_regime_probabilities()
        
        self._determine_current_regime()
        
        self._update_whitelist()
        
        self.last_update = current_time
        
        return {
            "current_regime": self.current_regime,
            "regime_probability": self.regime_probability,
            "current_whitelist": self.current_whitelist,
            "rotation_history": self.rotation_history
        }
        
    def _update_regime_indicators(self):
        """
        Update macro regime indicators.
        """
        
        self.regime_indicators = {
            "gdp_growth": 2.1,       # GDP growth rate (%)
            "inflation": 4.2,         # Inflation rate (%)
            "unemployment": 3.6,      # Unemployment rate (%)
            "yield_curve": -0.2,      # 10Y-2Y Treasury yield spread (%)
            "vix": 18.5               # VIX index
        }
        
    def _calculate_regime_probabilities(self):
        """
        Calculate probabilities for each macro regime.
        """
        self.regime_probability = {regime: 0.0 for regime in self.macro_regimes}
        
        gdp_growth = self.regime_indicators["gdp_growth"]
        inflation = self.regime_indicators["inflation"]
        unemployment = self.regime_indicators["unemployment"]
        yield_curve = self.regime_indicators["yield_curve"]
        vix = self.regime_indicators["vix"]
        
        expansion_prob = (
            self._score_indicator(gdp_growth, 2.5, 4.0) * 0.3 +
            self._score_indicator(inflation, 1.0, 2.5, inverse=True) * 0.2 +
            self._score_indicator(unemployment, 3.0, 5.0, inverse=True) * 0.2 +
            self._score_indicator(yield_curve, 0.5, 2.0) * 0.2 +
            self._score_indicator(vix, 10.0, 20.0, inverse=True) * 0.1
        )
        self.regime_probability["EXPANSION"] = expansion_prob
        
        inflation_prob = (
            self._score_indicator(gdp_growth, 1.5, 3.0) * 0.2 +
            self._score_indicator(inflation, 3.0, 6.0) * 0.4 +
            self._score_indicator(unemployment, 3.0, 5.0, inverse=True) * 0.1 +
            self._score_indicator(yield_curve, -0.5, 0.5) * 0.2 +
            self._score_indicator(vix, 15.0, 25.0) * 0.1
        )
        self.regime_probability["INFLATION"] = inflation_prob
        
        slowdown_prob = (
            self._score_indicator(gdp_growth, 0.0, 1.5, inverse=True) * 0.3 +
            self._score_indicator(inflation, 3.0, 6.0) * 0.2 +
            self._score_indicator(unemployment, 4.0, 6.0) * 0.2 +
            self._score_indicator(yield_curve, -1.0, 0.0, inverse=True) * 0.2 +
            self._score_indicator(vix, 20.0, 30.0) * 0.1
        )
        self.regime_probability["SLOWDOWN"] = slowdown_prob
        
        contraction_prob = (
            self._score_indicator(gdp_growth, -2.0, 0.0, inverse=True) * 0.3 +
            self._score_indicator(inflation, 0.0, 2.0) * 0.1 +
            self._score_indicator(unemployment, 6.0, 10.0) * 0.2 +
            self._score_indicator(yield_curve, -2.0, -0.5, inverse=True) * 0.2 +
            self._score_indicator(vix, 30.0, 50.0) * 0.2
        )
        self.regime_probability["CONTRACTION"] = contraction_prob
        
        recovery_prob = (
            self._score_indicator(gdp_growth, 0.0, 2.0) * 0.3 +
            self._score_indicator(inflation, 0.0, 2.0) * 0.1 +
            self._score_indicator(unemployment, 5.0, 8.0) * 0.2 +
            self._score_indicator(yield_curve, 0.0, 1.0) * 0.2 +
            self._score_indicator(vix, 20.0, 35.0) * 0.2
        )
        self.regime_probability["RECOVERY"] = recovery_prob
        
        total_prob = sum(self.regime_probability.values())
        if total_prob > 0:
            for regime in self.regime_probability:
                self.regime_probability[regime] /= total_prob
        
    def _score_indicator(self, value, min_val, max_val, inverse=False):
        """
        Score an indicator value between 0.0 and 1.0.
        
        Parameters:
        - value: Indicator value
        - min_val: Minimum value for full score
        - max_val: Maximum value for full score
        - inverse: Whether to invert the score
        
        Returns:
        - Score between 0.0 and 1.0
        """
        if min_val <= value <= max_val:
            score = 1.0
        elif value < min_val:
            score = max(0.0, 1.0 - (min_val - value) / min_val)
        else:  # value > max_val
            score = max(0.0, 1.0 - (value - max_val) / max_val)
            
        if inverse:
            score = 1.0 - score
            
        return score
        
    def _determine_current_regime(self):
        """
        Determine the current macro regime.
        """
        max_prob = 0.0
        max_regime = self.current_regime
        
        for regime, prob in self.regime_probability.items():
            if prob > max_prob:
                max_prob = prob
                max_regime = regime
        
        if max_regime != self.current_regime:
            self.rotation_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "from_regime": self.current_regime,
                "to_regime": max_regime,
                "probability": max_prob
            })
            
            self.current_regime = max_regime
        
    def _update_whitelist(self):
        """
        Update the whitelist based on current regime.
        """
        regime_assets = self.macro_regimes[self.current_regime]["assets"]
        
        if set(regime_assets) != set(self.current_whitelist):
            self.current_whitelist = regime_assets
        
    def get_current_whitelist(self):
        """
        Get the current whitelist of assets.
        
        Returns:
        - List of assets in the current whitelist
        """
        return self.current_whitelist
        
    def get_current_regime(self):
        """
        Get the current macro regime.
        
        Returns:
        - Current regime name
        """
        return self.current_regime
        
    def get_regime_description(self, regime=None):
        """
        Get the description of a regime.
        
        Parameters:
        - regime: Regime name (default: current regime)
        
        Returns:
        - Regime description
        """
        if regime is None:
            regime = self.current_regime
            
        if regime in self.macro_regimes:
            return self.macro_regimes[regime]["description"]
        else:
            return "Unknown regime"
        
    def get_regime_sectors(self, regime=None):
        """
        Get the sectors for a regime.
        
        Parameters:
        - regime: Regime name (default: current regime)
        
        Returns:
        - List of sectors
        """
        if regime is None:
            regime = self.current_regime
            
        if regime in self.macro_regimes:
            return self.macro_regimes[regime]["sectors"]
        else:
            return []
