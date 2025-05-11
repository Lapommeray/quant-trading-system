"""
Dynamic Leverage Bracket Switcher

Switches from 1.5x to 10x based on conviction for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class DynamicLeverageBracketSwitcher:
    """
    Switches from 1.5x to 10x based on conviction.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Dynamic Leverage Bracket Switcher.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("DynamicLeverageBracketSwitcher")
        self.logger.setLevel(logging.INFO)
        
        self.leverage_brackets = {
            "ultra_conservative": 1.5,
            "conservative": 2.0,
            "moderate": 3.0,
            "aggressive": 5.0,
            "ultra_aggressive": 10.0
        }
        
        self.conviction_thresholds = {
            "ultra_conservative": 0.0,
            "conservative": 0.6,
            "moderate": 0.75,
            "aggressive": 0.85,
            "ultra_aggressive": 0.95
        }
        
        self.risk_thresholds = {
            "vix": {
                "ultra_conservative": 40.0,
                "conservative": 30.0,
                "moderate": 20.0,
                "aggressive": 15.0,
                "ultra_aggressive": 12.0
            },
            "drawdown": {
                "ultra_conservative": 0.05,
                "conservative": 0.10,
                "moderate": 0.15,
                "aggressive": 0.20,
                "ultra_aggressive": 0.25
            }
        }
        
        self.current_bracket = "moderate"
        self.current_leverage = self.leverage_brackets[self.current_bracket]
        
        self.conviction_data = {
            "signal_strength": 0.0,
            "signal_consistency": 0.0,
            "signal_confirmation": 0.0,
            "overall_conviction": 0.0
        }
        
        self.risk_data = {
            "vix": 0.0,
            "drawdown": 0.0,
            "market_stress": 0.0
        }
        
        self.bracket_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
    def update(self, current_time, conviction_data=None, risk_data=None):
        """
        Update the leverage bracket with latest data.
        
        Parameters:
        - current_time: Current datetime
        - conviction_data: Conviction data (optional)
        - risk_data: Risk data (optional)
        
        Returns:
        - Dictionary containing bracket results
        """
        if current_time - self.last_update < self.update_frequency and conviction_data is None and risk_data is None:
            return {
                "current_bracket": self.current_bracket,
                "current_leverage": self.current_leverage,
                "conviction_data": self.conviction_data,
                "risk_data": self.risk_data
            }
            
        if conviction_data is not None:
            self._update_conviction_data(conviction_data)
        else:
            self._update_conviction_data_internal()
        
        if risk_data is not None:
            self._update_risk_data(risk_data)
        else:
            self._update_risk_data_internal()
        
        self._determine_bracket()
        
        self.last_update = current_time
        
        return {
            "current_bracket": self.current_bracket,
            "current_leverage": self.current_leverage,
            "conviction_data": self.conviction_data,
            "risk_data": self.risk_data
        }
        
    def _update_conviction_data(self, conviction_data):
        """
        Update conviction data.
        
        Parameters:
        - conviction_data: Conviction data
        """
        self.conviction_data["signal_strength"] = conviction_data.get("signal_strength", self.conviction_data["signal_strength"])
        self.conviction_data["signal_consistency"] = conviction_data.get("signal_consistency", self.conviction_data["signal_consistency"])
        self.conviction_data["signal_confirmation"] = conviction_data.get("signal_confirmation", self.conviction_data["signal_confirmation"])
        
        self._calculate_overall_conviction()
        
    def _update_conviction_data_internal(self):
        """
        Update conviction data internally.
        """
        
        self.conviction_data["signal_strength"] = 0.85
        self.conviction_data["signal_consistency"] = 0.80
        self.conviction_data["signal_confirmation"] = 0.90
        
        self._calculate_overall_conviction()
        
    def _calculate_overall_conviction(self):
        """
        Calculate overall conviction.
        """
        self.conviction_data["overall_conviction"] = (
            self.conviction_data["signal_strength"] * 0.4 +
            self.conviction_data["signal_consistency"] * 0.3 +
            self.conviction_data["signal_confirmation"] * 0.3
        )
        
    def _update_risk_data(self, risk_data):
        """
        Update risk data.
        
        Parameters:
        - risk_data: Risk data
        """
        self.risk_data["vix"] = risk_data.get("vix", self.risk_data["vix"])
        self.risk_data["drawdown"] = risk_data.get("drawdown", self.risk_data["drawdown"])
        self.risk_data["market_stress"] = risk_data.get("market_stress", self.risk_data["market_stress"])
        
    def _update_risk_data_internal(self):
        """
        Update risk data internally.
        """
        
        self.risk_data["vix"] = 18.5
        self.risk_data["drawdown"] = 0.08
        self.risk_data["market_stress"] = 0.3
        
    def _determine_bracket(self):
        """
        Determine the appropriate leverage bracket.
        """
        conviction = self.conviction_data["overall_conviction"]
        vix = self.risk_data["vix"]
        drawdown = self.risk_data["drawdown"]
        
        conviction_bracket = "ultra_conservative"
        for bracket, threshold in self.conviction_thresholds.items():
            if conviction >= threshold:
                conviction_bracket = bracket
        
        vix_bracket = "ultra_aggressive"
        for bracket, threshold in self.risk_thresholds["vix"].items():
            if vix >= threshold:
                vix_bracket = bracket
                break
        
        drawdown_bracket = "ultra_aggressive"
        for bracket, threshold in self.risk_thresholds["drawdown"].items():
            if drawdown >= threshold:
                drawdown_bracket = bracket
                break
        
        bracket_order = ["ultra_conservative", "conservative", "moderate", "aggressive", "ultra_aggressive"]
        bracket_indices = {
            "conviction": bracket_order.index(conviction_bracket),
            "vix": bracket_order.index(vix_bracket),
            "drawdown": bracket_order.index(drawdown_bracket)
        }
        
        min_index = min(bracket_indices.values())
        final_bracket = bracket_order[min_index]
        
        if final_bracket != self.current_bracket:
            self.bracket_history.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "from_bracket": self.current_bracket,
                "to_bracket": final_bracket,
                "conviction": conviction,
                "vix": vix,
                "drawdown": drawdown
            })
            
            self.current_bracket = final_bracket
            self.current_leverage = self.leverage_brackets[self.current_bracket]
        
    def get_current_leverage(self):
        """
        Get current leverage.
        
        Returns:
        - Current leverage
        """
        return self.current_leverage
        
    def get_current_bracket(self):
        """
        Get current bracket.
        
        Returns:
        - Current bracket
        """
        return self.current_bracket
        
    def get_bracket_history(self):
        """
        Get bracket history.
        
        Returns:
        - Bracket history
        """
        return self.bracket_history
        
    def calculate_position_leverage(self, base_leverage, asset_volatility=None):
        """
        Calculate position leverage based on current bracket.
        
        Parameters:
        - base_leverage: Base leverage
        - asset_volatility: Asset-specific volatility (optional)
        
        Returns:
        - Adjusted leverage
        """
        leverage = self.current_leverage
        
        if asset_volatility is not None:
            vol_adjustment = 0.2 / asset_volatility
            leverage *= vol_adjustment
        
        adjusted_leverage = base_leverage * leverage
        
        adjusted_leverage = min(adjusted_leverage, 10.0)
        
        return adjusted_leverage
