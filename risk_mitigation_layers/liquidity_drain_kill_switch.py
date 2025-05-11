"""
Liquidity Drain Kill-Switch

Exits if reverse repo > $700B for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class LiquidityDrainKillSwitch:
    """
    Exits if reverse repo > $700B.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Liquidity Drain Kill-Switch.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("LiquidityDrainKillSwitch")
        self.logger.setLevel(logging.INFO)
        
        self.warning_threshold = 500.0
        self.danger_threshold = 700.0
        self.critical_threshold = 1000.0
        
        self.liquidity_metrics = {
            "reverse_repo": 0.0,
            "reverse_repo_change_1d": 0.0,
            "reverse_repo_change_5d": 0.0,
            "reverse_repo_change_20d": 0.0,
            "fed_balance_sheet": 0.0,
            "fed_balance_sheet_change_1d": 0.0,
            "excess_reserves": 0.0,
            "excess_reserves_change_1d": 0.0
        }
        
        self.kill_switch_status = {
            "active": False,
            "warning_level": "NORMAL",
            "trigger_time": None,
            "trigger_reason": None
        }
        
        self.kill_switch_history = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=6)  # Less frequent updates for Fed data
        
    def update(self, current_time, liquidity_data=None):
        """
        Update the liquidity drain kill-switch with latest data.
        
        Parameters:
        - current_time: Current datetime
        - liquidity_data: Liquidity data (optional)
        
        Returns:
        - Dictionary containing kill-switch results
        """
        if current_time - self.last_update < self.update_frequency and liquidity_data is None:
            return {
                "kill_switch_status": self.kill_switch_status,
                "liquidity_metrics": self.liquidity_metrics
            }
            
        if liquidity_data is not None:
            self._update_liquidity_metrics(liquidity_data)
        else:
            self._update_liquidity_metrics_internal()
        
        self._check_kill_switch_conditions(current_time)
        
        self.last_update = current_time
        
        return {
            "kill_switch_status": self.kill_switch_status,
            "liquidity_metrics": self.liquidity_metrics
        }
        
    def _update_liquidity_metrics(self, liquidity_data):
        """
        Update liquidity metrics.
        
        Parameters:
        - liquidity_data: Liquidity data
        """
        for key, value in liquidity_data.items():
            if key in self.liquidity_metrics:
                self.liquidity_metrics[key] = value
        
    def _update_liquidity_metrics_internal(self):
        """
        Update liquidity metrics internally.
        """
        
        self.liquidity_metrics = {
            "reverse_repo": 650.0,  # $650 billion
            "reverse_repo_change_1d": 15.0,  # +$15 billion in 1 day
            "reverse_repo_change_5d": 45.0,  # +$45 billion in 5 days
            "reverse_repo_change_20d": 120.0,  # +$120 billion in 20 days
            "fed_balance_sheet": 8500.0,  # $8.5 trillion
            "fed_balance_sheet_change_1d": -5.0,  # -$5 billion in 1 day
            "excess_reserves": 3200.0,  # $3.2 trillion
            "excess_reserves_change_1d": -10.0  # -$10 billion in 1 day
        }
        
    def _check_kill_switch_conditions(self, current_time):
        """
        Check kill switch conditions.
        
        Parameters:
        - current_time: Current datetime
        """
        reverse_repo = self.liquidity_metrics["reverse_repo"]
        reverse_repo_change_1d = self.liquidity_metrics["reverse_repo_change_1d"]
        reverse_repo_change_5d = self.liquidity_metrics["reverse_repo_change_5d"]
        fed_balance_sheet_change_1d = self.liquidity_metrics["fed_balance_sheet_change_1d"]
        excess_reserves_change_1d = self.liquidity_metrics["excess_reserves_change_1d"]
        
        if reverse_repo > self.critical_threshold:
            self._activate_kill_switch(current_time, "CRITICAL", f"Reverse repo > ${self.critical_threshold}B (${reverse_repo}B)")
            return
        
        if reverse_repo > self.danger_threshold:
            self._activate_kill_switch(current_time, "DANGER", f"Reverse repo > ${self.danger_threshold}B (${reverse_repo}B)")
            return
        
        if reverse_repo > self.warning_threshold:
            if self.kill_switch_status["warning_level"] != "WARNING":
                self._activate_kill_switch(current_time, "WARNING", f"Reverse repo > ${self.warning_threshold}B (${reverse_repo}B)")
            return
        
        if reverse_repo_change_1d > 50.0:  # $50 billion increase in 1 day
            self._activate_kill_switch(current_time, "DANGER", f"Reverse repo increased by ${reverse_repo_change_1d}B in 1 day")
            return
        
        if reverse_repo_change_5d > 150.0:  # $150 billion increase in 5 days
            self._activate_kill_switch(current_time, "DANGER", f"Reverse repo increased by ${reverse_repo_change_5d}B in 5 days")
            return
        
        if (reverse_repo > self.warning_threshold and 
            reverse_repo_change_1d > 20.0 and 
            fed_balance_sheet_change_1d < -20.0 and 
            excess_reserves_change_1d < -20.0):
            self._activate_kill_switch(current_time, "DANGER", "Combined liquidity drain detected")
            return
        
        if self.kill_switch_status["warning_level"] != "NORMAL":
            self._deactivate_kill_switch(current_time)
        
    def _activate_kill_switch(self, current_time, warning_level, reason):
        """
        Activate kill switch.
        
        Parameters:
        - current_time: Current datetime
        - warning_level: Warning level
        - reason: Activation reason
        """
        if (self.kill_switch_status["active"] and 
            self._warning_level_value(self.kill_switch_status["warning_level"]) >= 
            self._warning_level_value(warning_level)):
            return
        
        previous_status = self.kill_switch_status.copy()
        
        self.kill_switch_status["active"] = warning_level in ["DANGER", "CRITICAL"]
        self.kill_switch_status["warning_level"] = warning_level
        self.kill_switch_status["trigger_time"] = current_time
        self.kill_switch_status["trigger_reason"] = reason
        
        self.kill_switch_history.append({
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "action": "ACTIVATE" if self.kill_switch_status["active"] else "WARNING",
            "from_level": previous_status["warning_level"],
            "to_level": warning_level,
            "reason": reason,
            "reverse_repo": self.liquidity_metrics["reverse_repo"]
        })
        
        self.logger.warning(f"Kill switch {warning_level}: {reason}")
        
    def _deactivate_kill_switch(self, current_time):
        """
        Deactivate kill switch.
        
        Parameters:
        - current_time: Current datetime
        """
        previous_status = self.kill_switch_status.copy()
        
        self.kill_switch_status["active"] = False
        self.kill_switch_status["warning_level"] = "NORMAL"
        self.kill_switch_status["trigger_time"] = None
        self.kill_switch_status["trigger_reason"] = None
        
        self.kill_switch_history.append({
            "date": current_time.strftime("%Y-%m-%d"),
            "time": current_time.strftime("%H:%M:%S"),
            "action": "DEACTIVATE",
            "from_level": previous_status["warning_level"],
            "to_level": "NORMAL",
            "reason": "Liquidity conditions normalized",
            "reverse_repo": self.liquidity_metrics["reverse_repo"]
        })
        
        self.logger.info("Kill switch deactivated: Liquidity conditions normalized")
        
    def _warning_level_value(self, warning_level):
        """
        Get numeric value for warning level.
        
        Parameters:
        - warning_level: Warning level
        
        Returns:
        - Numeric value
        """
        warning_levels = {
            "NORMAL": 0,
            "WARNING": 1,
            "DANGER": 2,
            "CRITICAL": 3
        }
        
        return warning_levels.get(warning_level, 0)
        
    def is_kill_switch_active(self):
        """
        Check if kill switch is active.
        
        Returns:
        - Boolean indicating if kill switch is active
        """
        return self.kill_switch_status["active"]
        
    def get_warning_level(self):
        """
        Get current warning level.
        
        Returns:
        - Current warning level
        """
        return self.kill_switch_status["warning_level"]
        
    def get_kill_switch_status(self):
        """
        Get kill switch status.
        
        Returns:
        - Kill switch status
        """
        return self.kill_switch_status
        
    def get_kill_switch_history(self):
        """
        Get kill switch history.
        
        Returns:
        - Kill switch history
        """
        return self.kill_switch_history
        
    def calculate_position_size(self, base_size, asset_liquidity_sensitivity=None):
        """
        Calculate position size based on kill switch status.
        
        Parameters:
        - base_size: Base position size
        - asset_liquidity_sensitivity: Asset sensitivity to liquidity (optional)
        
        Returns:
        - Adjusted position size
        """
        warning_level = self.kill_switch_status["warning_level"]
        
        if warning_level == "CRITICAL":
            multiplier = 0.0  # No position
        elif warning_level == "DANGER":
            multiplier = 0.25  # 75% reduction
        elif warning_level == "WARNING":
            multiplier = 0.5  # 50% reduction
        else:  # NORMAL
            multiplier = 1.0  # No reduction
        
        if asset_liquidity_sensitivity is not None:
            sensitivity_adjustment = 1.0 - (asset_liquidity_sensitivity * 0.5)
            multiplier *= sensitivity_adjustment
        
        adjusted_size = base_size * multiplier
        
        return adjusted_size
