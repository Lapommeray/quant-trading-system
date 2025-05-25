"""
Anti-Loss Guardian - Advanced loss prevention system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

class AntiLossGuardian:
    """
    Advanced anti-loss protection system with multiple redundant safeguards
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Anti-Loss Guardian
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = self._setup_logger()
        self.loss_prevention_active = True
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.emergency_mode = False
        
        self.protection_levels = {
            "level_1": {"drawdown": 0.05, "action": "reduce_position"},
            "level_2": {"drawdown": 0.10, "action": "halt_new_trades"},
            "level_3": {"drawdown": 0.15, "action": "emergency_liquidation"}
        }
        
        self.peak_portfolio_value = None
        
        self.risk_multiplier = 1.0
        
        self.trade_history = []
        self.max_trade_history = 100
        
        self.logger.info("Anti-Loss Guardian initialized")
        
    def _setup_logger(self):
        """Set up logger"""
        logger = logging.getLogger("AntiLossGuardian")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
        
    def check_anti_loss_conditions(self, portfolio_value, current_positions):
        """
        Comprehensive anti-loss condition checking
        
        Parameters:
        - portfolio_value: Current portfolio value
        - current_positions: Dictionary of current positions
        
        Returns:
        - Dictionary with anti-loss check results
        """
        if not self.loss_prevention_active:
            return {"allowed": True, "action": "none"}
            
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = portfolio_value
            
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        ordered_levels = sorted(self.protection_levels.items(), 
                               key=lambda x: x[1]["drawdown"], 
                               reverse=True)
        
        for level, config in ordered_levels:
            if current_drawdown >= config["drawdown"]:
                self.logger.warning(f"Anti-loss {level} triggered: {current_drawdown:.2%} drawdown")
                return {"allowed": False, "action": config["action"], "drawdown": current_drawdown}
                
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.critical(f"Too many consecutive losses ({self.consecutive_losses}) - entering emergency mode")
            self.emergency_mode = True
            return {"allowed": False, "action": "emergency_mode", "consecutive_losses": self.consecutive_losses}
            
        if current_positions:
            total_value = sum(abs(pos) for pos in current_positions.values())
            max_position = max(abs(pos) for pos in current_positions.values())
            
            if max_position / total_value > 0.5:  # Single position > 50% of portfolio
                self.logger.warning(f"Position concentration risk detected: {max_position/total_value:.2%}")
                return {"allowed": False, "action": "reduce_concentration", "concentration": max_position/total_value}
                
        if self._detect_unusual_patterns():
            self.logger.warning("Unusual trading pattern detected")
            return {"allowed": False, "action": "pause_trading", "reason": "unusual_pattern"}
            
        return {"allowed": True, "action": "none"}
        
    def update_trade_result(self, trade_pnl, trade_data=None):
        """
        Update consecutive loss tracking and trade history
        
        Parameters:
        - trade_pnl: Profit/loss from the trade
        - trade_data: Optional additional trade data
        
        Returns:
        - None
        """
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Reset on winning trade
            
        trade_record = {
            "timestamp": datetime.now(),
            "pnl": trade_pnl,
            "data": trade_data
        }
        
        self.trade_history.append(trade_record)
        
        if len(self.trade_history) > self.max_trade_history:
            self.trade_history = self.trade_history[-self.max_trade_history:]
            
        self._adjust_risk_multiplier()
            
    def emergency_protocols(self):
        """
        Emergency loss prevention protocols
        
        Returns:
        - Dictionary with emergency protocol actions
        """
        if self.emergency_mode:
            self.logger.critical("EMERGENCY PROTOCOLS ACTIVATED")
            return {
                "liquidate_all": True,
                "block_new_trades": True,
                "notify_admin": True,
                "create_backup": True
            }
        return {}
        
    def _detect_unusual_patterns(self):
        """
        Detect unusual trading patterns
        
        Returns:
        - Boolean indicating if unusual pattern detected
        """
        if len(self.trade_history) < 10:
            return False
            
        recent_trades = self.trade_history[-10:]
        
        alternating = True
        for i in range(1, len(recent_trades)):
            if (recent_trades[i]["pnl"] > 0) == (recent_trades[i-1]["pnl"] > 0):
                alternating = False
                break
                
        if alternating:
            return True
            
        losses = [t["pnl"] for t in recent_trades if t["pnl"] < 0]
        if len(losses) >= 3:
            if all(losses[i] < losses[i-1] for i in range(1, len(losses))):
                return True
                
        return False
        
    def _adjust_risk_multiplier(self):
        """
        Dynamically adjust risk multiplier based on recent performance
        
        Returns:
        - None
        """
        if len(self.trade_history) < 5:
            return
            
        recent_trades = self.trade_history[-5:]
        
        win_count = sum(1 for t in recent_trades if t["pnl"] > 0)
        win_rate = win_count / len(recent_trades)
        
        profits = [t["pnl"] for t in recent_trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in recent_trades if t["pnl"] < 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if win_rate < 0.4:
            self.risk_multiplier = max(0.5, self.risk_multiplier * 0.9)
        elif win_rate > 0.6 and avg_profit > abs(avg_loss):
            self.risk_multiplier = min(1.0, self.risk_multiplier * 1.1)
            
        self.logger.info(f"Risk multiplier adjusted to {self.risk_multiplier:.2f}")
        
    def get_position_size_multiplier(self):
        """
        Get the current position size multiplier
        
        Returns:
        - Float representing the position size multiplier
        """
        return self.risk_multiplier
        
    def reset_emergency_mode(self):
        """
        Reset emergency mode after manual intervention
        
        Returns:
        - None
        """
        if self.emergency_mode:
            self.emergency_mode = False
            self.consecutive_losses = 0
            self.logger.info("Emergency mode reset")
