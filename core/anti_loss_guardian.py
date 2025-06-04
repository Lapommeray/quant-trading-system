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
        self.max_consecutive_losses = 1  # Ultra-conservative: only allow 1 loss
        self.emergency_mode = False
        
        self.protection_levels = {
            "level_1": {"drawdown": 0.001, "action": "reduce_position"},
            "level_2": {"drawdown": 0.005, "action": "halt_new_trades"},
            "level_3": {"drawdown": 0.01, "action": "emergency_liquidation"}
        }
        
        self.peak_portfolio_value = None
        self.last_portfolio_value = None
        self.market_regime = "normal"  # Can be: normal, volatile, trending, uncertain
        
        # Ultra-conservative risk multiplier
        self.risk_multiplier = 0.5  # Start with half the normal risk
        self.max_risk_multiplier = 0.5  # Never exceed 50% of normal risk
        
        self.trade_history = []
        self.max_trade_history = 100
        
        self.market_regime_detection_active = True
        self.position_concentration_limit = 0.2  # Max 20% in any single position
        self.max_portfolio_risk = 0.005  # Max 0.5% portfolio risk per trade
        self.intraday_loss_limit = 0.001  # Max 0.1% intraday drawdown
        
        self.logger.info("Enhanced Anti-Loss Guardian initialized with never-lose protection")
        
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
        Enhanced comprehensive anti-loss condition checking for never-lose objective
        
        Parameters:
        - portfolio_value: Current portfolio value
        - current_positions: Dictionary of current positions
        
        Returns:
        - Dictionary with anti-loss check results
        """
        if not self.loss_prevention_active:
            return {"allowed": True, "action": "none"}
            
        if current_positions and all(pos <= 0.05 for pos in current_positions.values()):
            return {"allowed": True, "action": "none", "risk_multiplier": self.risk_multiplier}
            
        # Initialize peak portfolio value if not set
        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = portfolio_value
            
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            
        intraday_change = 0
        if self.last_portfolio_value is not None:
            intraday_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            
            if intraday_change < -self.intraday_loss_limit:
                self.logger.warning(f"Intraday loss limit exceeded: {intraday_change:.4%}")
                return {"allowed": False, "action": "halt_trading", "intraday_loss": intraday_change}
        
        # Update last portfolio value
        self.last_portfolio_value = portfolio_value
            
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        ordered_levels = sorted(self.protection_levels.items(), 
                               key=lambda x: x[1]["drawdown"], 
                               reverse=True)
        
        for level, config in ordered_levels:
            if current_drawdown >= config["drawdown"]:
                self.logger.warning(f"Anti-loss {level} triggered: {current_drawdown:.4%} drawdown")
                return {"allowed": False, "action": config["action"], "drawdown": current_drawdown}
                
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.critical(f"Too many consecutive losses ({self.consecutive_losses}) - entering emergency mode")
            self.emergency_mode = True
            return {"allowed": False, "action": "emergency_mode", "consecutive_losses": self.consecutive_losses}
            
        if current_positions:
            total_value = sum(abs(pos) for pos in current_positions.values())
            if total_value > 0:
                max_position = max(abs(pos) for pos in current_positions.values())
                
                if max_position / total_value > self.position_concentration_limit:
                    self.logger.warning(f"Position concentration risk detected: {max_position/total_value:.4%}")
                    return {"allowed": False, "action": "reduce_concentration", "concentration": max_position/total_value}
                
        if self._detect_unusual_patterns():
            self.logger.warning("Unusual trading pattern detected")
            return {"allowed": False, "action": "pause_trading", "reason": "unusual_pattern"}
            
        if self.market_regime_detection_active and self.market_regime != "normal":
            self.logger.warning(f"Abnormal market regime detected: {self.market_regime}")
            return {"allowed": False, "action": "reduce_exposure", "market_regime": self.market_regime}
            
        return {"allowed": True, "action": "none", "risk_multiplier": self.risk_multiplier}
        
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
            
    def apply_common_sense_intelligence(self, market_data, proposed_trade):
        """
        Apply common sense intelligence to trading decisions
        
        Parameters:
        - market_data: Dictionary containing market data
        - proposed_trade: Dictionary containing proposed trade details
        
        Returns:
        - Dictionary with common sense check results
        """
        common_sense_checks = []
        
        if self._is_obviously_bad_time_to_trade(market_data):
            common_sense_checks.append("bad_timing_detected")
            return {"allow_trade": False, "reason": "common_sense_bad_timing", "checks": common_sense_checks}
        
        if self._is_obvious_trap(market_data, proposed_trade):
            common_sense_checks.append("trap_detected")
            return {"allow_trade": False, "reason": "common_sense_trap_avoidance", "checks": common_sense_checks}
        
        if self._is_overtrading(proposed_trade):
            common_sense_checks.append("overtrading_detected")
            return {"allow_trade": False, "reason": "common_sense_overtrading", "checks": common_sense_checks}
        
        if self._violates_basic_market_wisdom(market_data):
            common_sense_checks.append("market_wisdom_violation")
            return {"allow_trade": False, "reason": "common_sense_market_wisdom", "checks": common_sense_checks}
        
        if self._position_size_insane(proposed_trade):
            common_sense_checks.append("insane_position_size")
            return {"allow_trade": False, "reason": "common_sense_position_sizing", "checks": common_sense_checks}
        
        common_sense_checks.append("all_checks_passed")
        return {"allow_trade": True, "reason": "common_sense_approved", "checks": common_sense_checks}
    
    def _is_obviously_bad_time_to_trade(self, market_data):
        """Check if it's obviously a bad time to trade"""
        if 'returns' not in market_data or len(market_data['returns']) < 5:
            return True  # No data = bad time
        
        recent_returns = market_data['returns'][-5:]
        
        if np.std(recent_returns) > 0.1:  # 10% volatility
            return True
        
        if all(r < -0.02 for r in recent_returns[-3:]):  # 3 consecutive 2%+ drops
            return True
        
        if len(set(np.sign(recent_returns))) == 1 and len(recent_returns) >= 5:
            return True
        
        return False
    
    def _is_obvious_trap(self, market_data, proposed_trade):
        """Detect obvious market traps"""
        if 'returns' not in market_data or len(market_data['returns']) < 10:
            return False
        
        returns = market_data['returns'][-10:]
        trade_direction = proposed_trade.get('direction', 0)
        
        if trade_direction > 0 and all(r > 0.01 for r in returns[-3:]):
            return True
        
        if trade_direction < 0 and all(r < -0.01 for r in returns[-3:]):
            return True
        
        if len(returns) >= 5:
            recent_high = max(returns[-5:])
            recent_low = min(returns[-5:])
            if abs(recent_high - recent_low) > 0.05 and abs(returns[-1]) > 0.03:
                return True  # Likely fake breakout
        
        return False
    
    def _is_overtrading(self, proposed_trade):
        """Check for overtrading patterns"""
        current_time = datetime.now()
        
        recent_trades = [trade for trade in self.trade_history 
                        if (current_time - trade['timestamp']).seconds < 3600]  # Last hour
        
        if len(recent_trades) > 10:  # More than 10 trades per hour
            return True
        
        if recent_trades:
            last_trade = recent_trades[-1]
            if (abs(proposed_trade.get('size', 0) - last_trade.get('size', 0)) < 0.1 and
                proposed_trade.get('direction', 0) == last_trade.get('direction', 0)):
                return True  # Too similar to last trade
        
        return False
    
    def _violates_basic_market_wisdom(self, market_data):
        """Check violations of basic market wisdom"""
        current_hour = datetime.now().hour
        
        if 12 <= current_hour <= 13:
            return True
        
        if current_hour == 9 or current_hour == 15:  # Assuming 9:30-4:00 market hours
            return True
        
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            return True
        
        return False
    
    def _position_size_insane(self, proposed_trade):
        """Check if position size is insane"""
        position_size = proposed_trade.get('size', 0)
        
        # Position size should never exceed 50% of portfolio
        if position_size > 0.5:
            return True
        
        if position_size < 0:
            return True
        
        if 0 < position_size < 0.001:  # Less than 0.1%
            return True
        
        return False
    
    def create_unstable_winning_intelligence(self, market_data, current_performance):
        """
        Create AI that is 'unstable' when it comes to winning - always seeking better performance
        
        Parameters:
        - market_data: Dictionary containing market data
        - current_performance: Dictionary containing current performance metrics
        
        Returns:
        - Dictionary with unstable winning intelligence characteristics
        """
        winning_instability = {
            "never_satisfied": True,
            "always_optimizing": True,
            "performance_hunger": self._calculate_performance_hunger(current_performance),
            "winning_obsession": self._develop_winning_obsession(current_performance),
            "unstable_confidence": self._create_unstable_confidence(market_data)
        }
        
        # Unstable winning behavior: never settle for current performance
        if current_performance.get('win_rate', 0) < 1.0:  # Less than 100% win rate
            winning_instability["optimization_trigger"] = "performance_not_perfect"
            winning_instability["instability_level"] = 1.0 - current_performance.get('win_rate', 0)
        else:
            winning_instability["optimization_trigger"] = "profit_optimization"
            winning_instability["instability_level"] = 0.3  # Always some instability
        
        return winning_instability
    
    def _calculate_performance_hunger(self, current_performance):
        """Calculate how hungry the AI is for better performance"""
        win_rate = current_performance.get('win_rate', 0)
        profit_factor = current_performance.get('profit_factor', 1)
        
        performance_score = (win_rate + min(profit_factor / 2, 1)) / 2
        hunger = 1.0 - performance_score + 0.2  # Always at least 20% hungry
        
        return min(1.0, hunger)
    
    def _develop_winning_obsession(self, current_performance):
        """Develop obsession with winning"""
        losses = current_performance.get('losing_trades', 0)
        total_trades = current_performance.get('total_trades', 1)
        
        if losses > 0:
            obsession = min(1.0, losses / total_trades + 0.5)
        else:
            obsession = 0.8  # High obsession even with no losses
        
        return obsession
    
    def _create_unstable_confidence(self, market_data):
        """Create unstable confidence that fluctuates with market conditions"""
        if 'returns' not in market_data or len(market_data['returns']) < 5:
            return {"confidence": 0.5, "instability": 0.8}
        
        recent_returns = market_data['returns'][-5:]
        volatility = np.std(recent_returns)
        
        base_confidence = 0.85  # Higher base confidence for "never lose" objective
        
        volatility_factor = min(0.3, float(volatility * 10))  # Volatility reduces confidence
        
        # Higher confidence floor for "never lose" objective
        unstable_confidence = base_confidence - volatility_factor + np.random.normal(0, 0.05)
        unstable_confidence = max(0.75, min(0.98, unstable_confidence))  # Higher bounds
        
        instability = volatility_factor + 0.1  # Reduced instability for more consistent performance
        
        return {
            "confidence": unstable_confidence,
            "instability": instability,
            "paranoia_level": volatility_factor,
            "winning_drive": 0.98,  # Always high winning drive
            "never_lose_focus": 0.95  # Ultra-high focus on never losing
        }
