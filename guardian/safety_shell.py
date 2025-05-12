"""
Quantum Safety Shell

Risk firewall enforcing leverage, drawdown, and size limits for the QMP Overrider system.
"""

from AlgorithmImports import *
import logging
import json
import os
from datetime import datetime

class QuantumFirewall:
    """
    Risk firewall enforcing leverage, drawdown, and size limits.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Quantum Firewall.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("QuantumFirewall")
        self.logger.setLevel(logging.INFO)
        
        self.max_leverage = 50.0
        self.max_drawdown = 0.2  # 20% drawdown
        self.max_position = 0.1  # 10% of capital
        self.max_daily_loss = 0.05  # 5% daily loss
        self.max_trade_frequency = 100  # Max trades per day
        
        self.validation_history = []
        
        self.quarantine_dir = "/quarantine"
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        self.quarantine_log_path = os.path.join(self.quarantine_dir, "log.txt")
        
        self.logger.info("Quantum Firewall initialized")
        
    def validate_strategy(self, strategy):
        """
        Validate a strategy against risk limits.
        
        Parameters:
        - strategy: Strategy to validate
        
        Returns:
        - Boolean indicating if strategy is valid
        """
        self.logger.info(f"Validating strategy: {strategy.name if hasattr(strategy, 'name') else 'Unknown'}")
        
        leverage = getattr(strategy, "leverage", 1.0)
        estimated_drawdown = getattr(strategy, "estimated_drawdown", 0.0)
        max_position_size = getattr(strategy, "max_position_size", 0.0)
        daily_loss_limit = getattr(strategy, "daily_loss_limit", 0.0)
        trade_frequency = getattr(strategy, "trade_frequency", 0)
        
        checks = [
            {"name": "leverage", "value": leverage, "limit": self.max_leverage, "valid": leverage <= self.max_leverage},
            {"name": "drawdown", "value": estimated_drawdown, "limit": self.max_drawdown, "valid": estimated_drawdown < self.max_drawdown},
            {"name": "position_size", "value": max_position_size, "limit": self.max_position, "valid": max_position_size < self.max_position},
            {"name": "daily_loss", "value": daily_loss_limit, "limit": self.max_daily_loss, "valid": daily_loss_limit <= self.max_daily_loss},
            {"name": "trade_frequency", "value": trade_frequency, "limit": self.max_trade_frequency, "valid": trade_frequency <= self.max_trade_frequency}
        ]
        
        all_valid = all(check["valid"] for check in checks)
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.name if hasattr(strategy, "name") else "Unknown",
            "valid": all_valid,
            "checks": checks
        }
        
        self.validation_history.append(validation_result)
        
        if not all_valid:
            self.logger.warning(f"Strategy validation failed: {validation_result}")
            self._quarantine(strategy, checks)
        else:
            self.logger.info(f"Strategy validation passed: {validation_result['strategy']}")
            
        return all_valid
        
    def _quarantine(self, strategy, failed_checks):
        """
        Quarantine a banned strategy.
        
        Parameters:
        - strategy: Strategy to quarantine
        - failed_checks: List of failed checks
        """
        strategy_name = strategy.name if hasattr(strategy, "name") else "Unknown"
        
        self.logger.warning(f"Quarantining strategy: {strategy_name}")
        
        quarantine_entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "failed_checks": [check for check in failed_checks if not check["valid"]]
        }
        
        try:
            with open(self.quarantine_log_path, 'a') as f:
                f.write(f"Banned strategy: {strategy_name}\n")
                f.write(f"Timestamp: {quarantine_entry['timestamp']}\n")
                f.write("Failed checks:\n")
                
                for check in quarantine_entry["failed_checks"]:
                    f.write(f"  - {check['name']}: {check['value']} (limit: {check['limit']})\n")
                    
                f.write("\n")
                
            self.logger.info(f"Strategy quarantined: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error logging to quarantine: {str(e)}")
            
        if hasattr(strategy, "path") and os.path.exists(strategy.path):
            try:
                filename = os.path.basename(strategy.path)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                quarantine_filename = f"{timestamp}_{filename}"
                quarantine_path = os.path.join(self.quarantine_dir, quarantine_filename)
                
                shutil.copy2(strategy.path, quarantine_path)
                
                self.logger.info(f"Strategy file quarantined: {quarantine_path}")
                
            except Exception as e:
                self.logger.error(f"Error quarantining strategy file: {str(e)}")
        
    def validate_trade(self, trade):
        """
        Validate a trade against risk limits.
        
        Parameters:
        - trade: Trade to validate
        
        Returns:
        - Boolean indicating if trade is valid
        """
        self.logger.info(f"Validating trade: {trade}")
        
        symbol = trade.get("symbol", "Unknown")
        quantity = trade.get("quantity", 0)
        direction = trade.get("direction", "long")
        leverage = trade.get("leverage", 1.0)
        position_size = trade.get("position_size", 0.0)
        
        current_leverage = self.algorithm.Portfolio.TotalHoldingsValue / self.algorithm.Portfolio.TotalPortfolioValue if hasattr(self.algorithm, "Portfolio") else 1.0
        
        checks = [
            {"name": "leverage", "value": current_leverage + leverage, "limit": self.max_leverage, "valid": current_leverage + leverage <= self.max_leverage},
            {"name": "position_size", "value": position_size, "limit": self.max_position, "valid": position_size <= self.max_position}
        ]
        
        all_valid = all(check["valid"] for check in checks)
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "trade": {
                "symbol": symbol,
                "quantity": quantity,
                "direction": direction
            },
            "valid": all_valid,
            "checks": checks
        }
        
        self.validation_history.append(validation_result)
        
        if not all_valid:
            self.logger.warning(f"Trade validation failed: {validation_result}")
        else:
            self.logger.info(f"Trade validation passed: {symbol}")
            
        return all_valid
        
    def validate_portfolio(self):
        """
        Validate current portfolio against risk limits.
        
        Returns:
        - Boolean indicating if portfolio is valid
        """
        self.logger.info("Validating portfolio")
        
        if not hasattr(self.algorithm, "Portfolio"):
            self.logger.warning("No portfolio available for validation")
            return True
            
        current_leverage = self.algorithm.Portfolio.TotalHoldingsValue / self.algorithm.Portfolio.TotalPortfolioValue
        
        max_position_size = 0.0
        max_position_symbol = "None"
        
        for kvp in self.algorithm.Portfolio:
            security_hold = kvp.Value
            
            if security_hold.Invested:
                position_size = abs(security_hold.HoldingsValue) / self.algorithm.Portfolio.TotalPortfolioValue
                
                if position_size > max_position_size:
                    max_position_size = position_size
                    max_position_symbol = str(security_hold.Symbol)
        
        checks = [
            {"name": "leverage", "value": current_leverage, "limit": self.max_leverage, "valid": current_leverage <= self.max_leverage},
            {"name": "position_size", "value": max_position_size, "limit": self.max_position, "valid": max_position_size <= self.max_position}
        ]
        
        all_valid = all(check["valid"] for check in checks)
        
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": {
                "leverage": current_leverage,
                "max_position_size": max_position_size,
                "max_position_symbol": max_position_symbol
            },
            "valid": all_valid,
            "checks": checks
        }
        
        self.validation_history.append(validation_result)
        
        if not all_valid:
            self.logger.warning(f"Portfolio validation failed: {validation_result}")
        else:
            self.logger.info("Portfolio validation passed")
            
        return all_valid
        
    def enforce_portfolio_limits(self):
        """
        Enforce portfolio limits by liquidating positions if necessary.
        
        Returns:
        - Boolean indicating if enforcement was needed
        """
        self.logger.info("Enforcing portfolio limits")
        
        if self.validate_portfolio():
            return False
            
        if not hasattr(self.algorithm, "Portfolio"):
            self.logger.warning("No portfolio available for enforcement")
            return False
            
        current_leverage = self.algorithm.Portfolio.TotalHoldingsValue / self.algorithm.Portfolio.TotalPortfolioValue
        
        if current_leverage > self.max_leverage:
            self.logger.warning(f"Leverage too high: {current_leverage} (limit: {self.max_leverage})")
            
            reduction_factor = self.max_leverage / current_leverage
            
            for kvp in self.algorithm.Portfolio:
                security_hold = kvp.Value
                
                if security_hold.Invested:
                    self.algorithm.SetHoldings(security_hold.Symbol, security_hold.Quantity * reduction_factor)
                    
                    self.logger.info(f"Reduced position in {security_hold.Symbol} by factor {reduction_factor}")
            
            return True
            
        positions_reduced = False
        
        for kvp in self.algorithm.Portfolio:
            security_hold = kvp.Value
            
            if security_hold.Invested:
                position_size = abs(security_hold.HoldingsValue) / self.algorithm.Portfolio.TotalPortfolioValue
                
                if position_size > self.max_position:
                    self.logger.warning(f"Position too large: {security_hold.Symbol} at {position_size} (limit: {self.max_position})")
                    
                    new_quantity = security_hold.Quantity * (self.max_position / position_size)
                    self.algorithm.SetHoldings(security_hold.Symbol, new_quantity)
                    
                    self.logger.info(f"Reduced position in {security_hold.Symbol} from {position_size} to {self.max_position}")
                    
                    positions_reduced = True
        
        return positions_reduced
        
    def set_risk_limits(self, max_leverage=None, max_drawdown=None, max_position=None, max_daily_loss=None, max_trade_frequency=None):
        """
        Set risk limits.
        
        Parameters:
        - max_leverage: Maximum leverage
        - max_drawdown: Maximum drawdown
        - max_position: Maximum position size
        - max_daily_loss: Maximum daily loss
        - max_trade_frequency: Maximum trade frequency
        """
        if max_leverage is not None:
            self.max_leverage = max_leverage
            
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
            
        if max_position is not None:
            self.max_position = max_position
            
        if max_daily_loss is not None:
            self.max_daily_loss = max_daily_loss
            
        if max_trade_frequency is not None:
            self.max_trade_frequency = max_trade_frequency
            
        self.logger.info(f"Risk limits updated: leverage={self.max_leverage}, drawdown={self.max_drawdown}, position={self.max_position}, daily_loss={self.max_daily_loss}, trade_frequency={self.max_trade_frequency}")
        
    def get_validation_history(self):
        """
        Get validation history.
        
        Returns:
        - Validation history
        """
        return self.validation_history
        
    def get_risk_limits(self):
        """
        Get current risk limits.
        
        Returns:
        - Dictionary of risk limits
        """
        return {
            "max_leverage": self.max_leverage,
            "max_drawdown": self.max_drawdown,
            "max_position": self.max_position,
            "max_daily_loss": self.max_daily_loss,
            "max_trade_frequency": self.max_trade_frequency
        }
