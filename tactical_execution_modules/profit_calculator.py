"""
Profit Calculator

Implements Fibonacci-based take-profit and stop-loss logic for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

class ProfitCalculator:
    """
    Implements Fibonacci-based take-profit and stop-loss logic.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Profit Calculator.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("ProfitCalculator")
        self.logger.setLevel(logging.INFO)
        
        self.fib_levels = {
            "extension": [1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236],
            "retracement": [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        }
        
        self.tp_settings = {
            "default": {
                "levels": [0.382, 0.618, 1.0, 1.618],
                "weights": [0.2, 0.3, 0.3, 0.2]
            },
            "aggressive": {
                "levels": [0.618, 1.0, 1.618, 2.618],
                "weights": [0.1, 0.2, 0.4, 0.3]
            },
            "conservative": {
                "levels": [0.236, 0.382, 0.618, 1.0],
                "weights": [0.3, 0.3, 0.3, 0.1]
            }
        }
        
        self.sl_settings = {
            "default": {
                "levels": [0.382, 0.5, 0.618],
                "weights": [0.3, 0.4, 0.3]
            },
            "aggressive": {
                "levels": [0.5, 0.618, 0.786],
                "weights": [0.2, 0.3, 0.5]
            },
            "conservative": {
                "levels": [0.236, 0.382, 0.5],
                "weights": [0.4, 0.4, 0.2]
            }
        }
        
        self.trade_data = {}
        
    def calculate_targets(self, symbol, entry_price, direction, volatility=None, profile="default"):
        """
        Calculate take-profit and stop-loss targets.
        
        Parameters:
        - symbol: The trading symbol
        - entry_price: Entry price
        - direction: Trade direction ("LONG" or "SHORT")
        - volatility: Price volatility (optional)
        - profile: Risk profile ("default", "aggressive", or "conservative")
        
        Returns:
        - Dictionary containing target levels
        """
        if entry_price <= 0:
            self.logger.error(f"Invalid entry price: {entry_price}")
            return None
            
        if direction not in ["LONG", "SHORT"]:
            self.logger.error(f"Invalid direction: {direction}")
            return None
            
        if profile not in self.tp_settings:
            self.logger.warning(f"Invalid profile: {profile}, using default")
            profile = "default"
        
        tp_levels = self.tp_settings[profile]["levels"]
        tp_weights = self.tp_settings[profile]["weights"]
        sl_levels = self.sl_settings[profile]["levels"]
        sl_weights = self.sl_settings[profile]["weights"]
        
        avg_move = self._calculate_avg_move(symbol, volatility)
        
        tp_targets = []
        for level in tp_levels:
            if direction == "LONG":
                price = entry_price * (1 + level * avg_move)
            else:  # SHORT
                price = entry_price * (1 - level * avg_move)
                
            tp_targets.append({
                "level": level,
                "price": price
            })
        
        sl_targets = []
        for level in sl_levels:
            if direction == "LONG":
                price = entry_price * (1 - level * avg_move)
            else:  # SHORT
                price = entry_price * (1 + level * avg_move)
                
            sl_targets.append({
                "level": level,
                "price": price
            })
        
        weighted_tp = 0.0
        for target, weight in zip(tp_targets, tp_weights):
            weighted_tp += target["price"] * weight
            
        weighted_sl = 0.0
        for target, weight in zip(sl_targets, sl_weights):
            weighted_sl += target["price"] * weight
        
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.trade_data[trade_id] = {
            "symbol": symbol,
            "entry_price": entry_price,
            "direction": direction,
            "profile": profile,
            "avg_move": avg_move,
            "tp_targets": tp_targets,
            "sl_targets": sl_targets,
            "weighted_tp": weighted_tp,
            "weighted_sl": weighted_sl,
            "risk_reward": abs(weighted_tp - entry_price) / abs(weighted_sl - entry_price) if abs(weighted_sl - entry_price) > 0 else 0.0
        }
        
        return {
            "trade_id": trade_id,
            "entry_price": entry_price,
            "direction": direction,
            "profile": profile,
            "avg_move": avg_move,
            "tp_targets": tp_targets,
            "sl_targets": sl_targets,
            "weighted_tp": weighted_tp,
            "weighted_sl": weighted_sl,
            "risk_reward": self.trade_data[trade_id]["risk_reward"]
        }
        
    def _calculate_avg_move(self, symbol, volatility=None):
        """
        Calculate average move size based on volatility.
        
        Parameters:
        - symbol: The trading symbol
        - volatility: Price volatility (optional)
        
        Returns:
        - Average move size
        """
        if volatility is not None:
            return volatility
        else:
            return 0.02
        
    def calculate_position_exits(self, trade_id, current_price, partial_exits=True):
        """
        Calculate position exit signals based on current price.
        
        Parameters:
        - trade_id: Trade identifier
        - current_price: Current price
        - partial_exits: Whether to allow partial exits
        
        Returns:
        - Dictionary containing exit signals
        """
        if trade_id not in self.trade_data:
            self.logger.error(f"Invalid trade ID: {trade_id}")
            return None
            
        trade = self.trade_data[trade_id]
        entry_price = trade["entry_price"]
        direction = trade["direction"]
        tp_targets = trade["tp_targets"]
        sl_targets = trade["sl_targets"]
        
        exit_signals = {
            "full_exit": False,
            "partial_exits": [],
            "stop_loss": False,
            "take_profit": False,
            "exit_price": None,
            "exit_level": None,
            "exit_type": None
        }
        
        for sl in sl_targets:
            if (direction == "LONG" and current_price <= sl["price"]) or \
               (direction == "SHORT" and current_price >= sl["price"]):
                exit_signals["full_exit"] = True
                exit_signals["stop_loss"] = True
                exit_signals["exit_price"] = current_price
                exit_signals["exit_level"] = sl["level"]
                exit_signals["exit_type"] = "STOP_LOSS"
                break
        
        if not exit_signals["full_exit"] and partial_exits:
            for i, tp in enumerate(tp_targets):
                if (direction == "LONG" and current_price >= tp["price"]) or \
                   (direction == "SHORT" and current_price <= tp["price"]):
                    exit_size = 1.0 / len(tp_targets)
                    
                    exit_signals["partial_exits"].append({
                        "exit_price": current_price,
                        "exit_level": tp["level"],
                        "exit_size": exit_size,
                        "exit_type": "TAKE_PROFIT"
                    })
                    
                    if i == len(tp_targets) - 1:
                        exit_signals["full_exit"] = True
                        exit_signals["take_profit"] = True
                        exit_signals["exit_price"] = current_price
                        exit_signals["exit_level"] = tp["level"]
                        exit_signals["exit_type"] = "TAKE_PROFIT"
        
        return exit_signals
        
    def calculate_trailing_stop(self, trade_id, highest_price, lowest_price, atr=None):
        """
        Calculate trailing stop level.
        
        Parameters:
        - trade_id: Trade identifier
        - highest_price: Highest price since entry
        - lowest_price: Lowest price since entry
        - atr: Average True Range (optional)
        
        Returns:
        - Trailing stop price
        """
        if trade_id not in self.trade_data:
            self.logger.error(f"Invalid trade ID: {trade_id}")
            return None
            
        trade = self.trade_data[trade_id]
        entry_price = trade["entry_price"]
        direction = trade["direction"]
        
        multiplier = 2.0 if atr is None else 2.0
        
        if direction == "LONG":
            trailing_stop = highest_price * (1 - 0.382 * trade["avg_move"])
            
            if highest_price > entry_price * (1 + 0.618 * trade["avg_move"]):
                trailing_stop = max(trailing_stop, entry_price)
        else:  # SHORT
            trailing_stop = lowest_price * (1 + 0.382 * trade["avg_move"])
            
            if lowest_price < entry_price * (1 - 0.618 * trade["avg_move"]):
                trailing_stop = min(trailing_stop, entry_price)
        
        return trailing_stop
        
    def calculate_risk_reward(self, entry_price, take_profit, stop_loss):
        """
        Calculate risk-reward ratio.
        
        Parameters:
        - entry_price: Entry price
        - take_profit: Take profit price
        - stop_loss: Stop loss price
        
        Returns:
        - Risk-reward ratio
        """
        reward = abs(take_profit - entry_price)
        
        risk = abs(stop_loss - entry_price)
        
        if risk > 0:
            risk_reward = reward / risk
        else:
            risk_reward = 0.0
            
        return risk_reward
        
    def get_trade_data(self, trade_id):
        """
        Get trade data.
        
        Parameters:
        - trade_id: Trade identifier
        
        Returns:
        - Trade data
        """
        if trade_id in self.trade_data:
            return self.trade_data[trade_id]
        else:
            return None
            
    def get_all_trades(self):
        """
        Get all trade data.
        
        Returns:
        - Dictionary of all trades
        """
        return self.trade_data
