"""
Self-Destruct Protocol Module

This module automatically detects and disables failing strategies to protect capital.
It implements a multi-level safety system that isolates underperforming components
without compromising the overall system integrity.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os

class SelfDestructProtocol:
    """
    Automatically detects and disables failing strategies to protect capital.
    
    This module implements a multi-level safety system that isolates underperforming
    components without compromising the overall system integrity.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Self-Destruct Protocol module.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("SelfDestructProtocol")
        self.logger.setLevel(logging.INFO)
        
        self.strategy_performance = {}
        self.module_performance = {}
        self.symbol_performance = {}
        
        self.thresholds = {
            "drawdown": {
                "warning": 0.05,  # 5% drawdown
                "danger": 0.10,   # 10% drawdown
                "critical": 0.15  # 15% drawdown
            },
            "consecutive_losses": {
                "warning": 3,
                "danger": 5,
                "critical": 7
            },
            "win_rate": {
                "warning": 0.4,  # 40% win rate
                "danger": 0.3,   # 30% win rate
                "critical": 0.2  # 20% win rate
            },
            "profit_factor": {
                "warning": 0.9,
                "danger": 0.7,
                "critical": 0.5
            }
        }
        
        self.isolated_strategies = {}
        self.isolated_modules = {}
        self.isolated_symbols = {}
        
        self.recovery_conditions = {
            "min_idle_time": timedelta(hours=6),
            "min_market_change": 0.02,  # 2% market change
            "min_successful_signals": 3
        }
        
        self.isolation_log = []
        self.recovery_log = []
        
        self.log_dir = os.path.join(algorithm.DataFolder, "data", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.isolation_log_path = os.path.join(self.log_dir, "isolation_log.json")
        self.recovery_log_path = os.path.join(self.log_dir, "recovery_log.json")
        
        self._load_logs()
        
        algorithm.Debug("Self-Destruct Protocol initialized")
    
    def update_performance(self, strategy_name, module_name, symbol, trade_result):
        """
        Update performance metrics for a strategy, module, or symbol.
        
        Parameters:
        - strategy_name: Name of the strategy
        - module_name: Name of the module
        - symbol: Trading symbol
        - trade_result: Dictionary containing trade result information
        
        Returns:
        - Dictionary containing updated performance metrics
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = self._create_performance_record()
        
        if module_name not in self.module_performance:
            self.module_performance[module_name] = self._create_performance_record()
        
        symbol_str = str(symbol)
        if symbol_str not in self.symbol_performance:
            self.symbol_performance[symbol_str] = self._create_performance_record()
        
        is_win = trade_result.get("is_win", False)
        profit_loss = trade_result.get("profit_loss", 0)
        entry_time = trade_result.get("entry_time", self.algorithm.Time)
        exit_time = trade_result.get("exit_time", self.algorithm.Time)
        
        self._update_performance_record(self.strategy_performance[strategy_name], is_win, profit_loss)
        
        self._update_performance_record(self.module_performance[module_name], is_win, profit_loss)
        
        self._update_performance_record(self.symbol_performance[symbol_str], is_win, profit_loss)
        
        self._check_self_destruct_conditions(strategy_name, module_name, symbol_str)
        
        self._check_recovery_conditions()
        
        return {
            "strategy": self.strategy_performance[strategy_name],
            "module": self.module_performance[module_name],
            "symbol": self.symbol_performance[symbol_str]
        }
    
    def _create_performance_record(self):
        """
        Create a new performance record.
        
        Returns:
        - Dictionary containing performance metrics
        """
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "max_profit": 0.0,
            "max_drawdown": 0.0,
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "profit_factor": 0.0,
            "last_updated": self.algorithm.Time,
            "trade_history": []
        }
    
    def _update_performance_record(self, record, is_win, profit_loss):
        """
        Update a performance record with new trade information.
        
        Parameters:
        - record: Performance record to update
        - is_win: Boolean indicating if the trade was profitable
        - profit_loss: Profit or loss amount
        """
        record["total_trades"] += 1
        
        if is_win:
            record["wins"] += 1
            record["consecutive_wins"] += 1
            record["consecutive_losses"] = 0
        else:
            record["losses"] += 1
            record["consecutive_losses"] += 1
            record["consecutive_wins"] = 0
        
        record["win_rate"] = record["wins"] / record["total_trades"] if record["total_trades"] > 0 else 0.0
        
        record["profit_loss"] += profit_loss
        
        record["max_profit"] = max(record["max_profit"], record["profit_loss"])
        
        if record["profit_loss"] < record["max_profit"]:
            current_drawdown = (record["max_profit"] - record["profit_loss"]) / record["max_profit"] if record["max_profit"] > 0 else 0
            record["max_drawdown"] = max(record["max_drawdown"], current_drawdown)
        
        total_profit = sum(max(0, t["profit_loss"]) for t in record["trade_history"])
        total_loss = sum(abs(min(0, t["profit_loss"])) for t in record["trade_history"])
        record["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        record["trade_history"].append({
            "timestamp": self.algorithm.Time,
            "is_win": is_win,
            "profit_loss": profit_loss
        })
        
        if len(record["trade_history"]) > 100:
            record["trade_history"] = record["trade_history"][-100:]
        
        record["last_updated"] = self.algorithm.Time
    
    def _check_self_destruct_conditions(self, strategy_name, module_name, symbol_str):
        """
        Check if any component should be isolated due to poor performance.
        
        Parameters:
        - strategy_name: Name of the strategy
        - module_name: Name of the module
        - symbol_str: String representation of the symbol
        
        Returns:
        - Boolean indicating if any component was isolated
        """
        isolated = False
        
        if strategy_name not in self.isolated_strategies:
            strategy_record = self.strategy_performance[strategy_name]
            
            if self._should_isolate(strategy_record):
                self.isolated_strategies[strategy_name] = {
                    "isolation_time": self.algorithm.Time,
                    "reason": self._get_isolation_reason(strategy_record),
                    "metrics": strategy_record.copy(),
                    "recovery_attempts": 0
                }
                
                self.logger.warning(f"Strategy {strategy_name} isolated due to poor performance")
                isolated = True
                
                self.isolation_log.append({
                    "timestamp": self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "component_type": "strategy",
                    "component_name": strategy_name,
                    "reason": self.isolated_strategies[strategy_name]["reason"],
                    "metrics": {k: v for k, v in strategy_record.items() if k != "trade_history"}
                })
        
        if module_name not in self.isolated_modules:
            module_record = self.module_performance[module_name]
            
            if self._should_isolate(module_record):
                self.isolated_modules[module_name] = {
                    "isolation_time": self.algorithm.Time,
                    "reason": self._get_isolation_reason(module_record),
                    "metrics": module_record.copy(),
                    "recovery_attempts": 0
                }
                
                self.logger.warning(f"Module {module_name} isolated due to poor performance")
                isolated = True
                
                self.isolation_log.append({
                    "timestamp": self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "component_type": "module",
                    "component_name": module_name,
                    "reason": self.isolated_modules[module_name]["reason"],
                    "metrics": {k: v for k, v in module_record.items() if k != "trade_history"}
                })
        
        if symbol_str not in self.isolated_symbols:
            symbol_record = self.symbol_performance[symbol_str]
            
            if self._should_isolate(symbol_record):
                self.isolated_symbols[symbol_str] = {
                    "isolation_time": self.algorithm.Time,
                    "reason": self._get_isolation_reason(symbol_record),
                    "metrics": symbol_record.copy(),
                    "recovery_attempts": 0
                }
                
                self.logger.warning(f"Symbol {symbol_str} isolated due to poor performance")
                isolated = True
                
                self.isolation_log.append({
                    "timestamp": self.algorithm.Time.strftime("%Y-%m-%d %H:%M:%S"),
                    "component_type": "symbol",
                    "component_name": symbol_str,
                    "reason": self.isolated_symbols[symbol_str]["reason"],
                    "metrics": {k: v for k, v in symbol_record.items() if k != "trade_history"}
                })
        
        if isolated:
            self._save_logs()
        
        return isolated
    
    def _should_isolate(self, record):
        """
        Determine if a component should be isolated based on performance metrics.
        
        Parameters:
        - record: Performance record to check
        
        Returns:
        - Boolean indicating if the component should be isolated
        """
        if record["total_trades"] < 5:
            return False
        
        if record["max_drawdown"] >= self.thresholds["drawdown"]["critical"]:
            return True
        
        if record["consecutive_losses"] >= self.thresholds["consecutive_losses"]["critical"]:
            return True
        
        if record["win_rate"] <= self.thresholds["win_rate"]["critical"] and record["total_trades"] >= 10:
            return True
        
        if record["profit_factor"] <= self.thresholds["profit_factor"]["critical"] and record["total_trades"] >= 10:
            return True
        
        return False
    
    def _get_isolation_reason(self, record):
        """
        Get the reason for isolation based on performance metrics.
        
        Parameters:
        - record: Performance record to check
        
        Returns:
        - String describing the isolation reason
        """
        reasons = []
        
        if record["max_drawdown"] >= self.thresholds["drawdown"]["critical"]:
            reasons.append(f"Excessive drawdown ({record['max_drawdown']:.2%})")
        
        if record["consecutive_losses"] >= self.thresholds["consecutive_losses"]["critical"]:
            reasons.append(f"Too many consecutive losses ({record['consecutive_losses']})")
        
        if record["win_rate"] <= self.thresholds["win_rate"]["critical"] and record["total_trades"] >= 10:
            reasons.append(f"Low win rate ({record['win_rate']:.2%})")
        
        if record["profit_factor"] <= self.thresholds["profit_factor"]["critical"] and record["total_trades"] >= 10:
            reasons.append(f"Poor profit factor ({record['profit_factor']:.2f})")
        
        return ", ".join(reasons)
    
    def _check_recovery_conditions(self):
        """
        Check if any isolated component can be recovered.
        
        Returns:
        - Boolean indicating if any component was recovered
        """
        current_time = self.algorithm.Time
        recovered = False
        
        for strategy_name, isolation_info in list(self.isolated_strategies.items()):
            if self._can_recover(isolation_info):
                self.logger.info(f"Attempting to recover strategy {strategy_name}")
                
                self.strategy_performance[strategy_name] = self._create_performance_record()
                
                isolation_info["recovery_attempts"] += 1
                
                if isolation_info["recovery_attempts"] <= 1:
                    self.recovery_log.append({
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "component_type": "strategy",
                        "component_name": strategy_name,
                        "isolation_reason": isolation_info["reason"],
                        "isolation_duration": (current_time - isolation_info["isolation_time"]).total_seconds() / 3600,
                        "attempt_number": isolation_info["recovery_attempts"]
                    })
                    
                    del self.isolated_strategies[strategy_name]
                    recovered = True
        
        for module_name, isolation_info in list(self.isolated_modules.items()):
            if self._can_recover(isolation_info):
                self.logger.info(f"Attempting to recover module {module_name}")
                
                self.module_performance[module_name] = self._create_performance_record()
                
                isolation_info["recovery_attempts"] += 1
                
                if isolation_info["recovery_attempts"] <= 1:
                    self.recovery_log.append({
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "component_type": "module",
                        "component_name": module_name,
                        "isolation_reason": isolation_info["reason"],
                        "isolation_duration": (current_time - isolation_info["isolation_time"]).total_seconds() / 3600,
                        "attempt_number": isolation_info["recovery_attempts"]
                    })
                    
                    del self.isolated_modules[module_name]
                    recovered = True
        
        for symbol_str, isolation_info in list(self.isolated_symbols.items()):
            if self._can_recover(isolation_info):
                self.logger.info(f"Attempting to recover symbol {symbol_str}")
                
                self.symbol_performance[symbol_str] = self._create_performance_record()
                
                isolation_info["recovery_attempts"] += 1
                
                if isolation_info["recovery_attempts"] <= 1:
                    self.recovery_log.append({
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "component_type": "symbol",
                        "component_name": symbol_str,
                        "isolation_reason": isolation_info["reason"],
                        "isolation_duration": (current_time - isolation_info["isolation_time"]).total_seconds() / 3600,
                        "attempt_number": isolation_info["recovery_attempts"]
                    })
                    
                    del self.isolated_symbols[symbol_str]
                    recovered = True
        
        if recovered:
            self._save_logs()
        
        return recovered
    
    def _can_recover(self, isolation_info):
        """
        Determine if an isolated component can be recovered.
        
        Parameters:
        - isolation_info: Dictionary containing isolation information
        
        Returns:
        - Boolean indicating if the component can be recovered
        """
        current_time = self.algorithm.Time
        isolation_time = isolation_info["isolation_time"]
        
        if current_time - isolation_time < self.recovery_conditions["min_idle_time"]:
            return False
        
        if isolation_info["recovery_attempts"] > 0:
            required_idle_time = self.recovery_conditions["min_idle_time"] * (2 ** isolation_info["recovery_attempts"])
            
            if current_time - isolation_time < required_idle_time:
                return False
        
        return True
    
    def is_isolated(self, strategy_name=None, module_name=None, symbol=None):
        """
        Check if a component is currently isolated.
        
        Parameters:
        - strategy_name: Name of the strategy to check
        - module_name: Name of the module to check
        - symbol: Symbol to check
        
        Returns:
        - Boolean indicating if the component is isolated
        """
        if strategy_name and strategy_name in self.isolated_strategies:
            return True
        
        if module_name and module_name in self.isolated_modules:
            return True
        
        if symbol and str(symbol) in self.isolated_symbols:
            return True
        
        return False
    
    def get_isolation_info(self, strategy_name=None, module_name=None, symbol=None):
        """
        Get isolation information for a component.
        
        Parameters:
        - strategy_name: Name of the strategy to check
        - module_name: Name of the module to check
        - symbol: Symbol to check
        
        Returns:
        - Dictionary containing isolation information
        """
        if strategy_name and strategy_name in self.isolated_strategies:
            return self.isolated_strategies[strategy_name]
        
        if module_name and module_name in self.isolated_modules:
            return self.isolated_modules[module_name]
        
        if symbol and str(symbol) in self.isolated_symbols:
            return self.isolated_symbols[str(symbol)]
        
        return None
    
    def get_performance_metrics(self, strategy_name=None, module_name=None, symbol=None):
        """
        Get performance metrics for a component.
        
        Parameters:
        - strategy_name: Name of the strategy to check
        - module_name: Name of the module to check
        - symbol: Symbol to check
        
        Returns:
        - Dictionary containing performance metrics
        """
        metrics = {}
        
        if strategy_name:
            if strategy_name in self.strategy_performance:
                metrics["strategy"] = {k: v for k, v in self.strategy_performance[strategy_name].items() if k != "trade_history"}
        
        if module_name:
            if module_name in self.module_performance:
                metrics["module"] = {k: v for k, v in self.module_performance[module_name].items() if k != "trade_history"}
        
        if symbol:
            symbol_str = str(symbol)
            if symbol_str in self.symbol_performance:
                metrics["symbol"] = {k: v for k, v in self.symbol_performance[symbol_str].items() if k != "trade_history"}
        
        return metrics
    
    def get_isolation_status(self):
        """
        Get the current isolation status for all components.
        
        Returns:
        - Dictionary containing isolation status
        """
        return {
            "isolated_strategies": list(self.isolated_strategies.keys()),
            "isolated_modules": list(self.isolated_modules.keys()),
            "isolated_symbols": list(self.isolated_symbols.keys()),
            "total_isolated": len(self.isolated_strategies) + len(self.isolated_modules) + len(self.isolated_symbols)
        }
    
    def _save_logs(self):
        """
        Save isolation and recovery logs to disk.
        
        Returns:
        - Boolean indicating if save was successful
        """
        try:
            with open(self.isolation_log_path, "w") as f:
                json.dump(self.isolation_log, f, indent=2)
            
            with open(self.recovery_log_path, "w") as f:
                json.dump(self.recovery_log, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving logs: {str(e)}")
            return False
    
    def _load_logs(self):
        """
        Load isolation and recovery logs from disk.
        
        Returns:
        - Boolean indicating if load was successful
        """
        try:
            if os.path.exists(self.isolation_log_path):
                with open(self.isolation_log_path, "r") as f:
                    self.isolation_log = json.load(f)
            
            if os.path.exists(self.recovery_log_path):
                with open(self.recovery_log_path, "r") as f:
                    self.recovery_log = json.load(f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading logs: {str(e)}")
            return False
            
    def record_trade_result(self, symbol, result, trade_data=None):
        """
        Record trade result for a symbol. This is a wrapper around update_performance
        that simplifies the interface for the oversoul integration.
        
        Parameters:
        - symbol: Trading symbol
        - result: 1 for profit, 0 for loss
        - trade_data: Optional dictionary with additional trade data
        
        Returns:
        - Dictionary containing updated performance metrics
        """
        if trade_data is None:
            trade_data = {}
            
        strategy_name = trade_data.get('strategy', 'default_strategy')
        module_name = trade_data.get('module', 'default_module')
        
        trade_result = {
            "is_win": result == 1,
            "profit_loss": trade_data.get('profit_loss', 1.0 if result == 1 else -1.0),
            "entry_time": trade_data.get('entry_time', self.algorithm.Time - timedelta(minutes=15)),
            "exit_time": trade_data.get('exit_time', self.algorithm.Time)
        }
        
        return self.update_performance(strategy_name, module_name, symbol, trade_result)
        
    def check_isolation_criteria(self):
        """
        Check if any components should be isolated or recovered based on performance.
        This method combines _check_self_destruct_conditions and _check_recovery_conditions.
        
        Returns:
        - List of dictionaries containing isolation updates
        """
        updates = []
        
        for strategy_name, record in self.strategy_performance.items():
            if strategy_name not in self.isolated_strategies and self._should_isolate(record):
                reason = self._get_isolation_reason(record)
                updates.append({
                    'action': 'isolate',
                    'target': strategy_name,
                    'target_type': 'strategy',
                    'reason': reason
                })
        
        for module_name, record in self.module_performance.items():
            if module_name not in self.isolated_modules and self._should_isolate(record):
                reason = self._get_isolation_reason(record)
                updates.append({
                    'action': 'isolate',
                    'target': module_name,
                    'target_type': 'module',
                    'reason': reason
                })
        
        for symbol_str, record in self.symbol_performance.items():
            if symbol_str not in self.isolated_symbols and self._should_isolate(record):
                reason = self._get_isolation_reason(record)
                updates.append({
                    'action': 'isolate',
                    'target': symbol_str,
                    'target_type': 'symbol',
                    'reason': reason
                })
        
        for strategy_name, isolation_info in list(self.isolated_strategies.items()):
            if self._can_recover(isolation_info):
                updates.append({
                    'action': 'recover',
                    'target': strategy_name,
                    'target_type': 'strategy',
                    'isolation_duration': (self.algorithm.Time - isolation_info["isolation_time"]).total_seconds() / 3600
                })
        
        for module_name, isolation_info in list(self.isolated_modules.items()):
            if self._can_recover(isolation_info):
                updates.append({
                    'action': 'recover',
                    'target': module_name,
                    'target_type': 'module',
                    'isolation_duration': (self.algorithm.Time - isolation_info["isolation_time"]).total_seconds() / 3600
                })
        
        for symbol_str, isolation_info in list(self.isolated_symbols.items()):
            if self._can_recover(isolation_info):
                updates.append({
                    'action': 'recover',
                    'target': symbol_str,
                    'target_type': 'symbol',
                    'isolation_duration': (self.algorithm.Time - isolation_info["isolation_time"]).total_seconds() / 3600
                })
        
        return updates
