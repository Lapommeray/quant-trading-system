"""
Self-Salvaging Intelligence

Detects and replaces weak strategies via flaw healing for the QMP Overrider system.
"""

from AlgorithmImports import *
import logging
import os
import json
import shutil
from datetime import datetime
import numpy as np

class StrategySurgeon:
    """
    Detects and replaces weak strategies via flaw healing.
    """
    
    def __init__(self, algorithm, strategy_generator=None):
        """
        Initialize the Strategy Surgeon.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        - strategy_generator: Strategy generator instance (optional)
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("StrategySurgeon")
        self.logger.setLevel(logging.INFO)
        
        self.strategy_generator = strategy_generator
        
        self.weakness_threshold = -0.15  # 15% drawdown
        self.sharpe_threshold = 0.5  # Minimum acceptable Sharpe ratio
        self.win_rate_threshold = 0.4  # Minimum acceptable win rate
        
        self.strategy_performance = {}
        
        self.surgery_history = []
        
        self.quarantine_dir = "/strategies/quarantine"
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        self.logger.info("Strategy Surgeon initialized")
        
    def perform_surgery(self):
        """
        Identify and fix weak strategies.
        
        Returns:
        - Dictionary of surgery results
        """
        self.logger.info("Performing strategy surgery")
        
        weak_strategies = self._identify_weak_strategies()
        
        if not weak_strategies:
            self.logger.info("No weak strategies found")
            return {"status": "success", "weak_strategies": 0, "fixed_strategies": 0}
            
        self.logger.info(f"Found {len(weak_strategies)} weak strategies")
        
        fixed_strategies = []
        
        for strategy in weak_strategies:
            fixed = self._fix_strategy(strategy)
            
            if fixed:
                fixed_strategies.append(fixed)
        
        surgery_results = {
            "status": "success",
            "weak_strategies": len(weak_strategies),
            "fixed_strategies": len(fixed_strategies),
            "details": {
                "weak": [s["name"] for s in weak_strategies],
                "fixed": [s["name"] for s in fixed_strategies]
            }
        }
        
        self.logger.info(f"Surgery results: {surgery_results}")
        
        return surgery_results
        
    def _identify_weak_strategies(self):
        """
        Identify weak strategies based on performance metrics.
        
        Returns:
        - List of weak strategies
        """
        weak_strategies = []
        
        for name, performance in self.strategy_performance.items():
            if performance.get("trade_count", 0) < 10:
                continue
                
            is_weak = False
            flaws = []
            
            if performance.get("max_drawdown", 0) < self.weakness_threshold:
                is_weak = True
                flaws.append(f"High drawdown: {performance.get('max_drawdown', 0):.2%}")
                
            if performance.get("sharpe_ratio", 0) < self.sharpe_threshold:
                is_weak = True
                flaws.append(f"Low Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
                
            if performance.get("win_rate", 0) < self.win_rate_threshold:
                is_weak = True
                flaws.append(f"Low win rate: {performance.get('win_rate', 0):.2%}")
                
            if performance.get("profit_factor", 0) < 1.0:
                is_weak = True
                flaws.append(f"Profit factor below 1.0: {performance.get('profit_factor', 0):.2f}")
                
            if is_weak:
                weak_strategies.append({
                    "name": name,
                    "path": performance.get("path", ""),
                    "flaws": flaws,
                    "performance": performance
                })
        
        return weak_strategies
        
    def _fix_strategy(self, strategy):
        """
        Fix a weak strategy.
        
        Parameters:
        - strategy: Dictionary containing strategy information
        
        Returns:
        - Dictionary containing fixed strategy information or None if fix failed
        """
        name = strategy["name"]
        path = strategy["path"]
        flaws = strategy["flaws"]
        
        self.logger.info(f"Fixing strategy: {name}")
        self.logger.info(f"Flaws: {flaws}")
        
        self._quarantine_strategy(path)
        
        if not self.strategy_generator:
            self.logger.warning(f"No strategy generator available, strategy {name} removed")
            
            self.surgery_history.append({
                "timestamp": datetime.now().isoformat(),
                "strategy": name,
                "action": "remove",
                "flaws": flaws,
                "result": "removed"
            })
            
            return None
            
        try:
            market_state = {
                "fix_strategy": name,
                "flaws": flaws,
                "performance": strategy["performance"],
                "anomaly": "Strategy weakness",
                "volatility": "Unknown",
                "liquidity": "Unknown"
            }
            
            new_path = self.strategy_generator.generate_new_logic(market_state)
            
            if not new_path:
                self.logger.error(f"Failed to generate new strategy for {name}")
                return None
                
            self.surgery_history.append({
                "timestamp": datetime.now().isoformat(),
                "strategy": name,
                "action": "regenerate",
                "flaws": flaws,
                "result": "success",
                "new_path": new_path
            })
            
            return {
                "name": f"{name}_fixed",
                "path": new_path,
                "original": name
            }
            
        except Exception as e:
            self.logger.error(f"Error fixing strategy {name}: {str(e)}")
            
            self.surgery_history.append({
                "timestamp": datetime.now().isoformat(),
                "strategy": name,
                "action": "regenerate",
                "flaws": flaws,
                "result": "error",
                "error": str(e)
            })
            
            return None
        
    def _quarantine_strategy(self, strategy_path):
        """
        Move a weak strategy to quarantine.
        
        Parameters:
        - strategy_path: Path to strategy file
        
        Returns:
        - Boolean indicating if quarantine was successful
        """
        if not os.path.exists(strategy_path):
            self.logger.warning(f"Strategy file not found: {strategy_path}")
            return False
            
        try:
            filename = os.path.basename(strategy_path)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            quarantine_filename = f"{timestamp}_{filename}"
            quarantine_path = os.path.join(self.quarantine_dir, quarantine_filename)
            
            shutil.copy2(strategy_path, quarantine_path)
            
            self.logger.info(f"Strategy quarantined: {quarantine_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error quarantining strategy: {str(e)}")
            return False
        
    def update_strategy_performance(self, strategy_name, performance_data):
        """
        Update performance data for a strategy.
        
        Parameters:
        - strategy_name: Name of the strategy
        - performance_data: Dictionary containing performance metrics
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "path": performance_data.get("path", "")
            }
            
        for key, value in performance_data.items():
            if key in self.strategy_performance[strategy_name]:
                self.strategy_performance[strategy_name][key] = value
        
        perf = self.strategy_performance[strategy_name]
        
        if perf["trade_count"] > 0:
            perf["win_rate"] = perf["win_count"] / perf["trade_count"]
            
        if perf["total_loss"] != 0:
            perf["profit_factor"] = perf["total_profit"] / abs(perf["total_loss"])
            
        if perf["trade_count"] > 0:
            avg_win = perf["total_profit"] / perf["win_count"] if perf["win_count"] > 0 else 0
            avg_loss = abs(perf["total_loss"]) / perf["loss_count"] if perf["loss_count"] > 0 else 0
            perf["expectancy"] = (perf["win_rate"] * avg_win) - ((1 - perf["win_rate"]) * avg_loss)
        
    def get_strategy_performance(self, strategy_name=None):
        """
        Get performance data for strategies.
        
        Parameters:
        - strategy_name: Name of specific strategy (optional)
        
        Returns:
        - Strategy performance data
        """
        if strategy_name:
            return self.strategy_performance.get(strategy_name)
        else:
            return self.strategy_performance
        
    def get_surgery_history(self):
        """
        Get surgery history.
        
        Returns:
        - Surgery history
        """
        return self.surgery_history
        
    def set_strategy_generator(self, strategy_generator):
        """
        Set strategy generator.
        
        Parameters:
        - strategy_generator: Strategy generator instance
        """
        self.strategy_generator = strategy_generator
        
    def set_thresholds(self, weakness_threshold=None, sharpe_threshold=None, win_rate_threshold=None):
        """
        Set thresholds for identifying weak strategies.
        
        Parameters:
        - weakness_threshold: Drawdown threshold
        - sharpe_threshold: Sharpe ratio threshold
        - win_rate_threshold: Win rate threshold
        """
        if weakness_threshold is not None:
            self.weakness_threshold = weakness_threshold
            
        if sharpe_threshold is not None:
            self.sharpe_threshold = sharpe_threshold
            
        if win_rate_threshold is not None:
            self.win_rate_threshold = win_rate_threshold
