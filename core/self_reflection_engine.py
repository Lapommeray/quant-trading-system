"""
Self-Reflection Engine

Meta-learning loop for trade outcome analysis and continuous improvement.
"""

from AlgorithmImports import *
import logging
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from collections import deque

class MetaLearner:
    """
    Meta-learning loop for trade outcome analysis and continuous improvement.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Meta Learner.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("MetaLearner")
        self.logger.setLevel(logging.INFO)
        
        self.memory = deque(maxlen=10000)
        self.trade_history = []
        
        self.metrics = {
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "sharpe_ratio": 0.0
        }
        
        self.strategy_weights = {}
        
        self.model_path = "/models/meta_strategy.json"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self._load_model()
        
        self.logger.info("Meta Learner initialized")
        
    def process_trade(self, trade):
        """
        Process a completed trade and update the meta-learning model.
        
        Parameters:
        - trade: Dictionary containing trade information
        
        Returns:
        - Updated metrics
        """
        self.logger.info(f"Processing trade: {trade}")
        
        self.memory.append(trade)
        
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": trade.get("symbol", "Unknown"),
            "strategy": trade.get("strategy", "Unknown"),
            "entry_price": trade.get("entry_price", 0.0),
            "exit_price": trade.get("exit_price", 0.0),
            "pnl": trade.get("pnl", 0.0),
            "pnl_pct": trade.get("pnl_pct", 0.0),
            "duration": trade.get("duration", 0),
            "market_state": trade.get("market_state", {})
        })
        
        self._update_metrics()
        
        if len(self.memory) >= 50:
            self._update_strategy_weights()
        
        if len(self.memory) % 100 == 0:
            self._save_model()
            
        return self.metrics
        
    def _update_metrics(self):
        """
        Update performance metrics based on trade history.
        """
        if not self.trade_history:
            return
            
        df = pd.DataFrame(self.trade_history)
        
        if len(df) > 0:
            self.metrics["win_rate"] = len(df[df["pnl"] > 0]) / len(df)
        
        if len(df[df["pnl"] > 0]) > 0:
            self.metrics["avg_profit"] = df[df["pnl"] > 0]["pnl"].mean()
        
        if len(df[df["pnl"] < 0]) > 0:
            self.metrics["avg_loss"] = abs(df[df["pnl"] < 0]["pnl"].mean())
        
        total_profit = df[df["pnl"] > 0]["pnl"].sum()
        total_loss = abs(df[df["pnl"] < 0]["pnl"].sum())
        
        if total_loss > 0:
            self.metrics["profit_factor"] = total_profit / total_loss
        
        if self.metrics["avg_loss"] > 0:
            self.metrics["expectancy"] = (
                self.metrics["win_rate"] * self.metrics["avg_profit"] -
                (1 - self.metrics["win_rate"]) * self.metrics["avg_loss"]
            )
        
        if len(df) > 1:
            returns = df["pnl_pct"].values
            self.metrics["sharpe_ratio"] = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
    def _update_strategy_weights(self):
        """
        Update strategy weights based on performance.
        """
        strategy_performance = {}
        
        for trade in self.trade_history:
            strategy = trade.get("strategy", "Unknown")
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "trades": [],
                    "win_rate": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                    "expectancy": 0.0
                }
                
            strategy_performance[strategy]["trades"].append(trade)
        
        for strategy, data in strategy_performance.items():
            trades = data["trades"]
            
            if not trades:
                continue
                
            wins = sum(1 for t in trades if t["pnl"] > 0)
            data["win_rate"] = wins / len(trades)
            
            profits = [t["pnl"] for t in trades if t["pnl"] > 0]
            losses = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
            
            data["avg_profit"] = np.mean(profits) if profits else 0.0
            data["avg_loss"] = np.mean(losses) if losses else 0.0
            
            total_profit = sum(profits)
            total_loss = sum(losses)
            
            data["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            data["expectancy"] = (
                data["win_rate"] * data["avg_profit"] -
                (1 - data["win_rate"]) * data["avg_loss"]
            )
        
        total_expectancy = sum(max(0.01, data["expectancy"]) for data in strategy_performance.values())
        
        if total_expectancy > 0:
            for strategy, data in strategy_performance.items():
                self.strategy_weights[strategy] = max(0.01, data["expectancy"]) / total_expectancy
        
        self.logger.info(f"Updated strategy weights: {self.strategy_weights}")
        
    def get_strategy_allocation(self, strategy):
        """
        Get allocation weight for a specific strategy.
        
        Parameters:
        - strategy: Strategy name
        
        Returns:
        - Allocation weight (0.0 to 1.0)
        """
        return self.strategy_weights.get(strategy, 0.5)  # Default to 0.5 if unknown
        
    def get_market_state_performance(self, market_state):
        """
        Get performance metrics for similar market states.
        
        Parameters:
        - market_state: Current market state
        
        Returns:
        - Performance metrics for similar market states
        """
        if not self.trade_history:
            return None
            
        similar_trades = []
        
        for trade in self.trade_history:
            trade_market_state = trade.get("market_state", {})
            
            similarity = 0
            total_checks = 0
            
            for key, value in market_state.items():
                if key in trade_market_state:
                    if isinstance(value, (int, float)) and isinstance(trade_market_state[key], (int, float)):
                        if abs(value - trade_market_state[key]) / (abs(value) + 1e-10) < 0.2:
                            similarity += 1
                    elif value == trade_market_state[key]:
                        similarity += 1
                        
                    total_checks += 1
            
            if total_checks > 0 and similarity / total_checks > 0.7:
                similar_trades.append(trade)
        
        if not similar_trades:
            return None
            
        wins = sum(1 for t in similar_trades if t["pnl"] > 0)
        win_rate = wins / len(similar_trades)
        
        profits = [t["pnl"] for t in similar_trades if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in similar_trades if t["pnl"] < 0]
        
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        total_profit = sum(profits)
        total_loss = sum(losses)
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        expectancy = (
            win_rate * avg_profit -
            (1 - win_rate) * avg_loss
        )
        
        return {
            "similar_trades_count": len(similar_trades),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy
        }
        
    def retrain_model(self):
        """
        Retrain the meta-learning model using accumulated trade data.
        
        Returns:
        - Training results
        """
        self.logger.info("Retraining meta-learning model")
        
        if len(self.memory) < 100:
            self.logger.warning("Not enough data to retrain model")
            return False
            
        
        self._update_metrics()
        self._update_strategy_weights()
        self._save_model()
        
        return True
        
    def _save_model(self):
        """
        Save the meta-learning model to disk.
        """
        model_data = {
            "metrics": self.metrics,
            "strategy_weights": self.strategy_weights,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
                
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
        
    def _load_model(self):
        """
        Load the meta-learning model from disk.
        """
        if not os.path.exists(self.model_path):
            self.logger.info("No existing model found")
            return
            
        try:
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
                
            self.metrics = model_data.get("metrics", self.metrics)
            self.strategy_weights = model_data.get("strategy_weights", self.strategy_weights)
            
            self.logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
        
    def get_metrics(self):
        """
        Get current performance metrics.
        
        Returns:
        - Performance metrics
        """
        return self.metrics
        
    def get_trade_history(self):
        """
        Get trade history.
        
        Returns:
        - Trade history
        """
        return self.trade_history
        
    def get_strategy_weights(self):
        """
        Get strategy weights.
        
        Returns:
        - Strategy weights
        """
        return self.strategy_weights
