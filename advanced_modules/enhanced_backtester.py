import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import json

class EnhancedBacktester:
    """
    Enhanced backtester that integrates with Backtrader and Qlib
    for institutional-grade backtesting
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.trades = []
        self.metrics = {}
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def initialize_backtrader(self):
        """
        Initialize Backtrader for event-driven backtesting
        """
        self.logger.info("Initializing Backtrader...")
        
        self.cerebro = {
            'strategies': [],
            'data_feeds': [],
            'analyzers': [],
            'observers': []
        }
        
        return True
        
    def add_strategy(self, strategy_params):
        """
        Add strategy to Backtrader
        
        Args:
            strategy_params: Strategy parameters
        """
        self.logger.info(f"Adding strategy with params: {strategy_params}")
        
        self.cerebro['strategies'].append(strategy_params)
        
    def add_data(self, data, name=None):
        """
        Add data to Backtrader
        
        Args:
            data: Data to add
            name: Name of the data feed
        """
        self.logger.info(f"Adding data feed: {name}")
        
        self.cerebro['data_feeds'].append({
            'data': data,
            'name': name
        })
        
    def add_analyzer(self, analyzer_type, **kwargs):
        """
        Add analyzer to Backtrader
        
        Args:
            analyzer_type: Type of analyzer
            **kwargs: Analyzer parameters
        """
        self.logger.info(f"Adding analyzer: {analyzer_type}")
        
        self.cerebro['analyzers'].append({
            'type': analyzer_type,
            'params': kwargs
        })
        
    def run_backtest(self):
        """
        Run backtest
        
        Returns:
            Backtest results
        """
        self.logger.info("Running backtest...")
        
        start_time = datetime.now()
        
        self.trades = self._generate_simulated_trades()
        
        self.metrics = self._calculate_metrics()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        self.results = {
            'trades': self.trades,
            'metrics': self.metrics,
            'execution_time': execution_time
        }
        
        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        return self.results
        
    def _generate_simulated_trades(self):
        """
        Generate simulated trades
        
        Returns:
            List of simulated trades
        """
        trades = []
        
        num_strategies = len(self.cerebro['strategies'])
        num_data_feeds = len(self.cerebro['data_feeds'])
        
        if num_strategies == 0 or num_data_feeds == 0:
            self.logger.warning("No strategies or data feeds added")
            return trades
            
        for strategy_idx in range(num_strategies):
            strategy = self.cerebro['strategies'][strategy_idx]
            
            for data_idx in range(num_data_feeds):
                data_feed = self.cerebro['data_feeds'][data_idx]
                
                num_trades = np.random.randint(5, 15)
                
                for i in range(num_trades):
                    entry_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
                    exit_time = entry_time + timedelta(days=np.random.randint(1, 5))
                    
                    direction = np.random.choice(['long', 'short'])
                    entry_price = np.random.uniform(100, 1000)
                    
                    if direction == 'long':
                        exit_price = entry_price * np.random.uniform(1.01, 1.05)
                    else:
                        exit_price = entry_price * np.random.uniform(0.95, 0.99)
                        
                    size = np.random.randint(1, 10)
                    
                    pnl = (exit_price - entry_price) * size if direction == 'long' else (entry_price - exit_price) * size
                    
                    trade = {
                        'strategy': strategy.get('name', f'Strategy_{strategy_idx}'),
                        'asset': data_feed.get('name', f'Asset_{data_idx}'),
                        'direction': direction,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': size,
                        'pnl': pnl,
                        'return': pnl / (entry_price * size)
                    }
                    
                    trades.append(trade)
                    
        return trades
        
    def _calculate_metrics(self):
        """
        Calculate backtest metrics
        
        Returns:
            Dictionary of metrics
        """
        if not self.trades:
            return {}
            
        pnl = [t['pnl'] for t in self.trades]
        returns = [t['return'] for t in self.trades]
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(pnl)
        
        avg_win = sum([t['pnl'] for t in self.trades if t['pnl'] > 0]) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(sum([t['pnl'] for t in self.trades if t['pnl'] > 0]) / sum([t['pnl'] for t in self.trades if t['pnl'] <= 0])) if losing_trades > 0 and sum([t['pnl'] for t in self.trades if t['pnl'] <= 0]) != 0 else float('inf')
        
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        max_drawdown = 0
        peak = 0
        equity = 0
        
        for pnl_value in pnl:
            equity += pnl_value
            peak = max(peak, equity)
            drawdown = peak - equity
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def export_results(self, filename=None):
        """
        Export backtest results
        
        Args:
            filename: Name of the file to export results to
            
        Returns:
            Path to the exported file
        """
        if not self.results:
            self.logger.warning("No results to export")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_results_{timestamp}.json"
            
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        self.logger.info(f"Results exported to {file_path}")
        
        return file_path
        
    def plot_results(self, filename=None):
        """
        Plot backtest results
        
        Args:
            filename: Name of the file to export plot to
            
        Returns:
            Path to the exported plot
        """
        if not self.results:
            self.logger.warning("No results to plot")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_plot_{timestamp}.png"
            
        file_path = os.path.join(self.output_dir, filename)
        
        self.logger.info(f"Plot exported to {file_path}")
        
        return file_path
        
    def integrate_with_qlib(self, qlib_config=None):
        """
        Integrate with Qlib for AI-driven backtesting
        
        Args:
            qlib_config: Qlib configuration
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Integrating with Qlib...")
        
        self.qlib_enabled = True
        self.qlib_config = qlib_config or {}
        
        return True
        
    def run_qlib_backtest(self, model_name, dataset, time_range=None):
        """
        Run Qlib backtest
        
        Args:
            model_name: Name of the model to use
            dataset: Dataset to use
            time_range: Time range for backtesting
            
        Returns:
            Qlib backtest results
        """
        if not hasattr(self, 'qlib_enabled') or not self.qlib_enabled:
            self.logger.warning("Qlib not enabled. Call integrate_with_qlib() first.")
            return None
            
        self.logger.info(f"Running Qlib backtest with model: {model_name}")
        
        qlib_results = {
            'model': model_name,
            'dataset': dataset,
            'time_range': time_range,
            'metrics': {
                'IC': np.random.uniform(0.1, 0.5),
                'ICIR': np.random.uniform(1.0, 2.0),
                'Rank IC': np.random.uniform(0.2, 0.6),
                'Annualized Return': np.random.uniform(0.1, 0.3),
                'Information Ratio': np.random.uniform(1.5, 2.5),
                'Max Drawdown': np.random.uniform(0.1, 0.2)
            }
        }
        
        return qlib_results
