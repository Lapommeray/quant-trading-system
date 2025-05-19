"""
Fundamental Equation of Alpha
Implements the equation: Profit = (Edge Frequency × Edge Size) - (Error Frequency × Error Cost)
"""

import numpy as np
import pandas as pd
from datetime import datetime

class AlphaEquation:
    def __init__(self):
        """Initialize Alpha Equation calculator"""
        pass
    
    def calculate_alpha(self, trades_df):
        """
        Calculate alpha metrics from trade data
        
        Parameters:
        - trades_df: DataFrame with trade data (must have 'pnl' column)
        
        Returns:
        - Dictionary with alpha metrics
        """
        if 'pnl' not in trades_df.columns:
            raise ValueError("Trades DataFrame must have 'pnl' column")
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        edge_frequency = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        error_frequency = len(losing_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        edge_size = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        
        error_cost = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        expected_profit = (edge_frequency * edge_size) - (error_frequency * error_cost)
        
        profit_factor = (edge_frequency * edge_size) / (error_frequency * error_cost) if (error_frequency * error_cost) > 0 else float('inf')
        
        trades_per_day = len(trades_df) / 250  # Simplified assumption
        annualized_return = ((1 + expected_profit) ** (trades_per_day * 250)) - 1
        
        return {
            'edge_frequency': edge_frequency,
            'edge_size': edge_size,
            'error_frequency': error_frequency,
            'error_cost': error_cost,
            'expected_profit': expected_profit,
            'profit_factor': profit_factor,
            'annualized_return': annualized_return,
            'alpha_equation': f"({edge_frequency:.2f} * {edge_size:.4f}) - ({error_frequency:.2f} * {error_cost:.4f}) = {expected_profit:.4f}"
        }
    
    def calculate_alpha_by_symbol(self, trades_df):
        """
        Calculate alpha metrics grouped by symbol
        
        Parameters:
        - trades_df: DataFrame with trade data (must have 'symbol' and 'pnl' columns)
        
        Returns:
        - Dictionary with alpha metrics by symbol
        """
        if 'symbol' not in trades_df.columns or 'pnl' not in trades_df.columns:
            raise ValueError("Trades DataFrame must have 'symbol' and 'pnl' columns")
        
        symbols = trades_df['symbol'].unique()
        results = {}
        
        for symbol in symbols:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            results[symbol] = self.calculate_alpha(symbol_trades)
        
        return results
    
    def calculate_alpha_by_period(self, trades_df, period='month'):
        """
        Calculate alpha metrics grouped by time period
        
        Parameters:
        - trades_df: DataFrame with trade data (must have 'timestamp' and 'pnl' columns)
        - period: Time period for grouping ('day', 'week', 'month', 'quarter', 'year')
        
        Returns:
        - Dictionary with alpha metrics by period
        """
        if 'timestamp' not in trades_df.columns or 'pnl' not in trades_df.columns:
            raise ValueError("Trades DataFrame must have 'timestamp' and 'pnl' columns")
        
        if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
            trades_df = trades_df.copy()
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        if period == 'day':
            trades_df['period'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d')
        elif period == 'week':
            trades_df['period'] = trades_df['timestamp'].dt.strftime('%Y-%U')
        elif period == 'month':
            trades_df['period'] = trades_df['timestamp'].dt.strftime('%Y-%m')
        elif period == 'quarter':
            trades_df['period'] = trades_df['timestamp'].dt.year.astype(str) + '-Q' + (trades_df['timestamp'].dt.quarter).astype(str)
        elif period == 'year':
            trades_df['period'] = trades_df['timestamp'].dt.year
        else:
            raise ValueError("Invalid period. Choose from 'day', 'week', 'month', 'quarter', 'year'")
        
        periods = trades_df['period'].unique()
        results = {}
        
        for p in periods:
            period_trades = trades_df[trades_df['period'] == p]
            results[p] = self.calculate_alpha(period_trades)
        
        return results
    
    def optimize_position_sizing(self, trades_df, risk_free_rate=0.02):
        """
        Optimize position sizing based on alpha metrics
        
        Parameters:
        - trades_df: DataFrame with trade data
        - risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
        
        Returns:
        - Dictionary with optimal position sizing recommendations
        """
        alpha_metrics = self.calculate_alpha(trades_df)
        
        
        p = alpha_metrics['edge_frequency']
        q = alpha_metrics['error_frequency']
        
        if alpha_metrics['error_cost'] > 0:
            b = alpha_metrics['edge_size'] / alpha_metrics['error_cost']
            kelly_fraction = (p*b - q) / b
        else:
            kelly_fraction = 1.0  # No losses, but cap at 100%
        
        kelly_fraction = max(0, min(1, kelly_fraction))
        
        conservative_sizing = kelly_fraction / 2
        
        daily_returns = trades_df.groupby(pd.to_datetime(trades_df['timestamp']).dt.date)['pnl'].sum()
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        
        target_volatility = 0.15  # 15% annual volatility target
        volatility_sizing = target_volatility / (daily_returns.std() * np.sqrt(252))
        
        return {
            'kelly_fraction': kelly_fraction,
            'half_kelly': kelly_fraction / 2,
            'conservative_sizing': conservative_sizing,
            'sharpe_ratio': sharpe_ratio,
            'volatility_sizing': volatility_sizing,
            'recommended_sizing': min(conservative_sizing, volatility_sizing),
            'alpha_metrics': alpha_metrics
        }
    
    def generate_alpha_report(self, trades_df, output_file="alpha_report.json"):
        """
        Generate comprehensive alpha report
        
        Parameters:
        - trades_df: DataFrame with trade data
        - output_file: Output file path for the report
        
        Returns:
        - Dictionary with alpha report
        """
        import json
        
        overall_alpha = self.calculate_alpha(trades_df)
        
        try:
            alpha_by_symbol = self.calculate_alpha_by_symbol(trades_df)
        except ValueError:
            alpha_by_symbol = {"error": "Missing required columns"}
        
        try:
            alpha_by_month = self.calculate_alpha_by_period(trades_df, period='month')
        except ValueError:
            alpha_by_month = {"error": "Missing required columns"}
        
        try:
            position_sizing = self.optimize_position_sizing(trades_df)
        except Exception as e:
            position_sizing = {"error": str(e)}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(trades_df),
            "overall_alpha": overall_alpha,
            "alpha_by_symbol": alpha_by_symbol,
            "alpha_by_month": alpha_by_month,
            "position_sizing": position_sizing
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Alpha report saved to {output_file}")
        
        return report
