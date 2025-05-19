"""
Win Rate Auditing Script
Analyzes trade data to verify realistic win rates and performance metrics
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WinRateAuditor:
    def __init__(self, min_win_rate=0.50, max_win_rate=0.80):
        """
        Initialize the win rate auditor
        
        Parameters:
        - min_win_rate: Minimum acceptable win rate (default: 0.50)
        - max_win_rate: Maximum acceptable win rate (default: 0.80)
        """
        self.min_win_rate = min_win_rate
        self.max_win_rate = max_win_rate
        self.results = {}
    
    def audit_trades(self, file):
        """
        Audit trades from a CSV file
        
        Parameters:
        - file: Path to CSV file with trade data
        
        Returns:
        - Dictionary with audit results
        """
        print(f"Auditing trades from {file}")
        
        try:
            trades = pd.read_csv(file)
        except Exception as e:
            print(f"Error loading trades file: {e}")
            return {'error': str(e)}
        
        required_columns = ['timestamp', 'symbol', 'direction', 'fill_price']
        missing_columns = [col for col in required_columns if col not in trades.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return {'error': f"Missing required columns: {missing_columns}"}
        
        if 'pnl' not in trades.columns:
            print("PnL column not found. Calculating PnL from trade data...")
            trades = self._calculate_pnl(trades)
        
        trades_with_pnl = trades[trades['pnl'].notna() & (trades['pnl'] != 0)].copy()
        
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            completed_trades = trades_with_pnl[trades_with_pnl['entry_price'].notna() & 
                                              trades_with_pnl['exit_price'].notna()].copy()
            if len(completed_trades) > 0:
                trades_with_pnl = completed_trades
                print("Using completed trades with entry and exit prices")
        
        if len(trades_with_pnl) == 0:
            print("Warning: No trades with PnL values found. Using all trades.")
            trades_with_pnl = trades.copy()
        
        winning_trades = trades_with_pnl[trades_with_pnl['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_with_pnl) if len(trades_with_pnl) > 0 else 0
        
        print(f"Total trades: {len(trades)}")
        print(f"Trades with PnL values: {len(trades_with_pnl)}")
        print(f"Winning trades: {len(winning_trades)}")
        print(f"Win rate: {win_rate:.2%}")
        
        is_realistic = self.min_win_rate <= win_rate <= self.max_win_rate
        
        total_pnl = trades['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades[trades['pnl'] < 0]
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        symbols = trades['symbol'].unique()
        symbol_stats = {}
        
        for symbol in symbols:
            symbol_trades = trades[trades['symbol'] == symbol]
            symbol_wins = symbol_trades[symbol_trades['pnl'] > 0]
            symbol_win_rate = len(symbol_wins) / len(symbol_trades)
            symbol_is_realistic = self.min_win_rate <= symbol_win_rate <= self.max_win_rate
            
            symbol_stats[symbol] = {
                'trades': len(symbol_trades),
                'wins': len(symbol_wins),
                'win_rate': symbol_win_rate,
                'is_realistic': symbol_is_realistic,
                'total_pnl': symbol_trades['pnl'].sum(),
                'avg_win': symbol_wins['pnl'].mean() if len(symbol_wins) > 0 else 0,
                'avg_loss': symbol_trades[symbol_trades['pnl'] < 0]['pnl'].mean() if len(symbol_trades[symbol_trades['pnl'] < 0]) > 0 else 0
            }
        
        if 'timestamp' in trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            trades['week'] = trades['timestamp'].dt.isocalendar().week
            trades['year'] = trades['timestamp'].dt.isocalendar().year
            trades['year_week'] = trades['year'].astype(str) + '-' + trades['week'].astype(str)
            
            weekly_stats = {}
            for year_week in trades['year_week'].unique():
                week_trades = trades[trades['year_week'] == year_week]
                week_wins = week_trades[week_trades['pnl'] > 0]
                week_win_rate = len(week_wins) / len(week_trades)
                week_is_realistic = self.min_win_rate <= week_win_rate <= self.max_win_rate
                
                weekly_stats[year_week] = {
                    'trades': len(week_trades),
                    'wins': len(week_wins),
                    'win_rate': week_win_rate,
                    'is_realistic': week_is_realistic,
                    'total_pnl': week_trades['pnl'].sum()
                }
        else:
            weekly_stats = {'error': 'No timestamp column found'}
        
        self.results = {
            'file': file,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'is_realistic': is_realistic,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'symbol_stats': symbol_stats,
            'weekly_stats': weekly_stats,
            'min_win_rate': self.min_win_rate,
            'max_win_rate': self.max_win_rate
        }
        
        try:
            self.assert_realistic_win_rate()
            self.results['assertion_passed'] = True
        except AssertionError as e:
            print(f"❌ {str(e)}")
            self.results['assertion_passed'] = False
            self.results['assertion_error'] = str(e)
        
        return self.results
    
    def _calculate_pnl(self, trades):
        """Calculate PnL from trade data"""
        
        if 'timestamp' in trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            trades = trades.sort_values('timestamp')
        
        trades['pnl'] = 0.0
        
        symbols = trades['symbol'].unique()
        
        for symbol in symbols:
            symbol_trades = trades[trades['symbol'] == symbol].copy()
            
            position = 0
            entry_price = 0
            
            for i, trade in symbol_trades.iterrows():
                direction = trade['direction']
                price = trade['fill_price']
                
                if direction == 'BUY':
                    if position < 0:  # Close short position
                        pnl = (entry_price - price) * abs(position)
                        trades.loc[i, 'pnl'] = pnl
                        position = 0
                    else:
                        position = 1
                        entry_price = price
                
                elif direction == 'SELL':
                    if position > 0:  # Close long position
                        pnl = (price - entry_price) * position
                        trades.loc[i, 'pnl'] = pnl
                        position = 0
                    else:
                        position = -1
                        entry_price = price
        
        return trades
    
    def assert_realistic_win_rate(self):
        """Assert that win rate is realistic"""
        if not self.results:
            raise ValueError("No audit results available")
        
        win_rate = self.results['win_rate']
        
        if not (self.min_win_rate <= win_rate <= self.max_win_rate):
            raise AssertionError(
                f"Unrealistic win rate detected: {win_rate:.2%} "
                f"(acceptable range: {self.min_win_rate:.0%}-{self.max_win_rate:.0%})"
            )
        
        return True
    
    def generate_report(self, output_file="audit_report.json"):
        """Generate audit report and save to file"""
        if not self.results:
            print("No audit results available")
            return False
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Audit report saved to {output_file}")
        return True
    
    def plot_results(self, output_file="audit_chart.png"):
        """Plot audit results"""
        if not self.results:
            print("No audit results available")
            return False
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(['Win Rate'], [self.results['win_rate']], color='g' if self.results['is_realistic'] else 'r')
        plt.axhline(y=self.min_win_rate, color='r', linestyle='--', label=f"Min: {self.min_win_rate:.0%}")
        plt.axhline(y=self.max_win_rate, color='r', linestyle='--', label=f"Max: {self.max_win_rate:.0%}")
        plt.ylim(0, 1)
        plt.title("Overall Win Rate")
        plt.ylabel("Win Rate")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        symbols = list(self.results['symbol_stats'].keys())
        win_rates = [self.results['symbol_stats'][s]['win_rate'] for s in symbols]
        colors = ['g' if self.min_win_rate <= wr <= self.max_win_rate else 'r' for wr in win_rates]
        
        plt.bar(symbols, win_rates, color=colors)
        plt.axhline(y=self.min_win_rate, color='r', linestyle='--')
        plt.axhline(y=self.max_win_rate, color='r', linestyle='--')
        plt.ylim(0, 1)
        plt.title("Win Rate by Symbol")
        plt.ylabel("Win Rate")
        plt.xticks(rotation=45)
        
        if isinstance(self.results['weekly_stats'], dict) and 'error' not in self.results['weekly_stats']:
            plt.subplot(2, 1, 2)
            weeks = list(self.results['weekly_stats'].keys())
            weekly_win_rates = [self.results['weekly_stats'][w]['win_rate'] for w in weeks]
            weekly_colors = ['g' if self.min_win_rate <= wr <= self.max_win_rate else 'r' for wr in weekly_win_rates]
            
            plt.bar(weeks, weekly_win_rates, color=weekly_colors)
            plt.axhline(y=self.min_win_rate, color='r', linestyle='--')
            plt.axhline(y=self.max_win_rate, color='r', linestyle='--')
            plt.ylim(0, 1)
            plt.title("Win Rate by Week")
            plt.ylabel("Win Rate")
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Audit chart saved to {output_file}")
        return True


def audit_trades(file, min_win_rate=0.50, max_win_rate=0.80):
    """
    Audit trades from a CSV file
    
    Parameters:
    - file: Path to CSV file with trade data
    - min_win_rate: Minimum acceptable win rate (default: 0.50)
    - max_win_rate: Maximum acceptable win rate (default: 0.80)
    
    Returns:
    - Dictionary with audit results
    """
    auditor = WinRateAuditor(min_win_rate, max_win_rate)
    results = auditor.audit_trades(file)
    
    auditor.generate_report()
    auditor.plot_results()
    
    print("\n" + "="*50)
    print("Win Rate Audit Results")
    print("="*50)
    print(f"File: {file}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Acceptable Range: {min_win_rate:.0%}-{max_win_rate:.0%}")
    print(f"Status: {'✅ PASSED' if results['is_realistic'] else '❌ FAILED'}")
    print(f"Total PnL: {results['total_pnl']:.2f}")
    print(f"Average Win: {results['avg_win']:.2f}")
    print(f"Average Loss: {results['avg_loss']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    print("\nSymbol Statistics:")
    for symbol, stats in results['symbol_stats'].items():
        print(f"  {symbol}: {stats['win_rate']:.2%} win rate over {stats['trades']} trades - "
              f"{'✅ REALISTIC' if stats['is_realistic'] else '❌ UNREALISTIC'}")
    
    print("\nOutput Files:")
    print("  Audit Report: audit_report.json")
    print("  Audit Chart: audit_chart.png")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audit trade win rates')
    parser.add_argument('file', type=str, help='Path to CSV file with trade data')
    parser.add_argument('--min', type=float, default=0.50,
                        help='Minimum acceptable win rate (default: 0.50)')
    parser.add_argument('--max', type=float, default=0.80,
                        help='Maximum acceptable win rate (default: 0.80)')
    
    args = parser.parse_args()
    
    audit_trades(args.file, args.min, args.max)
