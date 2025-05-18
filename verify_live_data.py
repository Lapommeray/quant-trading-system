#!/usr/bin/env python3
"""
Real-World Verification Command
Tests the trading system with real market data and realistic constraints
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.fill_engine import FillEngine
    from tests.stress_loss_recovery import MarketStressTest
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


class LiveDataVerifier:
    def __init__(self, asset, start_date, end_date, slippage=True, drawdown_check=True):
        """
        Initialize the live data verifier
        
        Parameters:
        - asset: Asset symbol to test (e.g., 'XAU/USD')
        - start_date: Start date for testing (YYYY-MM-DD)
        - end_date: End date for testing (YYYY-MM-DD)
        - slippage: Whether to enable slippage simulation
        - drawdown_check: Whether to check for drawdowns
        """
        self.asset = asset.replace('/', '')  # Remove slash for file handling
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.slippage = slippage
        self.drawdown_check = drawdown_check
        
        self.fill_engine = FillEngine(slippage_enabled=slippage)
        self.stress_test = MarketStressTest(max_drawdown_threshold=0.05)
        
        self.results = {
            'asset': asset,
            'start_date': start_date,
            'end_date': end_date,
            'slippage_enabled': slippage,
            'drawdown_check': drawdown_check,
            'trades': [],
            'performance': {},
            'drawdowns': {}
        }
    
    def load_data(self, data_path=None):
        """
        Load historical data for the specified asset and date range
        
        Parameters:
        - data_path: Optional path to data file
        
        Returns:
        - DataFrame with historical data
        """
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            if 'timestamp' in data.columns:
                data = data[(data['timestamp'] >= self.start_date) & 
                           (data['timestamp'] <= self.end_date)]
            
            return data
        
        print(f"No data file provided. Generating synthetic data for {self.asset}")
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='1D')
        
        if 'XAU' in self.asset:
            base_price = 1800.0  # Gold around $1800/oz
        elif 'BTC' in self.asset:
            base_price = 40000.0  # Bitcoin around $40,000
        elif 'ETH' in self.asset:
            base_price = 2500.0   # Ethereum around $2,500
        else:
            base_price = 100.0    # Default
        
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0002, 0.015, len(date_range))  # Mean slightly positive
        price_series = base_price * np.cumprod(1 + returns)
        
        data = pd.DataFrame({
            'timestamp': date_range,
            'open': price_series,
            'high': price_series * np.random.uniform(1.0, 1.02, len(date_range)),
            'low': price_series * np.random.uniform(0.98, 1.0, len(date_range)),
            'close': price_series,
            'volume': np.random.uniform(1000, 5000, len(date_range))
        })
        
        return data
    
    def run_verification(self, data, trading_system=None):
        """
        Run verification on the provided data
        
        Parameters:
        - data: DataFrame with historical data
        - trading_system: Optional trading system to test
        
        Returns:
        - Dictionary with verification results
        """
        print(f"Running verification for {self.asset} from {self.start_date.date()} to {self.end_date.date()}")
        
        if trading_system is None:
            try:
                from core.qmp_engine import QMPOversoulEngine
                class TradingSystemAdapter:
                    def __init__(self):
                        class MockAlgorithm:
                            def Debug(self, message):
                                print(message)
                        
                        self.engine = QMPOversoulEngine(MockAlgorithm())
                    
                    def process_bar(self, bar):
                        symbol = self.asset
                        
                        result = self.engine.generate_signal(
                            symbol=symbol,
                            current_price=bar['close'],
                            history_bars=[bar]  # May need to accumulate bars
                        )
                        
                        if result and result[0]:
                            return {
                                'direction': result[0],
                                'price': bar['close'],
                                'confidence': result[1],
                                'size': 1.0  # Default position size
                            }
                        
                        return None
                
                trading_system = TradingSystemAdapter()
                print("Using real trading system")
            except ImportError:
                print("Warning: Could not import real trading system. Using mock system.")
                class MockTradingSystem:
                    def __init__(self):
                        self.asset = None  # Will be set after initialization
                        self.last_signal = None
                        self.signal_counter = 0
                    
                    def process_bar(self, bar):
                        if self.signal_counter % 20 == 0:  # Signal every 20 bars
                            if bar['close'] > bar['open']:
                                direction = 'BUY'
                            else:
                                direction = 'SELL'
                            
                            self.last_signal = {
                                'direction': direction,
                                'price': bar['close'],
                                'confidence': 0.7,
                                'size': 1.0
                            }
                        
                        self.signal_counter += 1
                        return self.last_signal
                
                trading_system = MockTradingSystem()
                trading_system.asset = self.asset  # Set the asset from the verifier
                print("Using mock trading system")
        
        initial_capital = 10000.0
        portfolio_value = initial_capital
        position = 0
        trades = []
        portfolio_values = [portfolio_value]
        
        print("Processing market data...")
        for i in tqdm(range(len(data))):
            bar = data.iloc[i]
            
            signal = trading_system.process_bar(bar)
            
            if signal:
                direction = signal['direction']
                raw_price = signal['price']
                confidence = signal.get('confidence', 0.5)
                size = signal.get('size', 1.0)
                
                volume = portfolio_value * 0.1 * confidence * size / raw_price
                
                if self.slippage:
                    fill = self.fill_engine.execute_order(
                        symbol=self.asset,
                        direction=direction,
                        price=raw_price,
                        volume=volume,
                        timestamp=bar['timestamp']
                    )
                    fill_price = fill['fill_price']
                else:
                    fill_price = raw_price
                    fill = {
                        'timestamp': bar['timestamp'],
                        'symbol': self.asset,
                        'direction': direction,
                        'requested_price': raw_price,
                        'fill_price': fill_price,
                        'volume': volume,
                        'slippage_bps': 0,
                        'latency_ms': 0
                    }
                
                if direction == 'BUY':
                    if position < 0:
                        entry_price = trades[-1]['fill_price']
                        pnl = (entry_price - fill_price) * abs(position)
                        portfolio_value += pnl
                        
                        fill['pnl'] = pnl
                        fill['pnl_pct'] = pnl / (entry_price * abs(position))
                        trades.append(fill)
                    
                    position = volume
                    
                    fill['position'] = position
                    fill['portfolio_value'] = portfolio_value
                    trades.append(fill)
                    
                elif direction == 'SELL':
                    if position > 0:
                        entry_price = trades[-1]['fill_price']
                        pnl = (fill_price - entry_price) * position
                        portfolio_value += pnl
                        
                        fill['pnl'] = pnl
                        fill['pnl_pct'] = pnl / (entry_price * position)
                        trades.append(fill)
                    
                    position = -volume
                    
                    fill['position'] = position
                    fill['portfolio_value'] = portfolio_value
                    trades.append(fill)
            
            if i > 0 and position != 0:
                price_change = bar['close'] / data.iloc[i-1]['close'] - 1
                portfolio_change = price_change * position * bar['close'] / portfolio_value
                portfolio_value *= (1 + portfolio_change)
            
            portfolio_values.append(portfolio_value)
        
        performance = self._calculate_performance(portfolio_values, data)
        
        drawdowns = {}
        if self.drawdown_check:
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            drawdowns = {
                'max_drawdown': max_drawdown,
                'max_drawdown_threshold': 0.05,
                'passed': max_drawdown < 0.05
            }
        
        self.results['trades'] = trades
        self.results['performance'] = performance
        self.results['drawdowns'] = drawdowns
        self.results['portfolio_values'] = portfolio_values
        
        self.fill_engine.save_trades_csv("trades.csv")
        
        self.fill_engine.generate_costs_log("costs.log")
        
        self._generate_stress_report()
        
        return self.results
    
    def _calculate_performance(self, portfolio_values, data):
        """Calculate performance metrics"""
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        total_return = final_value / initial_value - 1
        
        days = (self.end_date - self.start_date).days
        annualized_return = (1 + total_return) ** (365 / max(1, days)) - 1
        
        returns = [portfolio_values[i] / portfolio_values[i-1] - 1 for i in range(1, len(portfolio_values))]
        sharpe_ratio = (np.mean(returns) - 0.02/365) / (np.std(returns) * np.sqrt(365))
        
        trades = self.results['trades']
        winning_trades = [t for t in trades if 'pnl' in t and t['pnl'] > 0]
        win_rate = len(winning_trades) / max(1, len([t for t in trades if 'pnl' in t]))
        
        if trades and any('pnl' in t for t in trades):
            avg_profit = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0] or [0])
            avg_loss = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0] or [0])
            profit_factor = abs(sum([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0] or [0]) / 
                              sum([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0] or [-1]))
        else:
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'initial_capital': portfolio_values[0],
            'final_capital': portfolio_values[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len([t for t in trades if 'pnl' in t])
        }
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _generate_stress_report(self):
        """Generate stress report"""
        stress_report = {
            'asset': self.asset,
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'slippage_enabled': self.slippage,
            'drawdown_check': self.drawdown_check,
            'performance': self.results['performance'],
            'drawdowns': self.results['drawdowns'],
            'win_rate': self.results['performance']['win_rate'],
            'win_rate_acceptable': 0.5 <= self.results['performance']['win_rate'] <= 0.8,
            'max_drawdown': self.results['drawdowns'].get('max_drawdown', 0),
            'max_drawdown_acceptable': self.results['drawdowns'].get('passed', True)
        }
        
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                    return str(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, bool):
                    return bool(obj)  # Explicitly convert booleans
                return str(obj)
        
        with open("stress_report.json", 'w') as f:
            json.dump(stress_report, f, indent=4, cls=CustomEncoder)
        
        print("Stress report saved to stress_report.json")
        return stress_report
    
    def plot_results(self, output_file="performance_chart.png"):
        """Plot performance results"""
        if 'portfolio_values' not in self.results:
            print("No results to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.results['portfolio_values'])
        plt.title(f"{self.asset} Trading Performance")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        
        if 'drawdowns' in self.results and self.results['drawdowns']:
            max_drawdown = self.results['drawdowns'].get('max_drawdown', 0)
            plt.axhline(y=self.results['portfolio_values'][0] * (1 - max_drawdown), 
                       color='r', linestyle='--', 
                       label=f"Max Drawdown: {max_drawdown:.2%}")
            plt.legend()
        
        if 'trades' in self.results and self.results['trades']:
            plt.subplot(2, 1, 2)
            
            trade_times = [pd.to_datetime(t['timestamp']) for t in self.results['trades'] if 'pnl' in t]
            trade_pnls = [t['pnl'] for t in self.results['trades'] if 'pnl' in t]
            
            plt.bar(trade_times, trade_pnls, color=['g' if pnl > 0 else 'r' for pnl in trade_pnls])
            plt.title("Trade PnL")
            plt.ylabel("Profit/Loss")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Performance chart saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Verify trading system with live data')
    parser.add_argument('--asset', type=str, default='XAU/USD',
                        help='Asset to test (default: XAU/USD)')
    parser.add_argument('--start', type=str, default='2020-03-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2022-10-01',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--slippage', type=str, choices=['on', 'off'], default='on',
                        help='Enable slippage simulation (default: on)')
    parser.add_argument('--drawdown-check', action='store_true',
                        help='Check for drawdowns')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to data file (optional)')
    parser.add_argument('--record', type=str, default='trades.csv',
                        help='Output file for trade records (default: trades.csv)')
    
    args = parser.parse_args()
    
    verifier = LiveDataVerifier(
        asset=args.asset,
        start_date=args.start,
        end_date=args.end,
        slippage=args.slippage == 'on',
        drawdown_check=args.drawdown_check
    )
    
    
    data = verifier.load_data(args.data_file)
    
    results = verifier.run_verification(data)
    
    verifier.plot_results()
    
    print("\n" + "="*50)
    print(f"Verification Results for {args.asset}")
    print("="*50)
    print(f"Period: {args.start} to {args.end}")
    print(f"Slippage: {'Enabled' if args.slippage == 'on' else 'Disabled'}")
    print(f"Drawdown Check: {'Enabled' if args.drawdown_check else 'Disabled'}")
    print("\nPerformance:")
    print(f"  Total Return: {results['performance']['total_return']:.2%}")
    print(f"  Annualized Return: {results['performance']['annualized_return']:.2%}")
    print(f"  Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {results['performance']['win_rate']:.2%}")
    print(f"  Profit Factor: {results['performance']['profit_factor']:.2f}")
    print(f"  Total Trades: {results['performance']['total_trades']}")
    
    if args.drawdown_check:
        print("\nDrawdown Analysis:")
        print(f"  Max Drawdown: {results['drawdowns']['max_drawdown']:.2%}")
        print(f"  Threshold: {results['drawdowns']['max_drawdown_threshold']:.2%}")
        print(f"  Status: {'PASSED' if results['drawdowns']['passed'] else 'FAILED'}")
    
    print("\nOutput Files:")
    print(f"  Trade Records: {args.record}")
    print(f"  Costs Log: costs.log")
    print(f"  Stress Report: stress_report.json")
    print(f"  Performance Chart: performance_chart.png")
    
    win_rate = results['performance']['win_rate']
    if 0.5 <= win_rate <= 0.8:
        print("\n✅ Win rate is realistic and acceptable")
    else:
        print(f"\n❌ Win rate ({win_rate:.2%}) is outside acceptable range (50%-80%)")
    
    if args.drawdown_check:
        max_drawdown = results['drawdowns']['max_drawdown']
        if max_drawdown < 0.05:
            print(f"✅ Max drawdown ({max_drawdown:.2%}) is below threshold (5%)")
        else:
            print(f"❌ Max drawdown ({max_drawdown:.2%}) exceeds threshold (5%)")
    
    return results


if __name__ == "__main__":
    main()
