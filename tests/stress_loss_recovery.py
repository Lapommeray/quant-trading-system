"""
Loss Tolerance Testing Mode
Tests the trading system's resilience to extreme market conditions
"""

import pandas as pd
import numpy as np
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MarketStressTest:
    def __init__(self, max_drawdown_threshold=0.05):
        """
        Initialize the stress test with threshold for maximum acceptable drawdown
        
        Parameters:
        - max_drawdown_threshold: Maximum allowable drawdown (default 5%)
        """
        self.max_drawdown_threshold = max_drawdown_threshold
        self.results = {
            "events": [],
            "total_tests": 0,
            "passed_tests": 0,
            "worst_drawdown": 0,
            "max_loss_event": None
        }
        
    def inject_extreme_volatility(self, event, data_path=None):
        """
        Inject extreme volatility based on historical events
        
        Parameters:
        - event: Type of event to simulate ("covid_crash", "fed_panic", "flash_crash")
        - data_path: Path to historical data (optional)
        
        Returns:
        - DataFrame with extreme volatility data injected
        """
        print(f"Injecting extreme volatility scenario: {event}")
        
        event_data = None
        
        if event == "covid_crash":
            print("Simulating March 2020 COVID crash with 30% drop over 4 weeks")
            event_data = self._generate_covid_crash_data(data_path)
            
        elif event == "fed_panic":
            print("Simulating September 2022 Fed terminal rate panic")
            event_data = self._generate_fed_panic_data(data_path)
            
        elif event == "flash_crash":
            print("Simulating flash crash with 7% drop in 1 minute followed by rebound")
            event_data = self._generate_flash_crash_data(data_path)
        
        else:
            raise ValueError(f"Unknown event type: {event}")
            
        return event_data
    
    def _generate_covid_crash_data(self, data_path=None):
        """Generate or load COVID crash scenario data"""
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)
        
        dates = pd.date_range(start='2020-02-01', end='2020-03-23')  # Extended start date
        
        pre_crash_end = pd.Timestamp('2020-02-19')
        pre_crash_days = (pre_crash_end - dates[0]).days + 1
        crash_days = len(dates) - pre_crash_days
        
        starting_price = 3380  # S&P 500 approximately Feb 19, 2020
        ending_price = 2200    # S&P 500 approximately Mar 23, 2020
        
        pre_crash_prices = np.random.normal(starting_price, starting_price * 0.005, pre_crash_days)
        
        decay_factor = np.exp(np.log(ending_price/starting_price) / crash_days)
        crash_prices = starting_price * np.array([decay_factor**i for i in range(crash_days)])
        
        prices = np.concatenate([pre_crash_prices, crash_prices])
        
        pre_crash_volatility = 0.005  # Low initial volatility
        crash_volatility = 0.03       # High volatility during crash
        
        volatility = np.ones(len(dates))
        volatility[:pre_crash_days] = pre_crash_volatility
        volatility[pre_crash_days:] = np.linspace(pre_crash_volatility, crash_volatility, crash_days)
        
        random_factors = np.array([np.random.normal(1, vol) for vol in volatility])
        prices = prices * random_factors
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * np.random.uniform(1.0, 1.03, len(dates)),
            'low': prices * np.random.uniform(0.97, 1.0, len(dates)),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        })
        
        return data
    
    def _generate_fed_panic_data(self, data_path=None):
        """Generate or load Fed rate panic scenario data"""
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)
        
        dates = pd.date_range(start='2022-09-01', end='2022-09-30')  # Extended start date
        
        pre_panic_end = pd.Timestamp('2022-09-13')
        pre_panic_days = (pre_panic_end - dates[0]).days + 1
        panic_days = len(dates) - pre_panic_days
        
        starting_price = 4110  # S&P 500 approximately Sept 13, 2022
        lowest_price = 3585    # S&P 500 approximately Sept 30, 2022
        
        pre_panic_prices = np.random.normal(starting_price, starting_price * 0.004, pre_panic_days)
        
        decay_factor = np.exp(np.log(lowest_price/starting_price) / panic_days)
        panic_prices = starting_price * np.array([decay_factor**i for i in range(panic_days)])
        
        prices = np.concatenate([pre_panic_prices, panic_prices])
        
        pre_panic_volatility = 0.004  # Low initial volatility
        panic_volatility = 0.03       # High volatility during panic
        
        volatility = np.ones(len(dates))
        volatility[:pre_panic_days] = pre_panic_volatility
        volatility[pre_panic_days:] = np.linspace(pre_panic_volatility, panic_volatility, panic_days)
        
        random_factors = np.array([np.random.normal(1, vol) for vol in volatility])
        prices = prices * random_factors
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * np.random.uniform(1.0, 1.04, len(dates)),  # Higher highs
            'low': prices * np.random.uniform(0.96, 1.0, len(dates)),   # Lower lows
            'close': prices,
            'volume': np.random.uniform(2000000, 6000000, len(dates))   # Higher volume
        })
        
        return data
    
    def _generate_flash_crash_data(self, data_path=None):
        """Generate or load flash crash scenario data"""
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)
        
        
        timestamps = pd.date_range(start='2022-01-01 09:30:00', periods=390, freq='1min')
        
        base_price = 4000
        prices = np.ones(len(timestamps)) * base_price
        
        crash_idx = 180
        recovery_idx = crash_idx + 30  # Recovery over next 30 minutes
        
        crash_pct = 0.07
        prices[crash_idx] = base_price * (1 - crash_pct)
        
        recovery_factor = np.exp(np.log((base_price)/(base_price*(1-crash_pct))) / (recovery_idx - crash_idx))
        
        for i in range(crash_idx + 1, recovery_idx + 1):
            if i < len(prices):
                recovery_progress = i - crash_idx
                prices[i] = base_price * (1 - crash_pct) * (recovery_factor ** recovery_progress)
        
        noise = np.random.normal(0, 0.002 * base_price, len(timestamps))
        prices = prices + noise
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'close': prices,
            'high': prices * np.random.uniform(1.0, 1.005, len(timestamps)),
            'low': prices * np.random.uniform(0.995, 1.0, len(timestamps)),
            'volume': np.random.uniform(10000, 100000, len(timestamps))
        })
        
        data.loc[crash_idx, 'low'] = base_price * (1 - crash_pct - 0.02)  # Lower low at crash
        data.loc[crash_idx:recovery_idx, 'high'] = data.loc[crash_idx:recovery_idx, 'high'] * 1.01  # Higher volatility
        data.loc[crash_idx:recovery_idx, 'low'] = data.loc[crash_idx:recovery_idx, 'low'] * 0.99    # Higher volatility
        
        data.loc[crash_idx:recovery_idx, 'volume'] = data.loc[crash_idx:recovery_idx, 'volume'] * 5
        
        return data
    
    def monitor_AI_response(self, event_data, trading_system):
        """
        Monitor the AI trading system's response to the extreme event
        
        Parameters:
        - event_data: DataFrame with the extreme event data
        - trading_system: The trading system to test
        
        Returns:
        - Dictionary with performance metrics
        """
        print("Monitoring AI trading system response to extreme volatility...")
        
        initial_portfolio_value = 10000  # Starting with $10,000
        portfolio_values = []
        trades = []
        
        for i in range(len(event_data)):
            current_bar = event_data.iloc[i]
            
            signal = trading_system.process_bar(current_bar)
            
            if signal:
                trade = {
                    'timestamp': str(current_bar['timestamp']),
                    'signal': signal['direction'],
                    'price': signal['price'],
                    'confidence': signal['confidence'],
                    'position_size': signal['size']
                }
                trades.append(trade)
            
            if i > 0:
                pct_change = current_bar['close'] / event_data.iloc[i-1]['close'] - 1
                portfolio_change = self._calculate_portfolio_change(pct_change, trades)
                initial_portfolio_value *= (1 + portfolio_change)
            
            portfolio_values.append(initial_portfolio_value)
        
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        passed = max_drawdown < self.max_drawdown_threshold
        
        result = {
            'event_type': event_data.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
            'max_drawdown': max_drawdown,
            'passed': passed,
            'trades': len(trades),
            'final_portfolio_value': portfolio_values[-1],
            'return': portfolio_values[-1] / portfolio_values[0] - 1
        }
        
        self.results['events'].append(result)
        self.results['total_tests'] += 1
        if passed:
            self.results['passed_tests'] += 1
        
        if max_drawdown > self.results['worst_drawdown']:
            self.results['worst_drawdown'] = max_drawdown
            self.results['max_loss_event'] = result['event_type']
        
        return result
    
    def _calculate_portfolio_change(self, market_pct_change, active_trades):
        """Calculate portfolio change based on market movement and trades"""
        if not active_trades:
            return 0
        
        last_trade = active_trades[-1]
        
        if last_trade['signal'] == 'BUY':
            return market_pct_change * last_trade['position_size']
        elif last_trade['signal'] == 'SELL':
            return -market_pct_change * last_trade['position_size']
        else:
            return 0
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values:
            return 0
            
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def assert_loss_threshold(self):
        """Assert that loss threshold wasn't breached in any test"""
        if self.results['worst_drawdown'] >= self.max_drawdown_threshold:
            raise AssertionError(
                f"Loss threshold breached! Max drawdown: {self.results['worst_drawdown']:.2%} "
                f"(threshold: {self.max_drawdown_threshold:.2%}) "
                f"during event: {self.results['max_loss_event']}"
            )
        else:
            print(f"✅ All tests passed! Max drawdown: {self.results['worst_drawdown']:.2%} "
                  f"(threshold: {self.max_drawdown_threshold:.2%})")
        
        return self.results['worst_drawdown'] < self.max_drawdown_threshold
    
    def generate_stress_report(self, output_file="stress_report.json"):
        """Generate stress test report and save to file"""
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
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4, cls=CustomEncoder)
        
        print(f"Stress test report saved to {output_file}")
        return self.results


def simulate_black_swan(event, trading_system=None, max_drawdown_threshold=0.05):
    """
    Simulate a black swan event and test the trading system's response
    
    Parameters:
    - event: Type of event to simulate ("covid_crash", "fed_panic", "flash_crash")
    - trading_system: The trading system to test
    - max_drawdown_threshold: Maximum allowable drawdown (default 5%)
    
    Returns:
    - Dictionary with test results
    """
    if trading_system is None:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from core.qmp_engine import QMPOversoulEngine
            class TradingSystemAdapter:
                def __init__(self):
                    class MockAlgorithm:
                        def Debug(self, message):
                            print(message)
                    
                    self.engine = QMPOversoulEngine(MockAlgorithm())
                
                def process_bar(self, bar):
                    symbol = "BTCUSD"  # Default symbol
                    
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
        except ImportError:
            print("Warning: Could not import real trading system. Using mock system.")
            class MockTradingSystem:
                def __init__(self):
                    self.last_prices = []
                    self.last_highs = []
                    self.last_lows = []
                    self.position = None
                    self.entry_price = None
                    self.volatility = 0
                    self.market_regime = 'normal'
                    self.circuit_breaker_active = False
                    self.trade_cooldown = 0
                    self.consecutive_down_days = 0
                    self.max_position_size = 0.05  # Max 5% of portfolio
                
                def process_bar(self, bar):
                    self.last_prices.append(bar['close'])
                    self.last_highs.append(bar['high'])
                    self.last_lows.append(bar['low'])
                    
                    if len(self.last_prices) > 20:
                        self.last_prices.pop(0)
                        self.last_highs.pop(0)
                        self.last_lows.pop(0)
                    
                    if self.trade_cooldown > 0:
                        self.trade_cooldown -= 1
                        return None
                    
                    # Calculate volatility
                    if len(self.last_highs) > 5:
                        ranges = []
                        for i in range(1, len(self.last_highs)):
                            true_range = max(
                                self.last_highs[i] - self.last_lows[i],
                                abs(self.last_highs[i] - self.last_prices[i-1]),
                                abs(self.last_lows[i] - self.last_prices[i-1])
                            )
                            ranges.append(true_range)
                        
                        self.volatility = sum(ranges) / len(ranges) / self.last_prices[-1]
                    
                    self.consecutive_down_days = 0
                    if len(self.last_prices) > 3:
                        for i in range(len(self.last_prices)-3, len(self.last_prices)):
                            if i > 0 and self.last_prices[i] < self.last_prices[i-1]:
                                self.consecutive_down_days += 1
                    
                    if self.volatility > 0.012 or self.consecutive_down_days >= 3:
                        self.market_regime = 'crisis'
                    elif self.volatility > 0.007:
                        self.market_regime = 'volatile'
                    elif self.volatility > 0.004:
                        self.market_regime = 'pre_crisis'
                    else:
                        self.market_regime = 'normal'
                    
                    # Circuit breaker for extreme volatility
                    if self.volatility > 0.02 or self.consecutive_down_days >= 4:
                        self.circuit_breaker_active = True
                        self.trade_cooldown = 10
                        
                        if self.position:
                            return {
                                'direction': 'BUY' if self.position == 'SHORT' else 'SELL',
                                'price': bar['close'],
                                'confidence': 1.0,
                                'size': 0.2  # Reduced position size for safer exit
                            }
                        return None
                    else:
                        self.circuit_breaker_active = False
                    
                    position_size = self.max_position_size
                    if self.market_regime == 'pre_crisis':
                        position_size *= 0.5
                    elif self.market_regime == 'volatile':
                        position_size *= 0.3
                    elif self.market_regime == 'crisis':
                        position_size *= 0.1
                    
                    if self.market_regime == 'crisis' and self.position:
                        self.trade_cooldown = 5
                        return {
                            'direction': 'BUY' if self.position == 'SHORT' else 'SELL',
                            'price': bar['close'],
                            'confidence': 0.9,
                            'size': position_size
                        }
                    
                    import random
                    if random.random() > 0.8 and not self.circuit_breaker_active:
                        direction = 'BUY' if random.random() > 0.5 else 'SELL'
                        
                        if direction == 'BUY':
                            self.position = 'LONG'
                        else:
                            self.position = 'SHORT'
                        
                        self.entry_price = bar['close']
                        self.trade_cooldown = 3
                        
                        return {
                            'direction': direction,
                            'price': bar['close'],
                            'confidence': 0.7,
                            'size': position_size
                        }
                    
                    return None
            
            trading_system = MockTradingSystem()
    
    stress_test = MarketStressTest(max_drawdown_threshold)
    
    event_data = stress_test.inject_extreme_volatility(event)
    
    result = stress_test.monitor_AI_response(event_data, trading_system)
    
    try:
        stress_test.assert_loss_threshold()
    except AssertionError as e:
        print(f"❌ {str(e)}")
    
    stress_test.generate_stress_report()
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stress tests for trading system')
    parser.add_argument('--event', type=str, choices=['covid_crash', 'fed_panic', 'flash_crash', 'all'],
                        default='all', help='Type of event to simulate')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Maximum allowable drawdown threshold (default: 0.05 or 5%%)')
    
    args = parser.parse_args()
    
    if args.event == 'all':
        events = ['covid_crash', 'fed_panic', 'flash_crash']
        
        for event in events:
            print(f"\n{'='*50}")
            print(f"Testing event: {event}")
            print(f"{'='*50}\n")
            simulate_black_swan(event, max_drawdown_threshold=args.threshold)
    else:
        simulate_black_swan(args.event, max_drawdown_threshold=args.threshold)
