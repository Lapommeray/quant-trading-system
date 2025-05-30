# Institutional Components Integration Guide

This guide demonstrates how to use the new institutional trading components while maintaining backward compatibility with existing systems.

## Enhanced Limit Order Book

### Basic Usage (Backward Compatible)
```python
from advanced_modules.hft_order_book import LimitOrderBook

# Existing usage continues to work unchanged
lob = LimitOrderBook()
order_id = lob.add_order(100.0, 1000, True)  # price, volume, is_bid
trades = lob.match_orders()
```

### Enhanced Usage with Institutional Features
```python
from advanced_modules.hft_order_book import EnhancedLimitOrderBook

# New institutional features
lob = EnhancedLimitOrderBook(enable_institutional_features=True)
lob.process_trade(100.0, 1000, 'buy')
vpin = lob.calculate_vpin()
print(f"VPIN: {vpin}")
```

## Advanced Cointegration

```python
from arbitrage.advanced_cointegration import AdvancedCointegration
import pandas as pd

# Multi-asset cointegration
coint = AdvancedCointegration()
prices_df = pd.DataFrame({'AAPL': aapl_prices, 'MSFT': msft_prices})
result = coint.johansen_test(prices_df)

if result['cointegrated']:
    hedge_ratios = result['hedge_ratios']
    print(f"Hedge ratios: {hedge_ratios}")
```

## Optimal Execution

```python
from execution.advanced import OptimalExecution, VWAPExecution

# VWAP execution
vwap = VWAPExecution(historical_volumes)
schedule = vwap.get_execution_schedule(
    target_quantity=10000,
    start_time="09:30",
    end_time="16:00"
)

# Optimal execution
optimal = OptimalExecution(risk_aversion=1e-6)
strategy = optimal.solve_optimal_strategy(
    target_shares=10000,
    time_horizon=10,
    volatility=0.2
)
```

## Institutional Indicators

```python
from core.institutional_indicators import HestonVolatility, ML_RSI, OrderFlowImbalance

# Heston volatility
heston = HestonVolatility()
vol = heston.calculate(price_series)

# ML-enhanced RSI
ml_rsi = ML_RSI()
predictions = ml_rsi.calculate(prices, traditional_rsi)

# Order flow imbalance (requires tick data)
ofi = OrderFlowImbalance()
imbalance = ofi.calculate(tick_data)
```

## Migration Path

1. **Phase 1**: Use new features as opt-in alongside existing implementations
2. **Phase 2**: Gradually migrate strategies to use enhanced features
3. **Phase 3**: Full adoption of institutional components (existing code continues to work)

## Dependencies

The institutional components require additional Python packages:
- cvxpy: For optimal execution algorithms
- statsmodels: For cointegration tests
- scikit-learn: For ML-enhanced indicators
- hmmlearn: For regime detection

These dependencies are optional and the components will gracefully degrade if they're not available.

## Integration with Quantum Finance Architecture

The institutional components are designed to work seamlessly with the existing quantum finance architecture:

- **Enhanced LimitOrderBook**: Extends the existing HFT order book with institutional features
- **AdvancedCointegration**: Provides sophisticated statistical methods for arbitrage detection
- **Execution Algorithms**: Implements institutional-grade execution strategies
- **Institutional Indicators**: Enhances signal generation with advanced analytics

All components follow the progressive enhancement approach, maintaining backward compatibility while adding new capabilities.
