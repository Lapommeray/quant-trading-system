# QMP Overrider Production Deployment Guide

This guide explains how to deploy the enhanced QMP Overrider trading system to QuantConnect for live trading with maximum win rate.

## System Components

The production-ready QMP Overrider system includes these key components:

1. **Risk Management System** - Implements Kelly Criterion with volatility scaling
2. **Event Blackout System** - Prevents trading during high-impact news events
3. **Walk-Forward Backtesting** - Eliminates look-ahead bias for accurate performance metrics
4. **Live Data Integration** - Replaces static CSV with real-time market data
5. **Performance Optimization** - Accelerates critical calculations with numba

## Deployment Steps

### 1. QuantConnect Setup

1. Log in to your QuantConnect account
2. Create a new algorithm project
3. Upload all files from the `QMP_GOD_MODE_v2_5_FINAL` directory
4. Configure your API keys in the QuantConnect settings

### 2. Risk Management Configuration

The risk management system is pre-configured with conservative settings:
- 1% maximum portfolio risk per trade
- 25% maximum position size
- Volatility-based position scaling

To adjust these parameters, modify `risk_manager.py`:

```python
self.max_portfolio_risk = 0.01  # Adjust risk percentage (0.01 = 1%)
self.max_position_size = 0.25   # Maximum position size (0.25 = 25%)
self.volatility_lookback = 30   # Days to look back for volatility calculation
```

### 3. Event Blackout Configuration

The system automatically avoids trading during major economic events:
- Non-Farm Payrolls (NFP) - Fridays 8:30 AM EST
- FOMC Meetings - Wednesdays 2:00 PM EST
- CPI Releases - 8:30 AM EST on release days
- GDP Announcements - 8:30 AM EST on release days

To customize event blackouts, modify `event_blackout.py`:

```python
self.blackout_events = {
    "NFP": {"time": "08:30", "duration": 30, "days": [4]},  # Friday
    "FOMC": {"time": "14:00", "duration": 120, "days": [2]}, # Wednesday
    # Add custom events here
}
```

### 4. Backtesting Verification

Before deploying to live trading, verify system performance:

1. Run the walk-forward backtester to ensure no look-ahead bias:

```python
from backtest.walk_forward_quantum_backtest import run_walk_forward_backtest
from datetime import datetime

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)
results = run_walk_forward_backtest(data, start_date, end_date)
print(f"Win rate: {results['win_rate']:.2%}")
```

2. Verify that transaction costs are properly accounted for in backtests

### 5. Live Trading Deployment

1. In QuantConnect, click "Backtest" to verify the algorithm works
2. Review the backtest results and confirm:
   - Risk management is functioning correctly
   - Event blackouts are preventing trades during high-risk periods
   - Performance is optimized with numba acceleration
3. Click "Live Trading" to deploy the algorithm
4. Monitor the initial trades closely to ensure proper execution

## Performance Monitoring

The system includes built-in performance monitoring:

1. **Signal Feedback Log** - Records all trading signals and outcomes
2. **Detailed Signal Log** - Captures gate scores and environment state
3. **Live Alignment Log** - Tracks alignment data for analysis

Access these logs in the QuantConnect "Data" folder after running the algorithm.

## Troubleshooting

If you encounter issues:

1. **No Trades Executing**
   - Check that confidence threshold (default 0.65) isn't too high
   - Verify event blackout isn't blocking all trading periods
   - Ensure alignment conditions are being met

2. **Excessive Trading**
   - Increase confidence threshold in `main.py`
   - Extend blackout durations in `event_blackout.py`
   - Adjust risk parameters in `risk_manager.py`

3. **Performance Issues**
   - Ensure numba is properly installed
   - Reduce data processing frequency if needed
   - Optimize timeframe consolidation settings

## Advanced Configuration

For maximum win rate, consider these advanced settings:

1. **Increase Gate Thresholds**
   ```python
   qmp_engine.ultra_engine.min_gate_score = 0.7  # Default is 0.5
   ```

2. **Extend Alignment Requirements**
   ```python
   # Require longer alignment periods for stronger signals
   is_aligned = self.live_data_manager.get_live_alignment_data(
       now, symbol, self.symbol_data[symbol]["history_data"], 
       min_aligned_periods=3  # Default is 1
   )
   ```

3. **Implement Adaptive Position Sizing**
   ```python
   # In risk_manager.py, adjust position size based on market regime
   if market_volatility > historical_volatility * 1.5:
       position_size *= 0.5  # Reduce position size in high volatility
   ```

By following this guide, you'll have a production-ready QMP Overrider system deployed on QuantConnect with maximum win rate potential.
