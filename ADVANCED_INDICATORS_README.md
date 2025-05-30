# Advanced Trading Indicators

This document describes the four advanced indicators added to enhance the quant-trading-system.

## Overview

The advanced indicators complement existing functionality without replacing current indicators:

1. **HestonVolatility** - Advanced volatility modeling using Heston stochastic volatility
2. **ML_RSI** - Machine learning enhanced RSI with predictive capabilities  
3. **OrderFlowImbalance** - Tick-level order flow analysis
4. **RegimeDetector** - Hidden Markov Model for market regime detection

## Usage

### Import the indicators:
```python
from core.indicators import HestonVolatility, ML_RSI, OrderFlowImbalance, RegimeDetector
```

### Basic Usage Examples:

#### Heston Volatility
```python
heston_vol = HestonVolatility(lookback=30)
volatility = heston_vol.calculate(close_prices)  # pandas Series
```

#### ML-Enhanced RSI
```python
# First calculate traditional RSI (preserved from existing system)
traditional_rsi = calculate_rsi(close_prices)  # your existing RSI calculation

# Enhance with ML
ml_rsi = ML_RSI(window=14, lookahead=5)
ml_predictions = ml_rsi.calculate(close_prices, traditional_rsi)
```

#### Order Flow Imbalance
```python
# Requires tick data with columns: price, quantity, side (1=buy, -1=sell)
order_flow = OrderFlowImbalance(window=100)
imbalance = order_flow.calculate(tick_data_df)
```

#### Regime Detection
```python
regime_detector = RegimeDetector(n_regimes=3)
regimes = regime_detector.calculate(volatility, rsi_values, other_indicators)
```

## Integration with Existing System

### With Enhanced Indicator
```python
from QMP_v2.1_FINAL_SMART_RISK_FULL.core.enhanced_indicator import EnhancedIndicator
from core.indicators import HestonVolatility, ML_RSI

# In your strategy
enhanced_indicator = EnhancedIndicator()
heston_vol = HestonVolatility()

# Get traditional signal
signal = enhanced_indicator.get_signal(symbol, df)

# Enhance with advanced volatility
volatility = heston_vol.calculate(df['Close'])
if volatility.iloc[-1] > 0.3:  # High volatility
    signal['confidence'] *= 0.8  # Reduce confidence
```

### With Backtesting Engine
```python
# In strategy.py next() method
def next(self):
    # Existing indicators (preserved)
    rsi = self.I(RSI(self.data.Close, 14))
    
    # New indicators
    heston_vol = self.I(HestonVolatility().calculate(self.data.Close))
    ml_rsi = self.I(ML_RSI().calculate(self.data.Close, rsi))
    
    # Trading logic using enhanced indicators
    if (ml_rsi[-1] > 0.5) and (heston_vol[-1] < 0.2):
        self.buy()
```

## Performance Notes

- **Heston Volatility**: Computationally intensive - consider caching results
- **ML_RSI**: Requires at least 1,000 samples for training
- **Order Flow**: Needs real tick data for production use
- **Regime Detection**: Works best with multiple uncorrelated indicators

## Dependencies

Added to requirements.txt:
- `hmmlearn>=0.3.0` (for Regime Detection)

Existing dependencies used:
- `scikit-learn>=0.23.0` (for ML_RSI)
- `scipy>=1.5.0` (for Heston optimization)
- `numpy>=1.19.0`, `pandas>=1.0.0` (all indicators)
