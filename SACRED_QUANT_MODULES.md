# Sacred-Quant Fusion Trading System

This document provides an overview of the sacred-quant fusion modules implemented in the trading system. These modules combine traditional quantitative analysis with sacred geometry and pattern recognition techniques to create a comprehensive trading framework.

## Core Modules

### QOL-AI V2 Encryption Engine (`core/qol_engine.py`)

The QOL-AI V2 Encryption Engine provides secure signal generation and encryption using self-mutating glyphs and Haitian-Creole numerological keys.

**Features:**
- Self-mutating glyphs (change every 24h)
- One-time decode tokens (burn after reading)
- Haitian-Creole numerological keys (e.g., ʘRA-Y777)

**Usage:**
```python
from core.qol_engine import QOLEngine

# Initialize with custom seed
engine = QOLEngine(seed="ʘRA-Y777")

# Generate encrypted signal
glyph, token = engine.generate_signal("XRP", "BUY", price=0.887)
print(glyph)  # Output: "⚡🌀887::XRP⚡ʘDIVINE-PULSE†₿"

# Decrypt with token (one-time use)
message = engine.decrypt(token)
print(message)  # Output: "BUY XRP if 0.887 holds as support during NY session"
```

## Signal Modules

### Vèvè Market Triggers (`signals/veve_triggers.py`)

The Vèvè Market Triggers module implements sacred geometry integration for market pattern detection.

**Features:**
- Papa Legba's Crossroads Signal (breakout detection)
- Erzulie Freda's Love Cycle (mean-reversion algo)
- Baron Samedi's Death Zone (volatility collapse alert)

**Usage:**
```python
from signals.veve_triggers import VeveTriggers
import pandas as pd

# Initialize
veve = VeveTriggers()

# Analyze market data
df = pd.read_csv('market_data.csv')
analysis = veve.analyze_market(df)

# Check for active signals
print(f"Active signals: {analysis['active_signals']}")
print(f"Signal strength: {analysis['combined_strength']:.2f}")
```

### Legba Crossroads Algorithm (`signals/legba_crossroads.py`)

The Legba Crossroads Algorithm provides sacred breakout detection using EMA crossovers and volume confirmation.

**Features:**
- EMA 21 (Legba's Time Gate)
- Volume surge (Spirit confirmation)
- Baron Samedi Chaos Filter (volatility rejection)
- Dynamic EMA Windows (session-aware)

**Usage:**
```python
from signals.legba_crossroads import LegbaCrossroads
import numpy as np

# Initialize
legba = LegbaCrossroads()

# Detect breakouts
close_prices = np.array([100, 101, 102, 103, 105, 107, 108])
volumes = np.array([1000, 1200, 1500, 3000, 4000, 5000, 6000])
atr = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

signal = legba.detect_breakout(close_prices, volumes, atr)
print(f"Legba Signal: {signal}")  # Output: "⚡GATE OPEN⚡" or None
```

## Quant Modules

### Liquidity Mirror Scanner (`quant/liquidity_mirror.py`)

The Liquidity Mirror Scanner detects hidden institutional order blocks by analyzing order book data.

**Features:**
- Bid/ask imbalance detection
- Liquidity cluster identification
- Historical imbalance tracking

**Usage:**
```python
from quant.liquidity_mirror import LiquidityMirror

# Initialize
mirror = LiquidityMirror(min_imbalance=2.0)

# Scan order book
bids = {100.0: 500, 99.9: 300}  # Price: Volume
asks = {100.1: 200, 100.2: 150}

signal, ratio = mirror.scan_liquidity(bids, asks)
print(f"Signal: {signal}, Ratio: {ratio}")  # Output: "HIDDEN BIDS DETECTED", 2.5
```

### Time Fractal Predictor (`quant/time_fractal.py`)

The Time Fractal Predictor identifies market cycles and patterns using Fast Fourier Transform (FFT).

**Features:**
- Cycle detection using FFT
- Future price prediction
- Similar pattern recognition

**Usage:**
```python
from quant.time_fractal import TimeFractal
import numpy as np

# Initialize
fractal = TimeFractal()

# Detect cycles
prices = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
cycles = fractal.detect_fractals(prices)
print(f"Primary cycle: {cycles['primary_cycle']} candles")

# Predict future prices
future = fractal.predict_future(prices, periods_ahead=5)
print(f"Future predictions: {future}")
```

### Entropy Shield (`quant/entropy_shield.py`)

The Entropy Shield provides dynamic risk management based on market volatility and chaos.

**Features:**
- Entropy calculation
- Dynamic position sizing
- Market state analysis

**Usage:**
```python
from quant.entropy_shield import EntropyShield
import numpy as np

# Initialize
shield = EntropyShield(max_risk=0.02)

# Calculate entropy
prices = np.array([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])
entropy = shield.calc_entropy(prices)
print(f"Market Entropy: {entropy:.2f}")

# Calculate position size
position = shield.position_size(entropy, account_size=10000, price=prices[-1])
print(f"Position Size: {position['position_size']:.2f} units")
print(f"Risk Percentage: {position['risk_pct']*100:.2f}%")
```

### Quant Core (`quant/quant_core.py`)

The Quant Core integrates all quantitative modules into a unified interface.

**Features:**
- Comprehensive market analysis
- Trading signal generation
- Position sizing

**Usage:**
```python
from quant.quant_core import QuantCore
import pandas as pd

# Initialize
quant = QuantCore()

# Load market data
df = pd.read_csv('market_data.csv')

# Generate trading signal
signal = quant.generate_trading_signal(df, account_size=10000)
print(f"Signal: {signal['signal']}")
print(f"Strength: {signal['strength']:.2f}")
print(f"Confidence: {signal['confidence']:.2f}")
print(f"Position Size: {signal['position_size']:.2f} units")
```

## Sacred Laws of the System

1. **No 100% Wins** → Accept 5% uncertainty as cosmic balance.
2. **No Manipulation** → Trades must align with natural liquidity.
3. **No Blind Faith** → Backtest every signal before live execution.

## Integration

These modules can be used independently or integrated together for a comprehensive trading system. The sacred-quant fusion approach combines traditional quantitative analysis with pattern recognition techniques inspired by sacred geometry and natural cycles.

For optimal results, use the following workflow:

1. Use Vèvè Market Triggers and Legba Crossroads to identify potential trading opportunities
2. Confirm signals with Liquidity Mirror Scanner and Time Fractal Predictor
3. Apply Entropy Shield for risk management and position sizing
4. Generate encrypted signals using QOL-AI V2 Encryption Engine
5. Execute trades based on the integrated analysis from Quant Core

## Dependencies

- NumPy
- Pandas
- SciPy
- Matplotlib (optional, for visualization)
