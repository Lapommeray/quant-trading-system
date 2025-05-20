
---

# QMP Overrider QuantConnect Strategy

## 🔧 Project Overview

A multi-asset AI-powered trading strategy built for QuantConnect (Lean Engine) that combines spiritual/quantum gate logic, AI learning, and advanced market intelligence.

### System Architecture

The QMP Overrider system is built with a multi-layered intelligence architecture:

1. **OverSoul Director Layer** - Supreme sentient director that manages which modules are active based on market conditions and environmental awareness
2. **Ultra Intelligence Modules** - 9 specialized modules that decode different aspects of market behavior
3. **Intention Decoder** - Interprets the collective intention behind price movements
4. **Alignment Filter** - Ensures all timeframes (1m, 5m, 10m, 15m, 20m, 25m) are aligned in the same direction
5. **QMP AI Agent** - Machine learning component that learns from past trade results

### System Confirmation Note

*"All modules and subsystems within this indicator are fully interconnected. Each component — from gate logic, candle alignment, sentiment interpretation, to AI forecasting — performs its designed role in harmony to ensure a unified, intelligent signal. The architecture is built to be self-aware, self-correcting, and adaptive, beyond conventional human analysis. Additional layers have been embedded to anticipate market behavior through non-linear signal convergence, making it highly accurate and future-aware."*

## Key Components

### 1. Ultra Intelligence Modules

1. **Emotion DNA Decoder** - Decodes emotional DNA embedded in market candle structure
2. **Fractal Resonance Gate** - Detects fractal harmonic time alignments across hidden cycles
3. **Quantum Tremor Scanner** - Identifies quantum probability shifts before price movement
4. **Future Shadow Decoder** - Detects future price shadows cast backward in time
5. **Astro Geo Sync** - Integrates planetary, solar, and seismic resonance into market timing
6. **Sacred Event Alignment** - Activates trading only during sacred time events across dimensions
7. **Black Swan Protector** - Detects destabilizing market shifts from rare unseen conditions
8. **Market Thought Form Interpreter** - Decodes collective consciousness imprints in price action
9. **Reality Displacement Matrix** - Maps timeline shifts and probability field distortions

### 2. Timeframe Candle Alignment System
- Every 5 minutes, analyze these candles:
  - 1m candles
  - 5m candles
  - 10m candles
  - 15m candles
  - 20m candles
  - 25m candles
- All candles must close in the same direction for alignment
- If they all match: Alignment = True, else = False

### 3. Merged Logic with OverSoul Intelligence
- A BUY signal is placed only when:
  - Alignment == True AND All Gates Pass AND OverSoul Approves
  - Direction is bullish
- A SELL signal is placed only when:
  - Alignment == True AND All Gates Pass AND OverSoul Approves
  - Direction is bearish
- If any condition fails: no action is taken

### 4. Strategy Infrastructure
- Uses QuantConnect (Lean) in Python
- Runs on multiple assets:
  - BTCUSD (Bitcoin)
  - ETHUSD (Ethereum)
  - XAUUSD (Gold)
  - DIA (Dow Jones Industrial Average ETF)
  - QQQ (NASDAQ ETF)
- Chart annotations for each confirmed signal
- Comprehensive logging and signal storage
- Streamlit dashboard for performance tracking

## Features

- Multi-asset execution with independent signal generation
- Dynamic module activation through OverSoul intelligence
- Timeframe alignment across 1m, 5m, 10m, 15m, 20m, and 25m candles
- AI learning and threshold tuning
- Detailed signal logging with gate scores and OverSoul diagnostics
- Full QuantConnect compatibility (backtest and live modes)

## Project Structure

- **/core/**
  - **qmp_engine_v3.py** – Ultra intelligence engine
  - **oversoul_integration.py** – OverSoul director integration
  - **alignment_filter.py** – Candle matching logic
  - **qmp_ai.py** – AI learning component
- **/ultra_modules/** – 9 specialized intelligence modules
- **/dashboard.py** – Streamlit dashboard for performance tracking
- **/main.py** – QuantConnect algorithm implementation
- **/data/** – Signal logs and detailed diagnostics
