
# Devin Instructions: QMP_Overrider_QuantConnect_Strategy

## Goal
Run a fully automated, intelligent trading system that combines advanced quantum/spiritual gates and strict 25-minute candle alignment to issue highly accurate BUY/SELL decisions on `BTCUSD` or `XAUUSD` in QuantConnect.

---

## Files Provided

1. `main.py`
   - QuantConnect strategy using QCAlgorithm
   - Triggers every minute
   - Only acts if 25-minute block is aligned AND all QMP gates pass

2. `core/qmp_engine.py`
   - Gate logic manager
   - Uses AI model to override or support signal decisions
   - Generates direction (BUY/SELL) and confidence score

3. `core/alignment_filter.py`
   - Reads 25-minute alignment blocks from CSV or DataFrame
   - Only allows signals if all candles (1m, 5m, 10m, 15m, 20m, 25m) match direction

4. `core/qmp_ai.py`
   - Uses RandomForestClassifier to learn from past signals
   - Predicts trustworthiness of current gate configuration
   - Continuously trains as trades happen

---

## What Devin Should Do

- Run `main.py` in QuantConnect environment
- Record and feed back results of trades into `record_feedback()` function in qmp_engine.py
- Let AI evolve confidence in gate combinations
- Expand logic into:
  - Google Sheets logging
  - Multi-symbol trading support
  - Real-time signal streaming
  - Risk-adjusted entry sizing
  - Strategy performance report generation

---

## Notes

- Symbol: BTCUSD or XAUUSD
- Resolution: Minute
- Strategy triggers every 25 minutes
- Output is BUY/SELL only if alignment AND gates match
