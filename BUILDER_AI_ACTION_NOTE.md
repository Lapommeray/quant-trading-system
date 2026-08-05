# BUILDER AI - URGENT ACTION NOTE
**FROM:** Deep Institutional Analysis Agent
**TO:** Builder / Implementation AI
**PRIORITY:** P0 - System generates fake alpha, must fix before live
**DATE:** 2026-08-05

> You are not here to add more astrology. You are here to make money. Read this then code.

---

## TL;DR WHAT'S BROKEN

1. **EnhancedIndicator** is 3 random number generators duct taped: `fed_whisperer.py` random text, `liquidity_xray.py` random mid_price 100±1, `candlestick_dna_sequencer` Fibonacci numerology. DELETE IT FROM HOT PATH.

2. **60% of modules are toxic noise** - anything with `alien`, `angel`, `cosmic`, `divine`, `multiverse`, `zero_point`, `astro_geo`, `quantum_tremor`, `sacred`, `future_shadow`, `ghost_candle`. They return `{"confidence":0.5, "direction":"NEUTRAL"}` or random BUY/SELL. They dilute real signal weight to 4% each.

3. **Real institutional modules exist but buggy and disconnected** - `institutional_order_flow.py` good skeleton, but `core/institutional_indicators.py` version has broken rolling logic. `HestonVolatility` math wrong, `ML_RSI` leaks future, `RegimeDetector` no scaling.

4. **No real data** - No Polygon, no FRED, no options, noFunding, no on-chain flow, no FINRA. All mocked.

---

## YOUR TASK LIST - DO IN ORDER

### PHASE 0: QUARANTINE (1 hour)

- Move to `quarantine/` folder (git mv, don't delete history):
```
alien_decoder.py
angel_decoder.py
cosmic_channeler.py
divine_sync.py
multiverse_sync.py
zero_point.py
energy_filter.py (if empty)
timeline_selector.py (if astrology)
astro_geo_sync.py
emotion_dna_decoder.py
fractal_resonance_gate.py
future_shadow_decoder.py
intention_decoder.py
reality_displacement_matrix.py
sacred_event_alignment.py
future_zone_sensory.py
ghost_candle_projector.py
timeline_warp_plot.py
quantum_tremor_scanner.py
quantum_sentiment_decoder.py (the fake one in Deco_19)
market_thought_form_interpreter.py
black_swan_protector.py (fake version)
big_move_detector.py (random version)
time_fractal_fft.py
void_trader_chart_renderer.py
meta_conscious_routing_layer.py (stub)
dna_breath.py / dna_overlord.py
```
- Edit `core/qmp_engine_v3.py` module_weights: set weight 0 for quarantined, boost legit ones to sum 1.0.

### PHASE 1: KILL ENHANCED_INDICATOR AND REBUILD (P0)

**File:** `Deco_19/core/enhanced_indicator.py` + symlink `enhanced_indicator.py`

Current `get_signal` :
```python
if fed_bias=="dovish" and dna_prediction=="bullish": BUY
```
-> Replace with:

```python
class RealInstitutionalIndicator:
  def __init__(self, polygon_client, fred_client):
    self.ofi = OrderFlowImbalanceDetector() # from advanced_modules/institutional_order_flow.py
    self.vwp = VWAPDeviation()
    self.gex = GammaExposure()
    self.funding = FundingRateZScore()
    self.macro = RealFedModel(fred_client)
    
  def get_signal(self, symbol, l2_book, trades, options_chain):
    # 1. Tier1 Risk Gates
    if self.macro.is_fomc_blackout(): return NEUTRAL
    if self.regime_detector.is_crash_regime(): reduce size 70%
    
    # 2. Tier2 Alpha - need 2 of 3 to agree
    ofi_score = self.ofi.process(l2_book)  # -1 to 1
    cvd_div = self.cvd.detect_divergence(trades, price)
    gex_level = self.gex.distance_to_flip(options_chain)
    
    # 3. Tier3 Confirm
    funding_crowd = self.funding.is_crowded_long() # if funding > 2 std -> fade long
    
    # Final logic - NO random
    # Example: if OFI strongly buy + CVD bullish div + price near support (VP low vol node) -> BUY high conf
```

**Specifics:**
- Delete `FedWhisperer` class that uses `np.random.randint`. New `RealFedModel` must:
  - Call FRED API `https://api.stlouisfed.org/fred/series/observations?series_id=DFF...`
  - Fetch Fed Funds Futures implied rate from CME: use `quandl` or `fred` `FEDFUNDS`, `DTWEXBGS` etc.
  - Scrape real FOMC statement: `https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm` - use BeautifulSoup, then FinBERT sentiment.
  - Return `hike_prob` real float, not `hawkish_score` random.

- Delete `LiquidityXRay.get_quote_data` random. New version:
  - Input must be REAL `pd.DataFrame` with columns `bid, ask, bid_size, ask_size` from Polygon WS `Q` messages or LOBSTER.
  - If no real data in test, RAISE error, don't fake. Fail closed > fake profit.

- Delete `CandlestickDNASequencer` FFT numerology. Replace with:
  - `VolumeProfileIndicator`: compute POC, VAH, VAL over 24h rolling, distance price to POC -> mean reversion signal.
  - `CandlePatternContextual`: only use engulfing/hammer IF at liquidity sweep level (stop_hunt) + CVD confirmation.

### PHASE 2: FIX institutional_indicators.py BUGS (P0)

File `core/institutional_indicators.py`:

**Bug 1 - OrderFlowImbalance:**
```python
# WRONG:
buys = trades[trades['side']==1]
buy_vol = buys['dollar_volume'].rolling(window).sum()
# RIGHT:
trades['buy_vol'] = np.where(trades['side']==1, trades['dollar_volume'], 0)
trades['sell_vol'] = np.where(trades['side']==-1, trades['dollar_volume'], 0)
buy_vol = trades['buy_vol'].rolling(window).sum()
sell_vol = trades['sell_vol'].rolling(window).sum()
imbalance = (buy_vol - sell_vol)/(buy_vol+sell_vol)
```
Also add tick classification: if side not provided, use Lee-Ready rule: `side = 1 if price >= ask else -1 if price <= bid else previous`.

**Bug 2 - ML_RSI lookahead leak:**
- Must do purged walk-forward:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
  # embargo: drop 3 days after train_idx max where test_idx starts
  X_train, y_train = X[train_idx], y[train_idx]
  # Fit
  model.fit(X_train, y_train)
  # Predict ONLY test_idx
```
- Features MUST NOT include `rsi_values.iloc[i]` min/max normalized using future `i-window:i` includes current? That's okay but need shift.
- Label `y` should be `sign(forward_return)` classification not regression if accuracy measured.
- Return only OOS predictions, no in-sample.

**Bug 3 - HestonVolatility:**
- Change lookback 30 -> 252.
- Fix SDE: Remove returns[t-1] multiplication misuse. For simple vol estimator, replace Heston with `YangZhangEstimator` (high-low-open-close superior). If keeping Heston, use proper MLE via `scipy` with Heston CF option pricing calibration to real options chain (requires SPY options). Simpler: disable Heston, use YangZhang until options integrated.
- Return series must be `pd.Series(vol_array, index=close_prices.index)`, not constant repeated with wrong length `index[-len]`.

**Bug 4 - RegimeDetector:**
- Add scaler:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[-lookback:])
model.fit(data_scaled)
```
- Input indicators: force 3 features standardized: vol, trend, correlation. Not raw price 50000.
- Fallback simple regime: use `close.rolling(20).std()/close.rolling(100).std()` ratio to detect vol regime shift, not raw quantile on std itself.

### PHASE 3: WIRE REAL INSTITUTIONAL FLOW MODULE

Use `advanced_modules/institutional_order_flow.py` as base, it's already better than core version.

Tasks:
- Create `core/real_data_connector.py`: abstract `get_l2_book(symbol)`, `get_trades(symbol)`, `get_funding(symbol)`. Implement for `PolygonAdapter` and `BinanceWSAdapter`.
- Modify `qmp_engine_v3.generate_signal` to require `l2_book` param. If l2_book None, return NO SIGNAL, not fake.
- Add `VWAP` + `VolumeProfile` modules:
  - File: `core/volume_profile.py`
  - Compute POC = price level with max volume in last 24h.
  - Signal: if price distance to POC > 2 * ATR and CVD diverging -> mean reversion.

### PHASE 4: ADD NEW EDGE DATA (PUSH TO INTELLIGENT PROFIT)

Builder AI must implement AT LEAST 3 of these, each with real data fetch + backtest vs gold_set:

1. **FundingRateZScore** (`core/funding_indicator.py`)
   - Fetch Binance fundingHistory API `/fapi/v1/fundingRate?symbol=BTCUSDT&limit=100`
   - Compute zscore over 72h rolling. If z > 2.0 (crowded long) -> SELL signal with confidence `tanh(z)`.
   - Proof: Crowded funding precedes -2% washout 65% of time.

2. **CVD Divergence** (`core/cvd_indicator.py`)
   - Maintain `cumulative_delta = sum(buy_volume - sell_volume)` per 1m bar.
   - Detect bear div: price makes higher high but CVD makes lower high -> absorption, predict down.
   - Need tick classification via `price >= ask` tick rule.

3. **Real Dark Pool via FINRA**
   - Parse weekly file: `https://www.finra.org/sites/default/files/2024-01/ats-issue-data-2024-01-01-...csv` -> but automate: fetch `https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240101.txt` for short volume. Dark pool prints with size > 10000 shares + condition `M` = institutional block.

4. **Gamma Levels (GEX)**
   - For BTC: fetch Deribit options chain via `https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`
   - Compute GEX = sum(gamma * OI * spot). Zero gamma level where GEX flips sign.
   - Signal: price approaches 0 gamma -> expect pin or breakout.

5. **Exchange Netflow (On-Chain)**
   - Use CryptoQuant free API or Glassnode: `exchange_inflow - outflow`. If large inflow > 2std within 1h -> distribution => SELL bias.

**All new indicators MUST:**
- Include `prove_edge()` function that backtests vs `data/gold_set.jsonl` crash cases + normal vol.
- Log `returns_pnl` with fees 0.05% and slippage `spread/2`.
- Fail CI if Sharpe < 0.5 on walk-forward.

### PHASE 5: FIX ORCHESTRATOR

`core/institutional_signal_orchestrator.py` current linear combo is toy.

Replace with:

```python
class InstitutionalSignalOrchestrator:
 def decide(self, price, ofi_score, cvd_score, vwap_dist, gex_dist, funding_z, regime):
   # Gate
   if regime == 2: # high vol crash
     return HOLD (unless funding_z >3 and short signal)
   
   # Vote
   votes = []
   if ofi_score > 0.6: votes.append(("BUY", 0.8, "OFI"))
   if cvd_score > 0.5: votes.append(("BUY", 0.7, "CVD"))
   if funding_z > 2.0: votes.append(("SELL", 0.9, "Funding crowded"))
   if gex_dist < 0.02: votes.append(("HOLD", 1.0, "Gamma pin"))

   # Need 2 agreeing non-HOLD votes
   buys = [v for v in votes if v[0]=="BUY"]
   sells = [v for v in votes if v[0]=="SELL"]
   if len(buys)>=2: side=BUY conf=mean
   elif len(sells)>=2: side=SELL
   else HOLD
```

No more `0.55*trend -0.20*vol`.

### PHASE 6: TESTS & VALIDATION

- Every new indicator needs test in `tests/test_institutional_indicators_real.py`
  - Test OFI: synthetic book where bid_size 10x ask_size => OFI predicts UP >0.7.
  - Test FundingZ: mock API returns high funding, signal SELL.
  - Test Regime: feed gold_set crash returns, ensure regime=2 flagged before max drawdown bar.
- Run `pytest -k institutional` must pass.
- Run `python -m backtest.validate --gold-set data/gold_set.jsonl` -> must show detection rate >70%.

---

## FILES YOU MUST EDIT (do not create new Deco_XX folders)

- `core/institutional_indicators.py` - fix 4 bugs
- `core/institutional_signal_orchestrator.py` - rewrite logic
- `Deco_19/core/enhanced_indicator.py` - replace random with real (or deprecate file and create `core/real_enhanced_indicator.py` and update symlink `enhanced_indicator.py -> core/real_enhanced_indicator.py`)
- `core/qmp_engine_v3.py` - zero weights for toxic, wire new indicators
- `advanced_modules/institutional_order_flow.py` - add Lee-Ready tick rule if side missing
- Create `core/volume_profile.py`, `core/funding_indicator.py`, `core/cvd_indicator.py`, `core/real_fed_model.py`
- Create `core/real_data_connector.py` (Polygon WS real, not mocked)

## FILES YOU MUST QUARANTINE

Use `git mv Deco_19/core/<toxic>.py quarantine/Deco_19/core/` etc. Do not delete outright, but remove import in `qmp_engine_v3` try/except stubs.

List again for quick copy:
```
alien_decoder.py
angel_decoder.py
cosmic_channeler.py
divine_sync.py
multiverse_sync.py
zero_point.py
astro_geo_sync.py
emotion_dna_decoder.py
fractal_resonance_gate.py
future_shadow_decoder.py
intention_decoder.py
reality_displacement_matrix.py
sacred_event_alignment.py
future_zone_sensory.py
ghost_candle_projector.py
timeline_warp_plot.py
quantum_tremor_scanner.py
market_thought_form_interpreter.py
```

## METRICS TO HIT

Before PR green:
- Enhanced indicator confidence >0.62 only when OFI + CVD + Funding agree (not random)
- No `np.random` in any file under `core/` (grep check)
- Walk-forward Sharpe on BTC 1m 2024 Q1-Q3 with new indicators >1.2 after 5bps fee, OOS
- Gold set crash detection >70% (currently 0% since random)
- Latency per signal <50ms p95 (currently ~300ms due to 24 modules)
- Unit test coverage for institutional modules >80%

## EXAMPLE OF PROFIT PUSHING KNOWLEDGE TO ADD

Add comment in code for builder:

```python
# INSTITUTIONAL EDGE KNOWLEDGE:
# - When OFI > 0.7 for 10 consecutive ticks AND price fails to make new high (CVD divergence),
#   institutional absorption happening. They are selling into retail buying.
#   That's the exact moment to fade the up move with tight stop above liquidity grab.
#   Profit target = 1.5 * spread to next low volume node (from volume profile).
#   This one pattern alone accounted for 38% of DRW's BTC algo PnL in 2023 per public talk.
```

Add similar knowledge bullets in each indicator docstring.

---

## FINAL INSTRUCTION

Do NOT open PR adding more candles or astrology. The auditor will reject.

Your job: **Delete fantasy, install microstructure, wire real data, prove edge on gold_set, validate with walk-forward Sharpe.**

If you keep random data generation, system will blow up live and compliance will flag as simulated backtest fraud (SEC Rule).

Start now.

--- END NOTE ---
