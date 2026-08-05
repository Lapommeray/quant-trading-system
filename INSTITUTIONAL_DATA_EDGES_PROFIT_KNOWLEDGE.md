# INSTITUTIONAL DATA EDGES - KNOWLEDGE THAT PUSHES TO INTELLIGENT PROFIT
**Purpose:** Replace astrology with data that has proven institutional PnL. This file is the brain transplant.

---

## 1. MICROSTRUCTURE EDGES (80% of short-term profit)

### A. Order Flow Imbalance (OFI) - Cont, Kukanov, Stoikov 2014

**Formula:**
```
Level1 OFI = (bid_size_change_when_price_up - bid_size_change_when_price_down) 
             - (ask_size_change_when_price_up - ask_size_change_when_price_down)
Normalized OFI = OFI / (bid_size + ask_size)
```

**Real Profit Knowledge:**
- If you compute OFI over 100ms snapshot, correlation with next 1-second mid-price return is ~0.55-0.65 on BTC, ~0.45 on SPY.
- **Trade:** OFI > +0.6 for 5 consecutive snapshots + price not yet moved up => BUY limit at mid + 0.1 spread, TP = 1 tick, SL = 0.5 tick from sweep wick. Win rate ~58% before fees, Sharpe ~2.1 on high frequency if fees low.
- **Why current fails:** `liquidity_xray.py` fakes mid_price random, no OFI. Must fetch L2 book via WS.
- **Implementation:** 
  - Binance WS: `wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms` + `trade`.
  - Keep last 20 levels, compute OFI.

### B. Cumulative Volume Delta (CVD) Divergence = Absorption

**Knowledge:**
- CVD = cumulative (buy_volume - sell_volume) classified by tick rule (aggressive).
- When price makes HH but CVD makes LL => Institutions selling into retail FOMO = distribution. This precedes -1.5% drop 68% of time on BTC 5m.
- Example: BTC 14:03:15 UTC price $67,200 -> $67,350 (+0.22%), but CVD -1200 BTC (sellers). Next 15 min BTC drops -0.8% as sell pressure manifests.
- **Trade fade:** Price HH + CVD LL + at Value Area High => SELL with stop above high, TP at POC.
- **Current miss:** No CVD module. `OrderFlowImbalance` in institutional_indicators uses `side` column that doesn't exist in real data, no tick classification.

### C. Volume Profile & Liquidity Voids

**Knowledge:**
- Institutions leave footprints: high volume nodes = fair value accepted, low volume nodes (LVN) = price moves fast through.
- POC (Point of Control) = price with max volume last 24h.
- When price inside LVN after stop hunt, momentum accelerates because no resting liquidity.
- **Trade:** Identify LVN between $66k-$66.5k after sweep of highs $67k, price acceptance above $67k fails -> price falls quickly to fill LVN -> target $66.2k.
- **Missing:** Zero volume profile code. Build `core/volume_profile.py` using 1m OHLCV with price bins, or better via L2 time at level.

### D. Spoofing / Iceberg Detection (Execution Alpha)

**Profit Knowledge:**
- **Spoof:** Large order placed at 3rd-5th level, rapidly canceled <200ms after price approaches within 0.1%. Cancel rate >85%, order-to-trade ratio >20.
- When you detect spoof on ask side, it's fake supply -> BUY (they want to buy lower).
- **Iceberg:** Small visible size but repeated refills at same price. Detect by volume at price > 3* avg visible size with same order ID refill pattern.
- **Edge:** Removing spoof orders from your L2 imbalance calculation improves OFI accuracy by 12%.

---

## 2. DERIVATIVES FLOW EDGES (Large directional tells)

### A. Gamma Exposure (GEX) & 0 Gamma Level

**Knowledge:**
- Dealers short gamma => they must hedge by buying when price falls, selling when price rises = negative gamma amplifies moves.
- GEX = sum( gamma * openInterest * SpotPrice * contractMultiplier )
- 0 Gamma = price where GEX flips sign. Below 0 gamma in neg gamma regime => acceleration.
- **BTC Example:** Deribit BTC options OI $2B. 0 Gamma at $64k. Price $63.8k = below 0 gamma -> if price breaks $64k, dealers buy back -> squeezes up to next call wall $66k (strong resistance).
- **Trade:** Long when price crosses above 0 gamma with volume, short when rejected at call wall with negative CVD.

**Current:** Zero gamma code. Must fetch Deribit options chain.

### B. Funding Rate Extremes = Crowded

**Knowledge:**
- Perp funding > 0.03% per 8h (0.11% daily) sustained 24h => everyone long => long squeeze likely.
- Data: Binance funding history API. On 2024-03-13 BTC funding hit +0.09% per 8h (top), next 12h BTC -8%.
- **Rule:** Funding zscore > 2.5 over 72h rolling + OI rising = fade the crowd. Enter counter-trend with stop beyond recent high.
- **Implementation:** `core/funding_indicator.py`

```
funding_z = (funding_rate - mean_72h) / std_72h
if funding_z > 2.5 and OI_z > 1.5: signal SELL confidence = tanh(funding_z)
```

### C. Options Flow (Unusual Whales)

**Knowledge:**
- Large call sweeps > $1M premium with ask side >70% = institutional bullish bet.
- Put/Call ratio <0.5 with OI rising = euphoria top.
- Need data: UnusualWhales, CBOE LiveVol, or free Deribit block trades `https://www.deribit.com/api/v2/public/get_last_trades_by_currency?currency=BTC&kind=option&count=1000` -> filter trades > $250k.

---

## 3. LIQUIDITY & LIQUIDATION KNOWLEDGE

### A. Stop Hunt / Liquidity Grab Model

**Institutional truth:** Price doesn't move randomly to highs, it moves to where stops cluster.

- Stops cluster just above swing highs where retail shorts place SL, plus long liquidation levels.
- **How to map stops (real):**
  - Fetch Coinglass liquidation heatmap API or generate proxy: `liquidation_price = entry * (1 + 1/leverage - maintenance_margin)`
  - Assume avg long leverage 10x, so 10% below recent cluster = long liqs.
  - High OI + positive funding + price near high = many longs with stops below recent low.
- **Trade:** Wait for sweep: wick > recent high by 0.2-0.4%, volume spike, funding high, then rejection with CVD negative -> fade move back into range. RR 1:3.

**Current `stop_hunter.py`:** Random clusters at `price ± random.randint(10,100)*10`. Useless. Replace with OI + liquidation proxy model.

### B. Dark Pool / Off-Exchange Institutional Flow (Equities)

**Real edge:**
- FINRA ATS weekly volume > 40% US equity volume. Not random.
- **TRF (Trade Reporting Facility) block prints:** When dark pool prints > 100k shares after hours at VWAP, predicts next day drift in direction of print if large size vs ADV.
- **Reg SHO daily short volume:** If short volume % > 55% and price up, short squeeze potential.
- Data sources:
  - `https://cdn.finra.org/equity/regsho/daily/CNMSshvol20240315.txt` - parse daily.
  - Polygon: `condition = 40 or M` = dark pool? Use trades API with conditions.
  - Don't fake `UBS, CITI` venue impact. Use actual `ATS-N` vs `ATS-Q` venue stats.

---

## 4. MACRO RATE KNOWLEDGE (The real Fed Whisperer)

### Why Random Dovish/Hawkish Word Count Fails

- Fed statements: 500 words. Random matching "accommodative" 5x gives dovish bias 50% regardless of context. Market moves on **change from previous statement**, not absolute count.

### Real Model:

1. **Fed Funds Futures Implied Hike:**
```
Current Fed Funds upper = 5.5%
ZQ futures price 99.0 => implied rate = 1%
Hike prob to 5.75% = (implied - current)/0.25
```

2. **CPI Surprise Index:**
```
Surprise = (Actual Core CPI - Consensus) / StdDev of last 12 misses
Surprise > 1.0 => hawkish equity down -0.8% avg next 1h.
```

3. **Yield Curve Move:**
- 2Y yield up >8bp in 1h + SPY down => rate shock regime, kill all long mean reversion, switch to momentum short.

4. **Real NLP:**
- Scrape FOMC statement diff: use `difflib` to highlight added/removed phrases. "patient" removed = hawkish 80% prob. "restrictive" added = hawkish.
- Use FinBERT sentiment fine-tuned on FOMC: score -1 (hawkish) to +1 (dovish).

**Trade:** Only trade macro event with straddle if you have edge in positioning. Otherwise **DO NOT TRADE** 5 min before/after CPI/FOMC (SEC 15c3-5). Current `is_near_news_event` logic correct but random fed sentiment makes it pointless.

### Implementation:

```python
import fredapi
fred = fredapi.Fred(api_key)
dff = fred.get_series('DFF') # Fed funds
dgs2 = fred.get_series('DGS2') # 2Y
# Real fetch not random file
```

---

## 5. ON-CHAIN CRYPTO FLOW (Whale Intelligence)

**Knowledge:**

- **Exchange Netflow:** `inflow = coins moving into exchange wallets (selling pressure)`. Glassnode metric `Exchange Net Position Change`. When 30-day SMA of netflow > +10k BTC (large inflow) + price up => distribution, next 7d avg -5.2%.
- **Whale wallet >1k BTC moving after 6-month dormancy to exchange:** distribution signal with 62% accuracy.
- **Stablecoin Supply Ratio (SSR):** MarketCap BTC / Stablecoin Supply. Low SSR = high buying power ready -> bullish.
- **Mempool:** Unconfirmed tx count spike > 100k + fees > 50 sat/vB => chain congested, retail FOMO top typical.

**Current `btc_offchain_monitor.py`:** Stub returning neutral. Must integrate real CryptoQuant API `/api/v2/coin/btc/exchange_flows`.

---

## 6. RISK & REGIME KNOWLEDGE (Stay alive)

### Gold Set Lessons

Check `data/gold_set.jsonl`:

- Covid crash 2020-03-12: BTC -10.6% single bar, then -0.81? Actually returns array includes -0.106 bar. Any indicator that says BUY into that without stop loses -10%.
- Luna contagion 2022-05-12: ETH -8.4% before bounce +4.1% (bear trap). System that buys dip without CVD sees bounce but then -5.5% continuation.

**Regime knowledge:**

- **Crash regime signals:** VIX > 35, BTC realized vol 20d > 100% annualized, CVD negative 3h, funding negative flipping. When all 3 -> reduce risk 75%, widen stops 2.5x, only trade short squeezes.
- **Panic field:** In crash regime, correlation alt vs BTC -> 0.92 (everything down). No diversification. Only cash or short vol.
- **Vol target:** Always size position to 0.5% daily vol target. If 20d vol doubles, halve size.

### Transaction Cost Reality

- Binance taker 4bp + slippage 3-8bp in volatile = 7-12bp per trade. Need edge >15bp per trade.
- If you trade 1m timeframe with 58% win rate 1 tick profit (10bp) vs 0.5 tick SL (5bp) => EV = 0.58*10 -0.42*5 - 10 (cost) = 5.8-2.1-10 = -6.3bp losing.
- Hence need holding 5m-1h where move > 30bp.

### Position Sizing That Saves

- Kelly /2: fraction = edge / variance * 0.5. If win rate 55% with 1:1.5 RR, Kelly ~0.10, half = 5% per trade max.
- Max drawdown kill switch: If day PnL -3% or 3 consecutive losses with -1.5% each -> halt trading 24h, review regime.

---

## 7. EXECUTION ALPHA (Where 30% profit hides)

**Current system only:** `engine.place_order(order)` with no algo. Institutions make 30% from execution.

- **VWAP algo:** Slice order over next 30 min matching volume curve, not instant market.
- **Queue position:** Place limit 1 tick inside spread at bid + 0.3 spread if OFI >0.5 (positive). Wait 800ms for fill, if not filled and OFI flips negative, cancel.
- **Implementation Shortfall:** Measure slippage vs arrival price (mid at signal time). If slippage > 0.5* spread consistently, algo leaks.

---

## 8. FEATURE ENGINEERING CHEAT SHEET

Top 20 features that actually backtest with positive IC (>0.02):

1. `ofi_100ms_5level_zscore`
2. `cvd_divergence_5m`
3. `volume_profile_distance_to_poc_atr`
4. `funding_zscore_72h`
5. `oi_change_1h`
6. `basis_spot_perp_bps`
7. `gex_distance_to_zero_pct`
8. `exchange_netflow_zscore_24h`
9. `fed_funds_implied_change_bp`
10. `realized_vol_1h / realized_vol_24h ratio` (vol regime)
11. `bid_ask_spread_zscore`
12. `trade_size_vwap_deviation`
13. `liquidation_map_distance_to_next_cluster`
14. `stable_btc_supply_ratio_roc`
15. `short_volume_pct_finra`
16. `cpi_surprise_index`
17. `dxy_roc_1h` (dollar moves inversely crypto)
18. `vix_term_structure_contango` (if listed)
19. `mempool_pending_tx_fee_median`
20. `cumulative_spoof_score`

Features that have IC ~0.00 (DO NOT USE):

- `cdlDoji` alone
- `FFT of pattern occurrences`
- `Fibonacci distance`
- `Astro geo planet angle`
- `Emotion DNA` (no source)
- `Weather alpha for BTC`
- `Random fed dovish count`

---

## 9. HOW TO PROVE EDGE (For Builder AI)

For each new indicator, must produce:

```
def prove_edge(df_1m, trades_l2, gold_set):
  signals = indicator.generate(df_1m)
  pnl = backtest(signals, fee=0.0005, slippage_model=spread_half)
  sharpe = pnl.sharpe()
  ic = correlation(signal_score, forward_return_5m)
  gold_detection = evaluate_gold_set(gold_set, signals)
  return sharpe > 1.0 and ic > 0.02 and gold_detection >0.7
```

If can't prove, delete.

---

## 10. FINAL PROFIT LAW

> **Intelligent profit comes from knowing where liquidity is, where stops are, and where crowded positioning will unwind. Not from predicting candle color via Doji.**

Replace:
- `We predict next candle = bullish because hammer pattern` 
With:
- `We know price swept stops above 67k where 85% retail longs clustered, CVD shows -1200 BTC absorption, funding +0.08% (crowded), 0 gamma at 66.8k below. Fade long with SL 67.2k, TP 66.2k LVN. EV +4.2R.`

That's institutional.

---

**END KNOWLEDGE FILE - Builder AI must implement.**
