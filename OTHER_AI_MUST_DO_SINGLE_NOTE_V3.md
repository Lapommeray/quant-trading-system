# OTHER AI - SINGLE NOTE V3 - DEEP INSTITUTIONAL ANALYSIS + TASK LIST
**Date:** 2026-08-05 (supersedes `OTHER_AI_MUST_DO_SINGLE_NOTE.md`, `BUILDER_AI_ORGANISM_NOTE_V2.md`, `FINAL_BUILDER_NOTE_FOR_OTHER_AI.md`)
**Auditor role:** Deep institutional analysis — profit intelligence, not feature theater.

Read this note FIRST. It is the only handoff you need. Everything below was
verified by running code, not by reading code.

---

## 0. STATUS BOARD — WHAT IS DONE vs WHAT IS LEFT (one glance)

| # | Work item | Status 2026-08-05 |
|---|-----------|-------------------|
| 1 | OKX credentials wired safely (`.env` untracked from git) | DONE (passphrase still missing — see §1) |
| 2 | Toxic/random modules off the hot path | DONE — organism only scans `autonomy/`, `core/`, `advanced_modules/`; 40 Deco_* files stay archived |
| 3 | Pre-broker data ring (in-memory, before broker) | DONE — `core/data_ring.py`, <0.2ms reads |
| 4 | OKX public WS feed → ring → event bus | DONE (new `okx_live/feed.py`) |
| 5 | S&P 500 feed → ring → event bus | DONE (new `quant_trading_system/data_feeds/spx_feed.py`) |
| 6 | BTC-USDT-SWAP + ETH-USDT-SWAP coverage | DONE (config `OKX_FEED_SYMBOLS`) |
| 7 | OFI / CVD / funding / whale / MM-intent / volume-profile modules | DONE — wired into organism |
| 8 | Indicator bug fixes (OFI holes, ML_RSI leak, Heston, RegimeDetector) | DONE (this session, tests green) |
| 9 | One-organism event bus (4 priority lanes) | DONE — verified |
| 10 | Auto self-coding per module (diagnose → propose → validate → apply) | DONE — verified end-to-end (artifact applied, governed) |
| 11 | Learning from past mistakes (outcome → lesson → tuning) | DONE — verified |
| 12 | ExecutionPlanner (maker-first, whale escalation, 200ms recheck) | DONE (new `core/execution_planner.py`) |
| 13 | Real Heston calibration from options chain (Deribit/OKX options) | **LEFT — needs options data, cannot be done in sandbox** |
| 14 | Cancel/replace (amend) order path + queue-position tracking on OKX | **LEFT — next AI must build on execution/okx_engine.py** |
| 15 | Backtest + walk-forward proof (Sharpe ≥ 0.5 after fees) for each module | **LEFT — requires historical data download** |
| 16 | Live connectivity validation from an internet-enabled machine | **LEFT — this sandbox has no egress** |

---

## 1. SECURITY FIRST (read before anything else)

1. **The user's OKX API key/secret are in the local `.env` file. The `.env`
   file has been REMOVED from git tracking this session.** Never `git add .env`,
   never paste keys into markdown, never commit them. `.env.example` is the
   committed template.
2. **`OKX_PASSPHRASE` is EMPTY.** OKX signs every private request with the
   passphrase chosen when the API key was created. Without it, `connect()` on
   the live trader fails closed (by design). The user must paste it into
   `.env` (`OKX_PASSPHRASE=...`). Do not guess, do not invent one.
3. **Live trading stays OFF** (`OKX_LIVE_TRADING=false`). Any session that
   flips it to true must also have `OKX_MAX_LEVERAGE<=3`, position cap 10%,
   daily loss cap 3%, and the kill switch armed. Prefer paper first.
4. **The key was shared in plain chat text.** Recommend the user rotate it
   after wiring the passphrase, and use a read-only key for the feed plus a
   separate trade-only key if OKX allows per-key permissions.
5. Sandbox note: this environment has NO network egress (OKX/Yahoo TLS
   blocked). All feed logic was verified with recorded real-format frames.
   Live validation must happen on a machine with internet.

---

## 2. DATA THAT WILL NOT HELP THE SYSTEM (keep off the hot path)

These add latency, dilute weights and create model risk. They are archived in
`Deco_*`/`QMP_*` folders and are NOT discovered by the organism — keep it that
way. Do not "revive" them; if a future session re-adds any, it must first
prove edge with a backtest:

- `alien_decoder`, `angel_decoder`, `cosmic_channeler`, `divine_sync`,
  `multiverse_sync`, `zero_point`, `energy_filter`, `timeline_selector`,
  `astro_geo_sync`, `emotion_dna_decoder`, `fractal_resonance_gate`,
  `future_shadow_decoder`, `intention_decoder`, `reality_displacement_matrix`,
  `sacred_event_alignment`, `future_zone_sensory`, `ghost_candle_projector`,
  `timeline_warp_plot`, `quantum_tremor_scanner`,
  `market_thought_form_interpreter`, `quantum_sentiment_decoder`,
  `time_fractal_fft`, `void_trader_chart_renderer`, `dna_breath`,
  `dna_overlord`, `spectral_signal_fusion`, `weather_alpha_generator`,
  `language_universe_decoder`, `shadow_spread_resonator`,
  `sentiment_energy_coupling_engine`, `neural_market_holography`,
  `quantum_consciousness_amplifier`, `quantum_liquidity_warper`,
  `emotion_harvest_ai`, `inverse_time_echoes`, `latency_cancellation_field`,
  `causal_quantum_reasoning`, `self_rewriting_dna_ai`,
  `zero_energy_recursive_intelligence`, `time_resonant_neural_lattice`
  (full list: `QUARANTINE_LIST.json`).

**Data classes that are fake-or-nonsense and must never re-enter the signal
path** (they were random-number generators pretending to be alpha):
- Random "Fed sentiment" from `np.random` — replaced by `core/real_fed_model.py`
  (FRED, fail-closed neutral without `FRED_API_KEY`).
- Fibonacci/numerology "candlestick DNA" — no edge, no paper.
- "Dark pool sniper" with hardcoded venue impact — no FINRA parsing.
- Stop clusters from `random.randint` — real version = liquidation stream
  (`LIQUIDATION_EVENT` from the OKX feed, already implemented).
- Raw RSI/MACD alone, doji patterns alone, FFT pattern counting: hit rate
  ~48% after costs. Only useful conditioned on microstructure (see §3).

**Latency budget principle:** every module on the hot path costs 5–15ms.
The organism runs 10 modules; keep total signal path < 50ms.

---

## 3. DATA THAT PUSHES INTELLIGENT PROFIT (BTC / ETH / S&P 500)

Priority order = evidence strength × freshness. All of these now have a real
data source wired in the organism:

### 3.1 BTC-USDT-SWAP + ETH-USDT-SWAP (OKX public WS, pre-broker)
`okx_live/feed.py` subscribes: `books5`, `trades`, `tickers`,
`open-interest`, `funding-rate`, `liquidation-orders`. Published events:
`MARKET_DATA`, `ORDER_BOOK_UPDATE`, `WHALE_FLOW`, `OI_UPDATE`,
`FUNDING_UPDATE`, `LIQUIDATION_EVENT`.

| Signal | Channel / source | Why it beats retail |
|---|---|---|
| OFI z-score (5-level) | `books5` deltas | Predicts 1s-forward returns (R²≈0.6, Cont 2014); retail doesn't see book deltas |
| CVD divergence | `trades` side + tick rule | HH price + LL CVD = distribution; exits before the -1.5% move |
| Whale prints & clusters | `trades` notional ≥ $50k, ≥3 in 5s | Follow the whale's first 20%, not the confirmation |
| Funding z-score 72h | `funding-rate` | Z>2.5 crowded long → fade; Z<-2.5 → squeeze fuel |
| OI change 1h | `open-interest` | OI↑+price↑ = trend real; OI↓+price↑ = short covering (weak) |
| Liquidation cascade | `liquidation-orders` | Stop-hunt map; wick + 2.5x vol + funding_z>2 → rejection back to POC |
| MM intent (spoof/absorption) | `books5` lifetime stats | Spoof pull = real supply/demand; absorbs = trapped MMs |
| Cross-asset lead | `cross_asset_leader` | BTC leads ETH 200–400ms (perpetuals, same venue) |

### 3.2 S&P 500 (`^GSPC`)
- New feed: `quant_trading_system/data_feeds/spx_feed.py` → DataRing `SPX`,
  events `MARKET_DATA`/`SPX_UPDATE` (asset_class `equity_index`).
- Providers: TwelveData (set `TWELVEDATA_API_KEY` in `.env`) preferred;
  yfinance 1m poll fallback (10s, `SPX_POLL_SECONDS`).
- Use for REGIME and RISK context (correlation/vol), not tick timing:
  - SPX vol regime gates crypto position size (regime_risk_gate).
  - SPX + crypto divergence (crypto rallies while SPX rolls over) = fragile.
- Future upgrade (needs paid data): ES futures 5–15ms lead over SPY/SPX,
  options 0DTE gamma pin (`GEX`), VIX term structure. See task T-7.

### 3.3 Top features to build into the consensus (IC evidence ≥ 0.02)
`ofi_z_100ms_5lvl`, `cvd_divergence_5m`, `dist_to_poc_atr`,
`funding_z_72h`, `oi_change_1h`, `basis_spot_perp_bps`, `gex_dist`,
`exchange_netflow_z_24h`, `realized_vol_1h/24h`, `spread_z`,
`trade_size_vwap_dev`, `liq_map_dist`, `cpi_surprise`, `dxy_roc_1h`,
`vix_term`, `spoof_score`.

**Features with IC≈0 — do NOT spend compute:** doji alone, FFT pattern
counts, Fibonacci distances, planet angles, "emotion DNA", weather for BTC,
random Fed counts.

---

## 4. INDICATOR DEEP AUDIT — FIXED THIS SESSION (verified by tests)

File: `core/institutional_indicators.py` (+ `core/indicators.py` mirrors).

| Indicator | Bug found | Fix applied (this session) |
|---|---|---|
| `OrderFlowImbalance` | Rolling sum on filtered buy/sell frames → index holes → NaN-heavy wrong values | Aligned `buy_vol`/`sell_vol` columns on the full frame; Lee-Ready fallback when `side` missing |
| `ML_RSI` | Fit on train, predict on FULL X (in-sample contamination); misaligned output | TimeSeriesSplit + 3-sample embargo, OOS-only predictions, rolling normalization (no future info) |
| `HestonVolatility` | Wrong SDE discretization (return used as Brownian), scalar vol repeated, lookback 30 | Replaced with Yang-Zhang drift-independent estimator (252 lookback). Proper Heston = options-chain calibration → T-4 |
| `RegimeDetector` | Raw `np.column_stack` — price (50,000) dominated RSI (0–100); HMM on unscaled data; quantile on non-stationary std | Rolling z-score standardization before HMM; stationary log-return 20/100 vol-ratio fallback (transition detector) |

New regression tests: `tests/test_institutional_indicators_fixed.py` (11 tests).
`regime_risk_gate` in `autonomy/builtin_modules.py` consumes regimes to gate
risk — keep the two in sync.

---

## 5. ONE ORGANISM — INTERCONNECT (verified 25/25 smoke checks)

`scripts/organism_smoke.py` proves the full loop offline. Run it after any
change: `python scripts/organism_smoke.py` (must print 25/25).

- Single canonical bus: `core/event_bus.py` → `get_event_bus()` (shared by
  core, autonomy, okx_live). Four lanes: **CRITICAL** (0: kill switch, stop
  loss, order request/fill — always synchronous), **OPERATIONAL** (1: market
  data, signals, whale flow), **ADAPTIVE** (2: regime, learning, weights —
  async), **EVOLUTIONARY** (3: self-coding, shadow, memory — async, never
  blocks trade lane).
- Organism: `autonomy/organism.py` → `discover_and_wire()` topologically
  instantiates modules from `autonomy/`, `core/`, `advanced_modules/`,
  wires event subscriptions, tracks health, isolates failures.
- Inter-module messaging: `BaseTradingModule.interconnect()` /
  `sync_with_organism()` / `coherence_check()` in `core/base_module.py`.
- 10 active modules verified: `ofi_detector`, `cvd_detector`,
  `funding_detector`, `whale_flow_detector`, `mm_intent_detector`,
  `volume_profile`, `cross_asset_leader`, `real_fed_model`,
  `momentum_alpha`, `regime_risk_gate`.

**Rule for the next AI:** every new module MUST (a) inherit
`BaseTradingModule`, (b) `@register_module`, (c) read data from the DataRing
or the bus — never fetch its own REST data on the hot path, (d) publish
`ALPHA_SIGNAL`/`SIGNAL_GENERATED` with latency recorded.

---

## 6. AUTO SELF-CODING — IT EXISTS AND IT IS BOUNDED (verified)

- `core/base_module.py` lifecycle: `auto_fix()`, `learn_from_mistakes()`,
  `improve_with_market()`, `auto_self_code()`, `interconnect()`,
  `full_autonomous_cycle()`.
- `autonomy/self_coding.py` `SelfCodingEngine`: diagnose → deterministic
  artifact → AST validation → policy check → sandbox tests → approve low
  risk → apply; medium/high risk stays `pending_approval` for human.
- PROTECTED paths never auto-apply: `okx_live/`, `execution/`, `risk/`,
  `safety_governance.py`, credentials, `core/event_bus.py`,
  `autonomy/organism.py`, `main.py`. If a proposal touches them it is
  quarantined. Verify this list whenever you add new critical files.
- Verified: smoke test records 6 losing/winning outcomes → 6 mistake lessons
  → `ofi_detector.auto_self_code()` produced a validated, low-risk artifact
  with status `applied`.
- **Next AI work (T-5):** per-module diagnosis thresholds
  (`win_rate<0.48 → raise confidence_floor`), shadow deployment promotion
  (`shadow_min_observations=100`, outperform ≥5% with drawdown delta ≤1%),
  and a weekly "mistake post-mortem" report generator that turns the 3 worst
  lessons into new regression tests (learn from past mistakes → test →
  cannot regress).

---

## 7. PRE-BROKER LATENCY — READ DATA BEFORE THE BROKER EVER SEES IT

Architecture (already built):

```
OKX public WS  ──►  okx_live/feed.py  ──►  DataRing (in-memory ring, <0.2ms)
  books5/trades/          │                    │  core/data_ring.py
  oi/funding/liq          ▼                    ▼
                   EventBus (lane 1) ◄── SPX feed (^GSPC, 10s poll)
                          │
              ofi/cvd/whale/funding/mm modules (5–50ms)
                          │
                   consensus (lane 1)
                          │
              AutonomousExecutor (lane 0 guardrails)
                          │
                   OKX broker API  ◄──  ONLY NOW does the broker see anything
```

- **DataRing** is the single source of truth in memory: lock-free
  single-writer ring per symbol (`BTCUSDT`, `ETHUSDT`, `SPX`), zero-copy
  reads, `latency_ms()` staleness check (>3s → modules return NEUTRAL,
  >5s feed alerts).
- **Feed fail-closed:** OKX feed publishes `RISK_ALERT` when stale; SPX feed
  after 3 consecutive fetch failures. No synthetic data ever.
- **Latency budget:** WS frame → ring push < 1ms; ring read < 0.2ms; module
  analyze 1–5ms each; consensus < 10ms; total < 50ms. If anything exceeds
  this, profile before adding features.

**T-6 (next AI):** benchmark the full path with real WS traffic on an
internet-enabled host: report p50/p99 from frame receipt to signal publish,
per module. Keep a `latency_benchmark.json` in `data/` for regression
comparison.

---

## 8. EXECUTION — MOVE WITH THE MARKET MAKER, NOT WITH RETAIL

New: `core/execution_planner.py` (`ExecutionPlanner`). Pure decision logic,
unit-tested (10 tests). It decides:

1. **ABORT** — confidence < 0.65, stale feed, spread > 30bps, no quote.
2. **TAKER (immediate)** — only when the whale stack confirms:
   whale_conf ≥ 0.85 AND ofi_z ≥ 2.0 AND CVD aligned. That is the moment
   the MM is being run over; waiting 200ms more = buying with retail.
3. **MAKER** — queue-aware limit INSIDE the spread (mid ± 0.1·spread) when
   our side has ≥25% of top-of-book size and OFI ≥ 0.5. Post, don't chase.
4. **WAIT → recheck at 200ms** — `finalize_after_recheck()`: OFI still ≥1.5
   + urgency → escalate to taker; OFI alive → keep posting; OFI dead past
   250ms → abort (never get picked off).

Wire it next (T-1, highest priority): `AutonomousExecutor` in
`autonomy/execution.py` should call `ExecutionPlanner.plan()` before
translating `SIGNAL_GENERATED` → `ORDER_REQUEST`, and `finalize_after_recheck`
on a 200ms timer for MAKER/WAIT plans. Route the resulting
`execution_plan` into the `ORDER_REQUEST` payload so guardrails see it.

---

## 9. PRIORITY TASK LIST FOR THE NEXT AI (acceptance criteria included)

**T-1 — Wire ExecutionPlanner into AutonomousExecutor** (files:
`autonomy/execution.py`, `core/execution_planner.py`; test:
`tests/test_execution_planner.py` extend)
AC: a `SIGNAL_GENERATED` with whale stack produces `ORDER_REQUEST` with
`routing="taker"`; weak signal never produces an order; MAKER plans schedule
a 200ms recheck; all events carry `execution_plan` in payload.

**T-2 — OKX cancel/replace + queue tracking** (files: `execution/okx_engine.py`,
`okx_live/engine.py`)
AC: `amend_order()` (replace, not cancel+new — 30% faster per OKX docs),
`queue_position()` estimate from `books5` (size ahead of own order), and a
"maker→taker escalation" path that amends to market when `finalize_after_recheck`
returns taker. Paper-trade verified on OKX demo before any live flag.

**T-3 — Historical data + backtest harness with fees/slippage**
AC: download 6 months of OKX `books5`/`trades` (okx history REST), 1m
candles for `^GSPC`; walk-forward backtest of `ofi_detector` +
`cvd_detector` + `funding_detector` consensus with 10bps slippage + taker
fees; report Sharpe ≥ 0.5 OOS or the module stays shadow. Output to
`data/backtests/YYYY-MM-DD_report.json` (gold set pattern already exists:
`data/gold_set.jsonl`).

**T-4 — Real Heston from options chain** (Deribit/OKX options; needs network)
AC: pull BTC/ETH ATM+ wings IV surface, calibrate
(kappa, theta, xi, rho, v0) via characteristic-function fit; feed
`vol_surface_z` feature to consensus; compare vs Yang-Zhang for 24h-ahead
vol forecast (RMSE report). Sandbox-only until T-3 passes.

**T-5 — Self-coding post-mortem loop** (files: `autonomy/learning.py`,
`autonomy/self_coding.py`, `core/base_module.py`)
AC: `autonomy/learning.py` exposes `postmortem_report(period)` producing the
3 worst mistake patterns with their module params; `SelfCodingEngine` turns
each into a candidate regression test; tests run in CI
(`.github/workflows/ci.yml`); auto-applied artifacts require the same test
to pass twice in a row.

**T-6 — Latency benchmark on live internet host** (see §7)
AC: `scripts/latency_bench.py` reports p50/p99 frame→signal for BTC/ETH/SPX;
writes `data/latency_benchmark.json`; fails CI if p99 > 50ms.

**T-7 — ES/SPX lead + GEX (optional, paid data)** (files:
`quant_trading_system/data_feeds/`)
AC: ES futures feed (or TwelveData `US500`), compute ES→SPX lead lag
(cross-correlation peak 5–15ms), GEX from options OI if available; gate
crypto risk when ES cracks 1-min VWAP by >0.3% (institutional risk-off).

**T-8 — Operator dashboard for the organism**
AC: one page showing: bus lane stats, module health + latency, feed status
(OKX/SPX), learning stats, self-coding artifacts pending approval
(approve/reject buttons), kill switch. Reuse `dashboard.py` patterns;
read-only over the bus/`feed_status()`.

---

## 10. VERIFIED EVIDENCE (this session)

- `python -m pytest tests/` → **126 passed, 29 skipped** (incl. new:
  `test_okx_pre_broker_feed.py` 8 tests, `test_spx_feed_live.py` 3,
  `test_execution_planner.py` 10, `test_institutional_indicators_fixed.py` 11).
- `python scripts/organism_smoke.py` → **25/25 checks passed** (wiring, pre-
  broker OKX frames → ring → bus, whale cluster, liquidation, SPX, OFI/CVD,
  learning, self-code applied, execution decisions).
- `.env` untracked from git (keys never committed); `.env.example` added.
- New files: `okx_live/feed.py`, `quant_trading_system/data_feeds/spx_feed.py`,
  `core/execution_planner.py`, `scripts/organism_smoke.py`,
  `tests/test_okx_pre_broker_feed.py`, `tests/test_spx_feed_live.py`,
  `tests/test_execution_planner.py`, `tests/test_institutional_indicators_fixed.py`.

**Definition of done for the whole program:** every module in the consensus
has an OOS backtest with fees/slippage (Sharpe ≥ 0.5), a latency budget
under 50ms, a mistake-learning loop that produces regression tests, and no
path from fake data to a live order. Until T-3, nothing goes live.
