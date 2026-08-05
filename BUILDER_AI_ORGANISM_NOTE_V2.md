# BUILDER AI NOTE V2 - ORGANISM, SELF-CODING, WHALE, LOW LATENCY

FROM: Deep Institutional Audit + Organism Architect
TO: Builder / Coding AI - This is your MUST implement list
PRIORITY: P0 - Without this, system bleeds.
DATE: 2026-08-05

Read previous notes first:
- INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md
- BUILDER_AI_ACTION_NOTE.md
- Keep those tasks, this extends with self-coding organism + pre-broker.

---

## TASK 0: VERIFY ORGANISM WORKS

Run:

```
python -c "from autonomy.organism import Organism, OrganismConfig; o=Organism(OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True)); o.discover_and_wire(); print(o.get_registered_modules().keys())"
```

Must see at least 10 modules from core, advanced_modules.

If fails, fix import errors first. Quarantine toxic modules that break discovery (move to quarantine/ and they won't be discovered).

---

## TASK 1: ADD AUTO SELF-CODING TO EACH MODULE (MANDATORY)

### 1.1 Base class already supports, but new modules ignore it

Every new institutional indicator MUST follow template in `AUTO_SELF_CODING_ORGANISM_BLUEPRINT.md` Section 6.

Checklist for each file `core/*_indicator.py`:

- [ ] `@register_module` decorator
- [ ] `module_name = "my_module"` unique
- [ ] `initialize()` subscribes to `REGIME_CHANGE, WHALE_FLOW, ALPHA_SIGNAL` via `get_event_bus()`
- [ ] `analyze(market_data)` returns `ModuleResult` and publishes `ALPHA_SIGNAL` to bus
- [ ] `on_event(event)` adjusts adaptive params when regime or whale flow changes
- [ ] `learn_from_outcome(outcome)` adjusts `confidence_floor` up on loss, down on win
- [ ] `diagnose(context)` returns issue if win_rate <0.48 or mistake_rate >0.3
- [ ] Uses only allow-listed adaptive params: `confidence_floor, weight_multiplier, lookback, cooldown_seconds, volatility_multiplier, regime_affinity_multiplier`

### 1.2 Fix self_coding allow-list (small expansion)

Edit `autonomy/self_coding.py` ApprovalPolicy.allowed_imports:
Keep forbidden `os,sys,subprocess,socket,requests,ccxt,multiprocessing,threading` blocked.
Add `collections, deque, queue, json, time, numpy, pandas, numba` explicitly allowed (most already).

Generated code must NEVER call network directly. It may only tune params that consume data from `core/real_data_connector.py` (which is protected and human reviewed).

### 1.3 Wire learning loop

Find where trades close: `execution/okx_engine.py`, `execution/event_driven_executor.py`, `Deco_19/core/alpaca_executor.py`

After close:

```python
from autonomy.learning import LearningStore
learning = LearningStore()
learning.record_outcome(prediction_id=..., module_name=..., symbol=..., pnl=realized_pnl, correct=realized_pnl>0, reason="closed", regime=current_regime)
bus.publish("MISTAKE_RECORDED" if pnl<0 else "OUTCOME_RECORDED", ..., priority=ADAPTIVE)
```

Ensure `Organism.generate_consensus_signal` already records prediction. Tie prediction_id to order id.

Test: Simulate 5 losses for ofi_detector via `learning.record_outcome(..., module_name="ofi_detector", pnl=-0.01)`, then run organism self_improvement cycle, check if it generates CodeProposal increasing confidence_floor.

### 1.4 Market improvement self-improvement

In `autonomy/organism.py`, after regime detection, add:

```python
if prev_regime == "crash" and new_regime in ["trend","range"]:
  for name, module in self.modules.items():
    if module.category in ["microstructure","trend"]:
      module.apply_adaptive_parameters({"weight_multiplier":1.0, "cooldown_seconds":1.0})
  bus.publish("REGIME_IMPROVEMENT", {"from":prev_regime,"to":new_regime}, priority=ADAPTIVE)
```

Trend modules get re-enabled quickly when market improves.

---

## TASK 2: ONE ORGANISM INTERCONNECT

### 2.1 Event Bus Lanes

Current `core/event_bus.py` has lanes. Use them correctly:

- CRITICAL (0): `KILL_SWITCH, COMPLIANCE_FAIL, SELF_DESTRUCT` - sync, must be fast, no async
- OPERATIONAL (1): `ALPHA_SIGNAL, FINAL_SIGNAL, WHALE_FLOW, MM_INTENT` - sync if possible, async fallback, target <50ms
- ADAPTIVE (2): `REGIME_CHANGE, MISTAKE_RECORDED, OUTCOME_RECORDED, WEIGHT_UPDATE` - async
- EVOLUTIONARY (3): `CODE_PROPOSAL, SHADOW_DEPLOYED, SHADOW_PROMOTED` - async, never blocks trading

Modify `OrganismConfig` to ensure:

```python
enable_coherence_protocol=True
module_mutation_cooldown_sec=180 # 60 for test
max_auto_changes_per_cycle=3
```

### 2.2 Module dependencies

Set `dependencies = ["regime_detector", "volume_profile"]` for OFI detector that needs POC to confirm.

Organism `ModuleAutoDiscovery` should topologically sort by dependencies so regime_detector initializes first.

Add simple sort in `organism.py` discover_and_wire: if module has dependencies, ensure those modules initialized before.

### 2.3 BTC/ETH/SPY one organism

Single Organism holds 3 symbols. In `generate_consensus_signal(symbol, ...)` loop over `["BTCUSDT","ETHUSDT","SPY"]`

Cross module listening: ETH module subscribes to BTC alpha:

```python
bus.subscribe("ALPHA_SIGNAL", callback where payload symbol == "BTCUSDT" and self.symbol == "ETHUSDT")
```

Spy module subscribes to ES futures alpha if using ES.

Create `core/cross_asset_risk.py` that subscribes to all ALPHA_SIGNAL and if 2 crash signals across BTC+SPY -> publish `SYSTEM_CRASH_RISK` event to reduce size globally.

---

## TASK 3: PRE-BROKER CORE DATA SYSTEM <50MS

### 3.1 Build DataRing

Create file `core/data_ring.py` per blueprint Section 2. Must be zero-copy numpy circular buffer, push in WS thread, read in analyzer.

Test: Push 100k ticks, read latest 100, latency <0.2ms.

### 3.2 Replace WebSocketStreams Queue with Ring

Edit `live_data/websocket_streams.py`:

Current:
```python
self.message_queue.put((stream,data))
```
New:
```python
from core.data_ring import get_global_ring
ring = get_global_ring(symbol) # symbol specific
# parse with orjson
bid = data['bids'][0][0] etc
ring.push(ts, bid, ask, ...)
```

- Use `orjson.loads` not `json.loads` (3x faster)
- No `time.sleep(5)` reconnect: use `time.sleep(0.1), 0.2, 0.4, 0.8, 1.6` exponential backoff max 5s
- Persistent thread per symbol, not per stream duplicate

### 3.3 Split Data vs Execution

Create:

- `live_data/binance_depth_connector.py`: Connects to `wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@trade/btcusdt@forceOrder` -> pushes to DataRingBTC
- `live_data/polygon_connector.py`: Connects to `wss://socket.polygon.io/stocks` -> pushes to DataRingSPY
- Keep `Deco_19/core/alpaca_adapter.py` ONLY for execution, DELETE its `get_bars` usage for data. Execution adapter subscribes to FINAL_SIGNAL, not raw ticks.

Document: "Data before broker" means DataRing is populated FROM EXCHANGE WS, not from broker REST. Broker only used to place order.

### 3.4 Latency instrumentation

Every ModuleResult includes `latency_ms`. Add in Organism consensus:

```python
consensus_latency = time.time() - first_tick_ts
log if >50ms
bus.publish("LATENCY_ALERT" if >100ms)
```

Benchmark: Write script `scripts/latency_bench.py` that measures from WS recv to final signal.

Goal p95 <50ms for BTC, <40ms for SPY via Polygon.

---

## TASK 4: WHALE + MARKET MAKER MODULES (Move instantly)

Build these files (use existing skeletons where possible):

**File: `core/real_data_connector.py`**
Abstracts `get_l2_book(symbol)` returning numpy view from DataRing, not REST.

**File: `core/whale_flow_detector.py`**
- Methods:
  - `fetch_exchange_netflow(symbol)` -> calls CryptoQuant API (needs API key, cache 1m)
  - `detect_stablecoin_inflow()` -> Nansen or CryptoQuant stablecoin exchange balance
  - `detect_options_block_trades()` -> Deribit public API `get_last_trades_by_currency`
  - `publish_whale_flow()` -> bus.publish("WHALE_FLOW", {"type":"distribution"/"accumulation", "size_btc":..., "zscore":...})
- Publishes to OPERATIONAL lane, confidence high.

**File: `core/mm_intent_detector.py`**
- Reads DataRing latest 100 snapshots
- Detects spoof and absorption per blueprint
- Publishes `MM_INTENT` event: `spoof_ask_pulled` bullish, `absorption_bearish` etc.

**File: `core/volume_profile.py`**
- Compute POC, VAH, VAL from 24h 1m bars or from DataRing aggregated by price bin using `np.histogram`
- Signal: distance to POC in ATR units

**File: `core/cvd_indicator.py`**
- CVD = cumulative buy_volume - sell_volume with Lee-Ready tick classification if side missing
- Divergence detection: price HH + CVD LL etc.

**File: `core/funding_indicator.py`**
- Fetch `https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=100`
- Compute zscore 72h, signal SELL if z>2.5

**File: `core/gex_indicator.py`**
- Fetch Deribit options chain `get_book_summary_by_currency`
- Compute GEX per strike, find zero gamma level

**File: `core/liquidation_map.py`**
- Build proxy heatmap from OI + price: long liq cluster = recent swing low - (entry *0.10)
- Or integrate Coinglass API if key available
- Signal sweep + rejection

All these modules must be `@register_module`, have self-coding hooks, publish ALPHA_SIGNAL.

For each, add `prove_edge()` that backtests on `data/gold_set.jsonl`:

```python
def prove_edge():
  # load gold_set
  # for each crash case, did module flag risk before?
  # Sharpe >0.5 required
```

---

## TASK 5: EXECUTION - MOVE INSTANTLY

Edit `execution/event_driven_executor.py`:

- Keep persistent WS connections for order placement (Binance keepalive listenKey)
- Pre-place limit inside spread:
```python
# on start
place_limit(symbol, side="BUY", price=mid-0.2*spread, size=small)
# on signal
if signal.confidence>0.6:
  amend_order(price=mid+0.1*spread, size=signal.size) # replace not new
```
- TWAP bypass:
```python
if whale_flow.confidence>0.85 and ofi_z>2 and final_conf>0.85:
  # bypass TWAP, market 50% now
  place_market(...)
```

- No market orders unless confidence >0.85 and latency critical

---

## TASK 6: FINAL INTEGRATION TEST

Run:

```
pytest tests/test_institutional_indicators_real.py -v
python scripts/latency_bench.py --symbol BTCUSDT --ticks 1000 (must show p95 <50ms)
python -m backtest.validate --gold-set data/gold_set.jsonl (detection >70%)
```

If latency bench fails >100ms p95, go back to numba + orjson + DataRing.

---

## TASK 7: DOCUMENT TO HAND OFF

Update `README.md` with:

- Organism diagram lanes
- DataRing pre-broker architecture
- How to add new module with self-coding
- BTC/ETH/SPY data sources

---

## WHAT TO DELETE (Remind)

- Do NOT keep random fed_whisperer, liquidity_xray random. Delete.
- Do NOT keep alien_decoder etc in core discovery path. Move to quarantine/
- Do NOT use Alpaca get_bars for data. Use Polygon.

---

## SUCCESS METRICS

Builder AI done when:

- [ ] All new institutional modules have @register_module, self-coding hooks, publish ALPHA_SIGNAL
- [ ] DataRing exists and WS pushes directly, no Queue
- [ ] Latency bench p95 <50ms BTC, <40ms SPY
- [ ] Whale flow detector publishes WHALE_FLOW events and modules adjust confidence on them
- [ ] MM intent detector detects spoof/absorption
- [ ] LearningStore records outcome -> mistake -> CodeProposal -> shadow -> promote loop works (test with forced low win_rate)
- [ ] Gold set detection >70%
- [ ] No np.random in core/ (grep check)
- [ ] Organism discovers and wires all modules as one bus
- [ ] Cross-asset: BTC leads ETH, ES leads SPY wired

---

**NOW CODE. No astrology.**

