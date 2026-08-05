# AUTO SELF-CODING ORGANISM BLUEPRINT - ONE BRAIN, SELF-FIXING, WHALE-ALIGNED

Date: 2026-08-05
Goal: Each module auto fixes, creates code, approves on its own, learns from past mistakes, improves when market regime improves, interconnects as ONE ORGANISM. Move with whales/market makers, not retail.

---

## 1. CURRENT ORGANISM STATUS - WHAT EXISTS

**Good Foundation Found:**

- `autonomy/organism.py` - Organism class already exists with:
  - `BaseModule(BaseTradingModule)` with `module_name, category, version, dependencies`
  - `ModuleAutoDiscovery` discovers decorated modules in packages `autonomy, core, advanced_modules`
  - `OrganismConfig`: `self_coding_enabled=True, auto_approve_low_risk=True, auto_apply_low_risk=True, self_code_each_cycle=True, max_auto_changes_per_cycle=3, module_mutation_cooldown_sec=180`
  - `LearningStore` records `prediction -> outcome -> mistake` with lesson
  - `ShadowManager` deploys candidate in shadow, compares vs live, promotes if `min_observations=100, min_outperformance=0.05, max_drawdown_delta=0.01`
  - `ConsensusEngine` weighted voting `votes[BUY]+= weight*confidence`
  - `EventBus` with priority lanes: CRITICAL=0, OPERATIONAL=1, ADAPTIVE=2, EVOLUTIONARY=3, with async worker queue

- `core/base_module.py` already has hooks:
```python
def learn_from_outcome(self, outcome): -> no-op to override
def diagnose(self, context): return diagnostic record
def apply_adaptive_parameters(self, parameters): allow-listed adaptive namespace only
def self_improve(self, performance_history)
def self_code(self, coder, context, apply=True)
```

- `autonomy/self_coding.py` implements bounded self-coding:
  - `CodeProposal` with risk LOW/MEDIUM/HIGH/CRITICAL, status GENERATED->VALIDATED->APPROVED->APPLIED
  - `SafeCodeValidator` AST allow-list, max_bytes 64k, max_lines 200, complexity 10, forbidden imports `os, sys, subprocess, socket, requests, ccxt`
  - `ApprovalPolicy`: auto_approve only LOW risk adaptive params, protected paths tokens `okx_live, execution, risk, safety, credential, secret, organism, event_bus, main.py` never auto-approved
  - Generated tests required, shadow deployment before live

**GAPS (Why not intelligent profit yet):**

1. **Modules don't talk to each other.** Each `analyze()` gets `market_data: Dict` but no lateral communication. `order_flow_hunter` doesn't know `funding_zscore` is crowded. No cross-module memory.
2. **Self-coding allow-list too restrictive.** `allowed_imports = numpy, pandas, ta, scipy, datetime` - Blocks `ccxt, requests, websocket` needed for real data adapters. So self-coder cannot fix `polygon_adapter` or `websocket_streams`.
3. **Learning store not fed per tick.** `record_outcome` called manually from execution, not automatic per bar. No online learning.
4. **Regime detection not wired to self-coding.** `MarketRegimeDetector` exists but doesn't trigger mutation cooldown or weight adjustment fast enough. In crash regime, system still runs mean reversion modules that should be isolated.
5. **No pre-broker core data bus.** Current `live_data/websocket_streams.py` uses `websocket.WebSocketApp` with `time.sleep(5)` reconnect and `Queue()` dispatch - latency ~80-150ms vs needed 5-15ms. No zero-copy ring buffer. Data goes to callback, then to `EventBus`, then to consensus - 3 hops.
6. **BTCUSD/ETHUSD/SP500 not unified.** BTC uses Binance WS, SPY uses Alpaca REST bars `1Min` (REST = 200ms+ latency). No L2 for SPY. No futures for ES. So cannot move with market maker for SPX.
7. **No whale hooks.** No mempool, no exchange netflow, no options block detector wired to event bus.

---

## 2. ONE ORGANISM ARCHITECTURE - EVERY MODULE CONNECTED

### Goal: Single Event Bus, 4 Lanes, All Modules As Neurons

```
LANE 0 - CRITICAL (0-5ms deadline, sync, no async)
  - Kill switch, compliance firewall, self_destruct_protocol, position limits
  - Pre-trade check: if fails, drop signal immediately, no further processing

LANE 1 - OPERATIONAL (5-50ms deadline, sync where possible, async fallback)
  - Core alpha modules: institutional_order_flow, cvd, volume_profile, funding, gex, liquidations, whale_flow
  - Each publishes ModuleResult to bus: event_type="ALPHA_SIGNAL"
  - ConsensusEngine subscribes to ALPHA_SIGNAL, computes final vote
  - Latency budget: OFI 5ms, CVD 10ms, funding 1ms (cached), gex 5ms (cached), consensus 5ms = 25ms total

LANE 2 - ADAPTIVE (50ms-1s, async)
  - LearningStore: consumes OUTCOME event, records mistake if pnl<0
  - Adaptive weight tuner: adjusts module weights based on regime win_rate
  - Regime detector: publishes REGIME_CHANGE event with label crash/trend/range/vol

LANE 3 - EVOLUTIONARY (1s+, async, never blocks trade)
  - SelfCodingEngine: consumes MISTAKE event, diagnoses module, generates CodeProposal
  - ShadowManager: deploys proposal to shadow clone, observes 100 ticks
  - If shadow outperformance >5% and DD delta <1%, promote
```

**Interconnect pattern: Every module implements:**

```python
class MyInstitutionalModule(BaseTradingModule):
  module_name = "ofi_detector"
  dependencies = ["volume_profile", "regime"]  # Declare dependencies
  
  def initialize(self):
    bus = get_event_bus()
    bus.subscribe("REGIME_CHANGE", self.on_event)  # Listen to regime
    bus.subscribe("ALPHA_SIGNAL", self.on_event)   # Listen to other alphas (cross-pollination)
    bus.subscribe("WHALE_FLOW", self.on_event)     # Whale detector output
    return True
  
  def analyze(self, market_data):
    # Core logic
    result = ModuleResult(module_name=self.module_name, signal=..., confidence=..., features={...})
    # Publish to organism
    get_event_bus().publish("ALPHA_SIGNAL", result.to_dict(), source=self.module_name, priority=EventPriority.OPERATIONAL)
    return result
  
  def on_event(self, event):
    if event.event_type == "REGIME_CHANGE":
      if event.payload["label"] == "crash":
        self.config["adaptive"]["weight_multiplier"] = 0.2  # reduce self in crash
    if event.event_type == "WHALE_FLOW":
      if event.payload["flow"] == "exchange_inflow_large":
        # Whale deposit = imminent sell pressure, boost confidence if I'm SELL
        ...
  
  def learn_from_outcome(self, outcome):
    # Called automatically by Organism after trade closes
    if outcome["pnl"] < 0:
      # Mistake recorded to LearningStore automatically
      # Adjust internal threshold
      self.config["adaptive"]["confidence_floor"] += 0.02
      return {"learned": True, "lesson": "Raised confidence floor after loss in regime "+outcome["regime"]}
    return {"learned": False}
  
  def diagnose(self, context):
    stats = context.get("stats") # win_rate, avg_pnl, mistake_rate from LearningStore
    if stats["win_rate"] < 0.48:
      return {"issue": "low_win_rate", "suggestion": "tighten OFI threshold from 0.6 to 0.7"}
    return {"issue": "none"}
  
  def self_code(self, coder, context, apply=True):
    # Bounded coder generates low-risk adaptive param change artifact
    # Example: coder proposes {"confidence_floor": 0.72}
    return super().self_code(coder, context, apply)
```

**Key: Organism `discover_and_wire` already does this, but must be enforced for ALL new institutional modules.**

### Organism Wiring for BTCUSD / ETHUSD / SP500

- Single `Organism` instance holds 3 symbols: `BTCUSDT, ETHUSDT, SPY` (or `ES` futures via IBKR)
- Each module gets `symbol` param in `generate_signal(symbol, history)`
- Modules that are crypto-specific (funding) return NEUTRAL for SPY; SPY-specific (Reg SHO short vol) return NEUTRAL for BTC
- But they share bus: e.g., BTC whale inflow can publish `CRYPTO_STRESS` event -> SPY mean reversion module reduces size

---

## 3. AUTO SELF-CODING IN EACH MODULE - SPEC

### Current Self-Coding Policy (Safe but slow)

- Generates file in `strategies/evolved/module_name_timestamp.py` or shadow artifact
- Validation: syntax, AST complexity <10, no forbidden imports, max 200 lines
- Tests: auto-generates pytest, must pass + baseline tests if enabled
- Approval: LOW risk only (adaptive params: `confidence_floor, weight_multiplier, lookback, cooldown_seconds, volatility_multiplier, regime_affinity_multiplier`) auto approved
- Protected: `execution, risk, credential, organism, event_bus` tokens require human

### Upgrade Needed for Intelligent Profit

**A. Expand Allow-List for Real Data Fixes:**

Add to `ApprovalPolicy.allowed_imports`:
```
allowed_imports = (
  "numpy", "pandas", "ta", "scipy", "datetime", "math",
  "statistics", "typing", "dataclasses",
  "collections", "deque", "queue", # for ring buffer
  "json", "time",  # safe
  " PolygonAdapter stub", "alpaca_adapter", # internal adapters should be allowed
)
```
Still block `os, sys, subprocess, requests, ccxt` direct network calls in generated code - those belong in `core/real_data_connector.py` which is protected and human reviewed. Generated code can only tune params that *use* data connector output, not call network itself. This keeps safety but allows self-coding to fix logic.

**B. Per-Module Self-Coding Triggers:**

In `OrganismConfig`, already `self_code_each_cycle=True`. For each module:

```python
# Pseudo in organism.py self_improvement cycle
for module_name, module in self.modules.items():
  stats = learning_store.module_stats(module_name, window=100)
  mistakes = learning_store.mistakes(module_name, limit=10)
  if stats["win_rate"] < 0.50 or stats["mistake_rate"] > 0.3:
    context = {"stats": stats, "mistakes": mistakes, "regime": self._last_regime}
    diagnosis = module.diagnose(context)
    if diagnosis["issue"] != "none":
      coder.run_for_module(module, context, apply=auto_apply_low_risk)
```

**C. Learn From Past Mistakes - Automated Loop:**

`LearningStore.record_outcome` already creates mistake lesson if `pnl<0` or `reward<0.5`. Need auto-feed:

1. `execution/okx_engine.py` or `alpaca_executor` after fill closes position -> calls `learning_store.record_outcome(prediction_id, module_name, symbol, pnl, correct, reason)`
2. `Organism.generate_consensus_signal` records `prediction` with `prediction_id` -> ties signal to later outcome
3. Mistake event published: `bus.publish("MISTAKE_RECORDED", mistake_dict, priority=ADAPTIVE)`
4. `SelfCodingEngine` subscribes to `MISTAKE_RECORDED`, diagnoses, proposes fix

**Example Mistake Record:**

```json
{"type":"mistake","module_name":"ofi_detector","symbol":"BTCUSDT","regime":"crash","lesson":"OFI >0.6 gave BUY but price fell -2% due to funding crowded + exchange inflow 5k BTC. Reduce confidence when funding_z>2 and inflow_z>2","pnl":-0.015}
```

Self-coder reads this and generates proposal: increase confidence floor when `funding_z>2` -> change `apply_adaptive_parameters({"confidence_floor": 0.75})`.

**D. Market Improvement Self-Improvement:**

When regime switches from crash to trend (detected via `MarketRegimeDetector` vol ratio falling, VIX <20), organism should:

- Reset `weight_multiplier` for trend modules (OFI, CVD) from 0.2 back to 1.0
- Reduce `cooldown_seconds` for mean reversion modules to trade more often
- Publish `REGIME_IMPROVEMENT` event, modules increase size

Implementation: `Organism` already has `_last_regime`. Add:

```python
if previous_regime.label == "crash" and new_regime.label == "trend":
  for m in trend_modules: m.apply_adaptive_parameters({"weight_multiplier": 1.0})
  bus.publish("REGIME_IMPROVEMENT", {"from":"crash","to":"trend"}, priority=ADAPTIVE)
```

**E. Auto-Approve Safely:**

- Keep `protected_path_tokens` - generated code can NEVER mutate `execution/*, risk/*, organism, event_bus, credential`. It can only mutate `core/*_indicator.py` adaptive params and `advanced_modules/*` non-execution logic.
- Add `max_auto_changes_per_cycle=3` already prevents mutation storm (metabolic guard)
- Add cooldown 180s prevents same module mutating wildly
- `GoldSetStressTester` must pass before promotion: new candidate must not degrade crash detection >70%

---

## 4. CORE DATA SYSTEM BEFORE BROKER - PRE-BROKER READING

User says: "capabilities of reading data as soon as it's in the core data system, before it get to the broker"

Current path is slow:
```
Binance WS -> WebSocketStreams.Queue -> callback -> EventBus -> Consensus -> AlpacaAdapter REST -> Broker
Latency: 80-150ms (WS) + 50ms (consensus) + 200ms (Alpaca REST) = 330ms. Market maker moved already.
```

**Target path for instant move:**

### A. Zero-Copy In-Memory Bus

- Replace `Queue()` with `ring buffer` (Python `collections.deque` maxlen or better `disruptor` pattern using `numpy` circular buffer in shared memory)
- `live_data/websocket_streams.py` should write directly to `core/data_ring.py` shared memory array: `timestamp, bid, ask, bid_size, ask_size, last_price, last_qty`
- No JSON decoding per trade in hot path: use `orjson` or `ujson`, pre-allocate dict

```python
# core/data_ring.py
class DataRing:
  def __init__(self, size=100000):
    self.buffer = np.zeros(size, dtype=[('ts','f8'),('bid','f4'),('ask','f4'),('b_size','f4'),('a_size','f4'),('price','f4'),('qty','f4')])
    self.head = 0
  def push_tick(self, tick):
    self.buffer[self.head] = tick
    self.head = (self.head+1) % len(self.buffer)
  def latest(self, n=100):
    # zero copy view
    return self.buffer[max(0,self.head-n):self.head]
```

- `institutional_order_flow` reads `latest(100)` directly from ring, no bus hop. Publish only final signal to bus, not every tick.

### B. Direct Exchange Feed Before Broker (Don't wait for broker)

For BTCUSD ETHUSD:
- Use Binance Futures WS `wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@trade/btcusdt@markPrice` - this is exchange direct, not broker aggregated.
- For SPY / SPX500: 
  - Best free low latency: Polygon.io WS `wss://socket.polygon.io/stocks` -> trades + quotes + FMV. Latency 15-30ms. Better than Alpaca REST.
  - Better paid: Databento or Rithmic for ES futures depth.
  - Alpaca Adapter should only be for execution, NOT for data. Data must come from Polygon, execution via Alpaca/IBKR. Separate adapters.

```python
# core/real_data_connector.py
class RealDataConnector:
  def __init__(self):
    self.polygon_ws = PolygonWebSocket(api_key, subscribe=["T.SPY","Q.SPY","FMV.SPY"]) # Trades, Quotes, Fair Market Value
    self.binance_ws = BinanceDepthWS(symbols=["BTCUSDT","ETHUSDT"])
    self.data_ring = DataRing()
  
  def on_binance_depth(self, msg):
    # msg arrives 100ms
    self.data_ring.push_tick(...)
    # Trigger OFI instantly, before any broker sees it
    self.ofi_detector.process_tick(msg) # 2ms
    if ofi_signal confidence >0.8:
      # Publish immediately to bus with CRITICAL lane? No, OPERATIONAL but high priority
      get_event_bus().publish("ALPHA_SIGNAL", ..., priority=EventPriority.OPERATIONAL)
```

- Broker adapter `AlpacaAdapter` or `IBExecutor` subscribes to final consensus signal `FINAL_SIGNAL`, not raw ticks. So data is read in core before broker.

### C. Pre-Broker Data System Diagram

```
[Binance Exchange Direct WS 100ms] -> [DataRing in-memory] -> [OFI/CVD 5ms] -> [EventBus OPERATIONAL] -> [Consensus 5ms] -> [Execution Adapter Alpaca/IBKR] -> Broker
                                      \-> [VolumeProfile 10ms] ----^
[Polygon WS Trades/Quotes 20ms] ------> [SPY OFI] --------------------------------^
[Deribit Options WS] -----------------> [GEX] ------------------------------------^
[CryptoQuant WS Netflow] -------------> [Whale Flow] -----------------------------^
[Binance Funding REST 30s cached] ----> [FundingZ] -------------------------------^

All modules read from DataRing BEFORE broker.
Broker only sees final BUY/SELL, not raw tick stream.
```

---

## 5. MOVE WITH MARKET MAKER / WHALES, NOT RETAIL

Retail is late because they see candle close. Whales/market makers move first because they see:

- **Order book imbalance at L2 before price moves**
- **Options flow sweeps (huge call buys) before spot pumps**
- **Stablecoin deposits to exchange before buying**
- **Liquidation map clusters before sweep**
- **Funding extremes before squeeze**

### How to Move WITH Them:

**A. Whale Flow Detector (Must Build)**

Data sources:
- **CryptoQuant API**: `exchange_inflow = sum(btc moving into exchange wallets)`. Endpoint `/v2/coin/btc/exchange_flows`. If inflow_zscore 1h >2.5 => whale deposit.
- **Arkham/Nansen**: Whale wallet tagged (Binance cold, etc.) moving >1000 BTC after dormancy.
- **Mempool**: `mempool.space/api/mempool` pending tx count, fee spikes.
- **Deribit Block Trades**: `get_last_trades_by_currency?currency=BTC&kind=option&count=1000` filter `iv > 80% and premium > $500k`

Signals:
```
IF exchange_inflow_large (>5k BTC in 1h) AND funding positive (>0.05%)
  => Distribution imminent, even if price pumping. Move WITH whale: SELL or reduce long, not BUY like retail.
  Confidence 0.85

IF stablecoin inflow large (USDT mcap to exchanges up 200M in 1h) AND CVD flat
  => Buying power arriving, accumulation. BUY before retail sees candle.
```

Publish event `WHALE_FLOW` type `accumulation`, `distribution`.

**B. Market Maker Intent Detector**

Market makers:
- Place large bid/ask walls, then pull when approached (spoof).
- Absorb: CVD divergence shows they absorb retail market orders without price moving much.

Detection:
- **Absorption Cluster**: price range 0.1% with volume 3x average but price stays flat. See `AbsorptionCluster` in `institutional_order_flow.py` already good skeleton. Must wire.
- **Spoof Detector**: order at level 3-5, size >5x avg, lifetime <200ms, cancel rate >90%. When spoof on ask pulled => market maker wants price up (fake supply). Move WITH: BUY.

Code:
```python
def detect_mm_intent(l2_snapshots):
  # l2_snapshots last 100ms x 20 levels
  for level in l2[2:5]:
    if level.size > 5*avg_size and level.lifetime_ms < 200 and level.cancelled:
      return "spoof_ask_pulled" -> bullish intent
```

**C. Move Instantly - Latency Budget**

Target: <50ms from exchange tick to final signal. Current 330ms too slow.

How:
- No Python GIL heavy loops in hot path: use `numpy` vectorized, `numba` for OFI calc.
- OFI/CVD detectors as `numba.jit` compiled functions
- Consensus weighted vote as simple dot product, not dict loop
- Execution: use `ccxt.pro` WS for limit order placement? Or better FIX API for SPY (Alpaca doesn't have FIX, IBKR has). For crypto, placing limit via Binance REST `POST /fapi/v1/order` ~90ms. Can't beat 10ms HFT but can beat retail 500ms by using pre-placed limit inside spread and cancel fast.

**Tactics for Instant:**
1. **Pre-position**: Place limit order at 0.5 spread inside BEFORE signal, with size small. When OFI triggers, amend size via `replace` (faster than new order). Some exchanges support `amend`.
2. **No market orders for entry**: Market order slippage 5-10bp. Use limit at mid+0.1 spread, wait 200ms, if not filled and signal still valid, market in.
3. **TWAP bypass for whale signals**: If `WHALE_FLOW` distribution + `OFI>0.7`, confidence >0.85, skip TWAP, execute immediate market with 1% size, rest TWAP.

**D. BTCUSD / ETHUSD / SP500 Specific Hooks**

- **BTCUSD & ETHUSD**: Same exchange (Binance), same patterns. Share `funding`, `cvd`, `ofi` modules. ETH often lags BTC 200-400ms. Use BTC OFI as leader for ETH: if BTC OFI bullish + ETH OFI neutral but ETH funding not crowded, long ETH for mean reversion catch-up.
- **SP500**: SPY via Polygon. But better use ES futures `ES` via IBKR for true market maker depth (CME). SPY lags ES 5-10ms. Read ES order book from `ib_insync` market depth, then trade SPY. That's moving with futures market makers before SPY retail sees move.
- **Cross-asset**: DXY inverse correlation. If DXY spikes +0.3% in 5min + BTC OFI neutral -> BTC likely down (dollar up = risk off). Use DXY feed from `fred` or `OANDA` FX WS for early warning.

---

## 6. SELF-CODING MODULE TEMPLATE FOR BUILDER AI

Each new institutional module MUST inherit this:

```python
# core/modules_template.py
from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from autonomy.learning import LearningStore
import numpy as np

@register_module
class InstitutionalModuleBlueprint(BaseTradingModule):
  module_name = "my_ofi_detector"
  category = "microstructure"
  version = "1.0.0"
  dependencies = ["volume_profile", "regime_detector"]

  def __init__(self, config=None, event_bus=None):
    super().__init__(config, event_bus)
    self.config.setdefault("adaptive", {
      "confidence_floor": 0.62,
      "weight_multiplier": 1.0,
      "lookback": 100,
      "cooldown_seconds": 1.0
    })
    self.last_signal_ts = 0

  def initialize(self):
    bus = get_event_bus()
    # ONE ORGANISM - listen to siblings
    bus.subscribe("REGIME_CHANGE", self.on_event)
    bus.subscribe("WHALE_FLOW", self.on_event)
    bus.subscribe("ALPHA_SIGNAL", self.on_event) # cross-pollinate
    return True

  def analyze(self, market_data):
    # Read pre-broker data ring directly for speed
    # market_data contains ring ref: market_data["data_ring"].latest(100)
    start = time.time()
    ring = market_data.get("data_ring").latest(self.config["adaptive"]["lookback"])
    ofi_score = self.compute_ofi(ring) # numba compiled

    signal = "NEUTRAL"
    conf = 0.0
    if ofi_score > 0.6:
      signal = "BUY"
      conf = min(1.0, ofi_score)
    
    result = ModuleResult(
      module_name=self.module_name,
      signal=signal,
      confidence=conf,
      features={"ofi_score": ofi_score, "regime": market_data.get("regime")},
      latency_ms=(time.time()-start)*1000
    )
    # Publish to organism
    get_event_bus().publish("ALPHA_SIGNAL", result.to_dict(), source=self.module_name, priority=EventPriority.OPERATIONAL)
    return result

  def on_event(self, event):
    # Interconnect: adjust self when sibling says something
    if event.event_type == "WHALE_FLOW" and event.payload.get("type") == "distribution":
      # Whale dumping, reduce BUY confidence
      self.config["adaptive"]["confidence_floor"] = min(0.85, self.config["adaptive"]["confidence_floor"]+0.05)

  def learn_from_outcome(self, outcome):
    # Auto-learn from past mistakes
    if outcome["pnl"] < 0 and outcome["module_name"] == self.module_name:
      # Mistake
      self.config["adaptive"]["confidence_floor"] += 0.01
      return {"learned": True, "lesson": f"Raised floor to {self.config['adaptive']['confidence_floor']} after loss {outcome['pnl']} in {outcome['regime']}"}
    if outcome["pnl"] > 0:
      # Reinforce: slightly lower floor to capture more
      self.config["adaptive"]["confidence_floor"] = max(0.55, self.config["adaptive"]["confidence_floor"]-0.005)
    return {"learned": False}

  def diagnose(self, context):
    stats = context.get("stats", {})
    if stats.get("win_rate", 0.5) < 0.48:
      return {"issue": "low_win_rate", "suggestion": "tighten OFI threshold", "params": {"confidence_floor": 0.70}}
    if stats.get("mistake_rate", 0) > 0.3:
      return {"issue": "high_mistake_rate", "suggestion": "increase cooldown"}
    return {"issue": "none"}

  def apply_adaptive_parameters(self, parameters):
    # Only allow-listed params can be auto-tuned by self-coder
    return super().apply_adaptive_parameters(parameters)

  @staticmethod
  def compute_ofi(ring):
    # Numba for speed
    ...
```

**Self-coding flow:**
1. `diagnose` suggests param change
2. `SelfCodingEngine` generates `CodeProposal` with `parameters = {"confidence_floor":0.70}`
3. `SafeCodeValidator` checks AST, complexity <10, allowed import
4. `ShadowManager` deploys shadow clone with new param, observes 100 ticks real live data
5. If shadow Sharpe +5% vs live and DD not worse, `Organism._promote_shadow` swaps weight
6. `LearningStore` records promotion, future mistakes use new threshold

---

## 7. ROADMAP FOR BUILDER AI - TO IMPLEMENT ORGANISM

**Priority 0 (Make organism real):**
- Audit `autonomy/organism.py` -> enable `self_code_each_cycle=True`, reduce mutation cooldown 180s to 60s for testing
- In `core/base_module.py`, ensure `register_module` decorator used for ALL new institutional modules
- Create `core/data_ring.py` zero-copy ring buffer
- Refactor `live_data/websocket_streams.py` to push to ring, not Queue, use `orjson`
- Separate data vs execution adapters: `PolygonAdapter` for SPY data, `AlpacaAdapter` only execution; `BinanceWSAdapter` for BTC data, `BinanceExecutor` for execution

**Priority 1 (Whale & MM):**
- Build `core/whale_flow_detector.py` using CryptoQuant API + Deribit block trades
- Build `core/mm_intent_detector.py` spoof + absorption from `institutional_order_flow.py`
- Build `core/volume_profile.py` + `core/cvd_indicator.py` + `core/funding_indicator.py` + `core/gex_indicator.py`
- Wire all to publish `WHALE_FLOW`, `MM_INTENT`, `ALPHA_SIGNAL` events

**Priority 2 (Self-coding):**
- Expand `ApprovalPolicy.allowed_imports` slightly to include internal adapters
- Ensure `learning_store.record_outcome` auto-called from `execution/event_driven_executor.py`
- Test self-coding cycle: force a module to have low win_rate via fake outcomes, see if it proposes higher confidence_floor and promotes via shadow

**Priority 3 (Pre-broker <50ms):**
- Benchmark latency: instrument `time.time()` from WS message recv to final signal
- Replace Python loops with `numba`, use `numpy` dot for consensus
- Pre-place limit orders inside spread, amend on signal (instant)

---

## 8. CONCLUSION

One organism = one bus + 4 lanes + every module publishes/consumes.

Self-coding = diagnose from LearningStore mistakes + generate low-risk adaptive param proposal + validate AST + shadow test 100 ticks + auto-promote if +5% Sharpe.

Pre-broker core data = DataRing zero-copy from exchange direct WS, not broker REST. Read before broker.

Move with whales = detect whale deposits/options sweeps/stables inflow BEFORE candle close, publish WHALE_FLOW event, modules adjust confidence instantly, bypass TWAP for high confidence.

Move with market makers = OFI/CVD/absorption/spoof detection at L2 100ms, not 1m candle. That's how market makers move; retail sees candle 60s later.

Implement blueprint, then you move WITH flow, not AFTER.

