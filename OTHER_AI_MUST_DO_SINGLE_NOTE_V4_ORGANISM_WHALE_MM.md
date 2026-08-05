# OTHER AI MUST DO - SINGLE CANONICAL NOTE V4 - ONE ORGANISM, SELF-CODING, WHALE + MM, PRE-BROKER, BTC ETH SP500

**From:** Institutional Deep Analysis Agent (analysis-only, not builder)
**To:** Builder AI / Execution AI / Any AI that touches code next
**Date:** 2026-08-05
**Urgency:** P0 - This is the ONLY note you need to read. Ignore older BUILDER_AI_* notes (they are superseded but kept for history).
**Security:** OKX keys exposed in earlier prompt `a11bcf6c-6bab-...` MUST be rotated NOW. See SECURITY_RED_ALERT_OKX_KEYS.md. Never commit keys. `.env` gitignored.

---

## MISSION - ONE ORGANISM THAT MOVES WITH WHALES & MARKET MAKERS, NOT AFTER RETAIL

- **Assets:** BTCUSD (BTCUSDT swap), ETHUSD (ETHUSDT swap), S&P500 (SPY via Polygon + ES futures via IBKR/databento)
- **Latency target:** Data WS -> DataRing push 0.1ms -> OFI 3ms -> CVD 5ms -> Consensus 2ms -> Pre-placed limit amend 10ms = 20-40ms e2e <50ms p95
- **Edge source:** Order Flow Imbalance (OFI) + CVD divergence + MM spoof/absorption + Whale exchange netflow + Stablecoin flow + Liquidation maps + Options block sweeps + Funding crowded + Volume Profile + Cross asset leader BTC->ETH 300ms ES->SPY 10ms
- **Auto self-coding:** Each module auto fixes, creates code artifact, approves low-risk adaptive params ONLY via central SelfCodingEngine, learns from past mistakes, improves when market regime improves itself, interconnects with every other module as ONE ORGANISM via EventBus 4 lanes
- **Pre-broker:** Read data as soon as it lands in core DataRing (numpy circular buffer) BEFORE it gets to broker REST. Broker never reads ring. Only alpha modules. Broker only gets FINAL_SIGNAL BUY/SELL.
- **Fail-closed:** No random np.random in hot path, no synthetic fallback, no sim if live requested and creds missing.

---

## WHAT ALREADY EXISTS (DON'T REDO, JUST WIRE)

### Core Data System BEFORE Broker (Institutional-Grade, Keep)

- `core/data_ring.py` 87 lines: `DataRing(symbol,size=200k)` dtype ts f8 bid f4 ask f4 bid_size f4 ask_size f4 price f4 qty f4 side i1, `buf` numpy zeros, `head` atomic, `push(ts,bid,ask,bid_size,ask_size,price,qty,side)` 0.1ms, `latest(n)` zero-copy view contig, wrap copy rare, `latest_bid_ask()` O1, `latency_ms()`. Global registry `get_data_ring(symbol)`.
- `okx_live/feed.py` 464 lines BEST FILE: OKX public WS books5,trades,tickers,open-interest,funding-rate,liquidation-orders NO KEY NEEDED, `handle_frame(frame)` pure parser offline-testable, pushes to DataRing, whale detection notional >50k usd threshold, cluster >=3 whales in 5s window counts as whale_cluster, publishes WHALE_FLOW OPERATIONAL, ORDER_BOOK_UPDATE throttled 0.25s, OI_UPDATE, FUNDING_UPDATE, LIQUIDATION_EVENT, `is_stale()` >10s, `check_stale_and_alert()` publishes RISK_ALERT.
- `core/base_module.py` 622 lines ALREADY HAS full autonomous engine per module:
  - `auto_self_code(coder,context,apply,auto_approve)` routes through CENTRAL SelfCodingEngine AST whitelist + shadow path, metabolic cooldown 180s, coherence protocol
  - `auto_fix(coder,reason,context)`
  - `learn_from_mistakes(mistake_data)` stores last 20 lessons, adaptive tuning weight_multiplier 0.92 overfit or 1.05 under miss
  - `improve_with_market(regime_data,market_context)` bullish weight 1.08 bear 0.85, triggers self-code if market improving conf>0.65, publishes MODULE_MARKET_IMPROVED
  - `interconnect(target_modules,message)` publishes MODULE_INTERCONNECT + targeted MODULE_{TARGET}_MESSAGE, stores _interconnections
  - `sync_with_organism(organism_state)` applies global weights, triggers self_code, coherence_check on mutation_event
  - `coherence_check(mutation_event)` risk modules tighten when peer aggressive weight>1.05 => weight 0.95 conf floor 0.70+0.05
  - `full_autonomous_cycle(coder,organism_context)` master auto_fix+learn+self_code+market_improve+interconnect = living cell
  - Allow-listed adaptive params ONLY: confidence_floor, weight_multiplier, lookback, cooldown_seconds, volatility_multiplier, regime_affinity_multiplier. Never order size leverage credentials.

### Real Self-Coding Modules Already Implemented (8 modules, all in `core/` inherit BaseTradingModule)

1. `core/ofi_detector.py` 8.6K OFI numba fast 5-level proxy best level, QI zscore 100, whale_bias adjustment, funding filter should add, learn floor +0.02 loss -0.005 win, diagnose low win_rate tighten 0.70
2. `core/whale_flow_detector.py` 9.7K exchange netflow mock + real API skeleton CryptoQuant 60s cache zscore 72h threshold 2.5 raises 3.5 on loss, stablecoin flow $200M threshold, publishes WHALE_FLOW distribution/accumulation, fail-closed 0 not random
3. `core/mm_intent_detector.py` 7.7K spoof detection size>5x avg drops 80% in 2 ticks 200ms lifetime => fake wall pulled bullish BUY, absorption high vol 3x tight range 0.1% = MM absorbing, publishes MM_INTENT
4. `core/cvd_indicator.py` 6.7K CVD Lee-Ready tick classification if side missing, divergence price HH CVD LH bear dist SELL 68% -1.5% next 15m, price LL CVD HL bull acc BUY
5. `core/funding_indicator.py` 6.2K Binance fapi fundingRate 30s cache zscore 72h Z>2.5 SELL fade crowded long Z<-2.5 BUY squeeze, publish FUNDING_CROWDED
6. `core/volume_profile.py` 7.8K histogram 100 bins POC max vol VAH/VAL 70% value area expanding LVN <15% avg fast move, mean reversion >VAH+2ATR SELL <VAL-2ATR BUY LVN continuation
7. `core/real_fed_model.py` 6.4K replaces fake random fed_whisperer, FRED DFF via API key if present else neutral fail-closed no random, FOMC sentiment FinBERT placeholder, publish MACRO_SENTIMENT
8. `core/cross_asset_leader.py` 5.3K BTC leads ETH 300ms ES leads SPY 10ms, subscribes ALPHA_SIGNAL caches last BTC/ES signal ts, lag window 300ms/100ms, DXY inverse TODO
9. `core/modules_template/institutional_template.py` template every new module must copy

All verified import: `python -c "from core.data_ring import get_data_ring; from core.ofi_detector import OFIDetector; ... print('ok')"`

### Organism Already Has One Brain

- `autonomy/organism.py` 1311 lines Organism class:
  - ModuleAutoDiscovery discovers decorated modules in packages autonomy,core,advanced_modules
  - OrganismConfig self_coding_enabled=True auto_approve_low_risk=True auto_apply_low_risk=True self_code_each_cycle=True max_auto_changes_per_cycle=3 module_mutation_cooldown_sec=180 enable_coherence_protocol=True
  - LearningStore records prediction->outcome->mistake lesson
  - ShadowManager deploy candidate shadow compare 100 ticks min_outperformance 5% max_drawdown_delta 1% promote only if passes
  - ConsensusEngine weighted dot
  - EventBus 4 lanes CRITICAL 0 sync no async, OPERATIONAL 1 sync, ADAPTIVE 2 async, EVOLUTIONARY 3 async, priority queue ensures lane0 before lane3, worker thread daemon
  - Coherence Protocol MUTATION_EVENT listener _on_mutation_event broadcasts to all modules coherence_check() risk auto-realign
  - Metabolic guard 180s prevents mutation wars resource exhaustion
  - ONE_ORGANISM_HEARTBEAT published each cycle with all active modules list
- `autonomy/self_coding.py` 1302 lines SelfCodingEngine bounded:
  - CodeProposal risk LOW/MEDIUM/HIGH/CRITICAL status GENERATED->VALIDATED->APPROVED->APPLIED
  - SafeCodeValidator AST allow-list max_bytes 64k max_lines 200 complexity 10 forbidden imports os,sys,subprocess,socket,requests,ccxt,multiprocessing,threading,pathlib,importlib,ctypes,pickle, forbid eval exec compile __import__ open etc forbidden names __builtins__ environ, credential marker detection, obfuscation detection `chr(` >5 or base64 80
  - ApprovalPolicy auto_approve only LOW adaptive params protected tokens okx_live,execution,risk,safety,credential,secret,organism,event_bus,main.py
  - GeneratedTestRunner unittest companion non-skippable, RegressionTestRunner baseline pytest
  - PenaltyBox 1h after policy violation
  - Methods generate_proposal,validate_proposal,approve_proposal,apply_proposal,run_for_module,autonomous_self_code_cycle,auto_fix,attempt_auto_repair,create safe module policy artifact never overwrites live source writes to `strategies/evolved/module_name/proposal_{id}.py` inert
- `autonomy/shadow.py`, `gold_set.py`, `sentinel.py`, `learning.py`, `market.py`, `monitor.py` already.
- `core/event_bus.py` canonical singleton `get_event_bus()` shared with okx_live.

### Toxic Modules List (40 files, must quarantine)

See `INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md` Section Category B + `QUARANTINE_LIST.json` toxic_modules_to_quarantine:
fed_whisperer, liquidity_xray, candlestick_dna_sequencer, dark_pool_sniper, stop_hunter, order_flow_hunter, market_maker_slayer, future_shadow, future_zone_sensory, ghost_candle, timeline_selector, multiverse_sync, zero_point, alien_decoder, angel_decoder, cosmic_channeler, divine_sync, astro_geo_sync (all oversoul), emotion_dna, fractal_resonance, quantum_tremor, sacred_event, energy_filter, etc.

### Docs Already Created

- `INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md` 275 lines
- `INSTITUTIONAL_DATA_EDGES_PROFIT_KNOWLEDGE.md` 277 lines
- `AUTO_SELF_CODING_ORGANISM_BLUEPRINT.md` 518 lines
- `WHALE_AND_MARKET_MAKER_LATENCY_STRATEGY.md` 344 lines
- `BUILDER_AI_ACTION_NOTE.md`, `BUILDER_AI_ORGANISM_NOTE_V2.md`
- `FINAL_BUILDER_NOTE_FOR_OTHER_AI.md` 345 lines you are reading V4 supersedes it
- This V4 deep audit `INSTITUTIONAL_INDICATOR_DEEP_AUDIT_V4.md` you are reading now companion

---

## WHAT YOU MUST DO NOW - P0 P1 P2

### P0 - WIRING & ORGANISM (Make One Organism Real, Remove Theater)

#### P0.1 Verify Organism Discovers 8 New Modules

```bash
python -c "
from autonomy.organism import Organism, OrganismConfig
o=Organism(OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True, auto_apply_low_risk=True, self_code_each_cycle=True))
wired=o.discover_and_wire()
print('active', wired['active'])
print('modules keys', list(o.modules.keys()))
assert 'ofi_detector' in o.modules, 'ofi missing'
assert 'whale_flow_detector' in o.modules
assert 'mm_intent_detector' in o.modules
assert 'cvd_detector' in o.modules
assert 'funding_detector' in o.modules
assert 'volume_profile' in o.modules
assert 'cross_asset_leader' in o.modules
print('OK organism wired')
"
```

Expected: lists above plus momentum_alpha, regime_risk_gate builtin, maybe more but toxic not listed.

If not, ensure `core/__init__.py` or `autonomy/builtin_modules.py` imports them or add explicit imports in `core/__init__.py`? Simpler: `autonomy.organism.ModuleAutoDiscovery.discover_in_package("core")` uses pkgutil.iter_modules scanning `core/` folder for classes subclassing BaseTradingModule - since our 8 files do `@register_module` they are discovered via decorated registry `get_registered_modules()`, not package scan. Ensure each file imported once: add `import core.ofi_detector, core.whale_flow_detector, ...` to `core/__init__.py` or in organism bootstrap script.

File to edit: `core/__init__.py` currently lazy __getattr__ only institutional_indicators. Expand to import all self-coding modules so registry populated on import.

**Code to add in `core/__init__.py` or new `core/register_all.py`:**

```python
import importlib
for mod in ["ofi_detector","whale_flow_detector","mm_intent_detector","cvd_indicator","funding_indicator","volume_profile","cross_asset_leader","real_fed_model"]:
    try:
        importlib.import_module(f"core.{mod}")
    except Exception as e:
        print(f"failed {mod} {e}")
```

#### P0.2 Quarantine Toxic Modules

Create `quarantine/` dir, move toxic files, but keep git history (git mv). However organism config package path still scans Deco_11/core/oversoul etc. Need to edit OrganismConfig.module_packages to exclude Deco_* or edit ModuleAutoDiscovery to skip files listed in QUARANTINE_LIST.json + any file containing `def.*pass` stub.

Simpler immediate: edit `core/qmp_engine_v3.py` module_weights to 0 for toxic names (already blueprint says). But best: add `quarantine/` and move.

```bash
mkdir -p quarantine/Deco_19/core quarantine/Deco_11/core/oversoul
# read QUARANTINE_LIST.json
cat QUARANTINE_LIST.json
# loop mv
```

Ensure final `ls quarantine/` has ~40 files.

After move, re-run organism discover, toxic should not appear.

+ Add check in `autonomy/organism.py` ModuleAutoDiscovery.discover_in_package: if module name in quarantine list skip.

#### P0.3 Replace EnhancedIndicator Hot Path

File `Deco_19/core/enhanced_indicator.py` currently random. Create new `core/real_enhanced_indicator.py` compositor:

```python
from core.ofi_detector import OFIDetector
from core.whale_flow_detector import WhaleFlowDetector
...
class RealEnhancedIndicator:
  def __init__(self, event_bus=None):
    self.ofi=OFIDetector(event_bus=event_bus)
    self.whale=WhaleFlowDetector(event_bus=event_bus)
    self.mm=MarketMakerIntentDetector(event_bus=event_bus)
    self.cvd=CVDDetector(event_bus=event_bus)
    self.funding=FundingRateDetector(event_bus=event_bus)
    self.vp=VolumeProfileDetector(event_bus=event_bus)
    self.cross=CrossAssetLeaderDetector(event_bus=event_bus)
    self.fed=RealFedModel(event_bus=event_bus)
  def get_signal(self,symbol,data_ring):
    # Tier1 gates: regime crash, funding crowded, whale distribution
    # Tier2: OFI+CVD+MM+VP must agree at least 2
    # Tier3: whale confirms
    # Return TradeDecision
```

Update symlink repo root `enhanced_indicator.py` currently points to Deco_19/core. Change to `core/real_enhanced_indicator.py`.

Also edit `core/qmp_engine_v3.py` _stub wrapper to return neutral for toxic modules.

### P0 - Pre-Broker Data Path <50ms

#### P0.4 Implement live_data/binance_depth_connector.py and polygon_connector.py

**binance_depth_connector.py** spec:

```python
import websocket, orjson, time
from core.data_ring import get_data_ring

class BinanceDepthConnector:
  def __init__(self,symbols=["BTCUSDT","ETHUSDT"]):
    self.url="wss://fstream.binance.com/stream?streams=" + "/".join([f"{s.lower()}@depth20@100ms/{s.lower()}@trade/{s.lower()}@forceOrder" for s in symbols])
    self.rings={s:get_data_ring(s) for s in symbols}
  def on_message(self,ws,msg):
    data=orjson.loads(msg)
    # parse depth: data['data'] contains bids asks
    # for trade: price qty side
    # push to ring: ring.push(ts,bid,ask,bid_size,ask_size,price,qty,side)
  def start(self):
    ws=websocket.WebSocketApp(self.url,on_message=self.on_message,...)
    ws.run_forever(ping_interval=20)
```

Requirements: reconnect exponential backoff 0.1,0.2,0.4,0.8,1.6 sec not sleep 5s, orjson zero-copy parse 1ms, no Queue(), direct push to DataRing.

**polygon_connector.py:** similar for SPY: `wss://socket.polygon.io/stocks` with apikey, subscribe `T.SPY,Q.SPY,FMV.SPY`, push to DataRingSPY. For ES: use IBKR `ib_insync` reqMktDepth or Databento.

Ensure `live_data/websocket_streams.py` Queue path deprecated keep fallback but primary DataRing.

#### P0.5 Separate Data vs Execution Adapters

- `Deco_19/core/polygon_adapter.py` has `_generate_future_bars` fake future bars random DELETE method keep only real `get_aggregates` REST but for live use WS connector.
- `Deco_19/core/alpaca_adapter.py` REMOVE `get_bars` usage for signal only keep `place_order` comment `# EXECUTION ONLY, NOT DATA`
- `okx_live/` already separated: feed.py public WS NO KEY, trader.py needs key only if LIVE_TRADING true.

#### P0.6 Latency Bench

Create `scripts/latency_bench.py`:

```python
import time
from core.data_ring import get_data_ring
ring=get_data_ring("BTCUSDT")
start=time.time()
ring.push(time.time(),67000,67000.5,1.2,0.8,67000.2,0.5,1)
ticks=ring.latest(100)
from core.ofi_detector import OFIDetector
det=OFIDetector()
res=det.analyze({"symbol":"BTCUSDT","data_ring":ring})
print(f"e2e latency {res.latency_ms}ms")
# goal p95 <50ms
```

Run, ensure p95 <50ms BTC. If >100ms use numba already + avoid dict loops.

### P1 - Whale & MM & Execution & Real APIs

#### P1.1 Real API Keys Integration

Env vars:

- `CRYPTOQUANT_API_KEY` → real exchange netflow in whale_flow_detector. Implement real fetch in `_fetch_exchange_netflow_mock` rename to `_fetch_exchange_netflow` with requests.get timeout 2s cache 60s fail-closed 0.
- `POLYGON_API_KEY` → real SPY trades quotes.
- `FRED_API_KEY` → real DFF Fed model already.
- Deribit public no key needed for options blocks implement real `_fetch_options_block_mock` rename to real requests.get `https://www.deribit.com/api/v2/public/get_last_trades_by_currency?currency=BTC&kind=option&count=100` filter usd>500k.

All fetchers must cache 30-60s avoid spam and fail-closed return 0/neutral never random.

#### P1.2 Execution Pre-Place & Amend

Edit `execution/event_driven_executor.py` or `okx_live/trader.py`:

- On start place limit inside spread small size 0.01 BTC / 10 SPY shares bid=mid-0.2*spread ask=mid+0.2*spread
- On FINAL_SIGNAL consensus amend via replace not new order Binance PUT /fapi/v1/order amend or cancel+new. Replace 30% faster.
- TWAP bypass: if WHALE_FLOW conf>0.85 and OFI_z>2 and final_conf>0.85 execute 50% now market rest TWAP 1min
- No market entry unless conf>0.85
- Kill switch cooldown if 2 losses row with whale flow increase cooldown 2s avoid chasing
- Log p50 p95 p99 latency per module

#### P1.3 Cross-Asset Wiring Test

Push BTCUSDT BUY signal high conf, then within 200ms call ETHUSDT analyze should generate BUY lag.

```python
from core.cross_asset_leader import CrossAssetLeaderDetector
from core.data_ring import get_data_ring
det=CrossAssetLeaderDetector()
det.on_event(type('Event',(),{'event_type':'ALPHA_SIGNAL','payload':{'module_name':'ofi_detector','symbol':'BTCUSDT','signal':'BUY','confidence':0.85,'features':{'symbol':'BTCUSDT'}}}))
res=det.analyze({"symbol":"ETHUSDT","data_ring":get_data_ring("ETHUSDT")})
print(res.signal) # should BUY
```

### P1.4 Self-Coding Loop Verification

Test auto self-coding cycle:

```python
from autonomy.organism import Organism, OrganismConfig
from autonomy.learning import LearningStore

org=Organism(OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True, auto_apply_low_risk=True))
org.discover_and_wire()

# Simulate 5 losses for ofi_detector
for i in range(5):
  org.learning_store.record_outcome(prediction_id=f"test_{i}", module_name="ofi_detector", symbol="BTCUSDT", pnl=-0.01, correct=False, reason="test loss", regime="range")

stats=org.learning_store.module_stats("ofi_detector")
print(stats) # win_rate low

module=org.modules["ofi_detector"]
from autonomy.self_coding import SelfCodingEngine
coder=SelfCodingEngine(project_root=".", event_bus=org.event_bus)
result=module.auto_self_code(coder=coder, context={"stats":stats,"mistakes":org.learning_store.mistakes("ofi_detector")}, apply=True)
print(result) # should propose higher confidence_floor validated shadow deployment
```

Should output proposal with confidence_floor increased, status validated, shadow deployment.

Test market improvement: publish REGIME_CHANGE crash→trend, check weight 0.3→1.0 for trend modules via improve_with_market.

### P1.5 Fix Old Institutional Indicators Still Buggy Per V4 Audit

- `core/institutional_indicators.py` OrderFlowImbalance fixed rolling filtered df to np.where buy_vol sell_vol then rolling - already fixed in current version good.
- ML_RSI TimeSeriesSplit purged walk-forward no train contamination - already fixed good but needs more features.
- Heston replaced with YangZhang estimator lookback 30→252 return array not constant - already fixed good.
- RegimeDetector add StandardScaler 3 features only standardized - already fixed good.

Verify via tests.

### P2 - Validation & No Random

#### P2.1 Gold Set Test

```bash
python -m pytest tests/test_institutional_indicators_real.py -v # create if missing
python scripts/validate_gold_set.py --gold-set data/gold_set.jsonl
```

Must show detection >70% crash cases before -4% move.

#### P2.2 No Random Check

```bash
grep -R "np.random" core/*.py | grep -v "__pycache__" | grep -v test
# must be empty except maybe synthetic data generator with explicit flag
```

Any np.random in hot path = fail.

#### P2.3 Toxic Quarantine Check

```bash
ls quarantine/Deco_19/core/ | wc -l # should ~40
```

#### P2.4 Final Docs Update

Update README.md with organism diagram lanes DataRing pre-broker how to add new self-coding module copy template.

---

## METRICS FOR DONE - ACCEPTANCE CRITERIA

- [ ] SECURITY: Exposed OKX keys rotated, no keys in repo grep empty, `.env` gitignored chmod 600
- [ ] Organism discovers 8 new self-coding modules, toxic not discovered (list active modules = 10-12 including builtin)
- [ ] DataRing per symbol works BTCUSDT ETHUSDT SPY ES, WS pushes directly no Queue
- [ ] OFI + CVD + Whale + MM + Funding + VP + Fed + Cross implemented with self-coding hooks (already done, need wiring verified)
- [ ] Latency bench p95 <50ms BTC <40ms SPY e2e
- [ ] Whale flow publishes WHALE_FLOW event and OFI adjusts confidence via whale_bias
- [ ] Execution pre-place + amend works, no market unless conf>0.85
- [ ] Self-coding cycle test forced low win_rate → proposes higher floor → shadow → promote logic works
- [ ] Gold set detection >70%
- [ ] No np.random in core/ hot path
- [ ] EnhancedIndicator replaced with real RealEnhancedIndicator tiered gates not average
- [ ] Cross-asset BTC→ETH, ES→SPY lag works within 300ms/100ms
- [ ] One Organism heartbeat every cycle, interconnect publishes MODULE_INTERCONNECT, all modules list active modules
- [ ] Auto self-coding injection verified all modules have auto_self_code, auto_fix, learn_from_mistakes, improve_with_market, interconnect, full_autonomous_cycle, coherence_check methods (inherited from BaseTradingModule)

---

## CORE TEMPLATE - COPY FOR EVERY NEW MODULE

File `core/modules_template/institutional_template.py` contains full template. Key points:

- `@register_module` decorator mandatory
- `module_name` unique string
- `initialize()` subscribes to REGIME_CHANGE, WHALE_FLOW, MM_INTENT, ALPHA_SIGNAL
- `analyze(market_data)` reads `market_data.get("data_ring") or get_data_ring(symbol)` pre-broker, latency <10ms, returns ModuleResult signal/confidence/features/latency_ms, publishes ALPHA_SIGNAL OPERATIONAL lane
- `on_event(event)` handles Event object or (event_type,payload) tuple, adjusts whale_bias regime weight
- `learn_from_outcome(outcome)` raises confidence_floor +0.02 on loss, lowers 0.005 on win
- `diagnose(context)` returns issue low_win_rate + suggestion params
- All self-coding methods inherited from BaseTradingModule, no need to re-implement unless custom, but can override for custom tuning

**Central self-coding engine is ONLY allowed to mutate adaptive params allow-list, never execution.**

---

## INTERCONNECT PATTERN - ONE ORGANISM

Every module is neuron:

```python
# In analyze after computing signal
get_event_bus().publish("ALPHA_SIGNAL", result.to_dict(), source=self.module_name, priority=EventPriority.OPERATIONAL)

# In initialize subscribe siblings
bus.subscribe("WHALE_FLOW", self.on_event)
bus.subscribe("FUNDING_CROWDED", self.on_event)
bus.subscribe("MM_INTENT", self.on_event)
bus.subscribe("REGIME_CHANGE", self.on_event)
bus.subscribe("ALPHA_SIGNAL", self.on_event) # cross-pollination

# In on_event adjust
if event.payload["type"]=="distribution" and signal=="BUY": conf*=0.5

# In organism self_improvement cycle
for name,module in modules:
  module.interconnect(target_modules=list(modules.keys()), message={"organism":"unified","cycle":...})
# publishes MODULE_INTERCONNECT + MODULE_{TARGET}_MESSAGE
# stores _interconnections set
```

Organism publishes ONE_ORGANISM_HEARTBEAT each cycle with modules list.

Coherence protocol: MUTATION_EVENT broadcast to all modules coherence_check() risk modules tighten when peer aggressive.

Metabolic rate guard _MUTATION_COOLDOWN_SEC 180s prevents mutation wars.

---

## PRE-BROKER CORE DATA SYSTEM - WHY IT MAKES YOU MOVE WITH WHALE/MM

DataRing is single source of truth IN MEMORY before broker sees anything. Zero-copy pre-broker.

- OKX public WS → DataRing push 0.1ms
- Alpha modules read latest(n) 0.1ms
- No JSON dict per tick in hot path, numpy struct
- Single writer WS thread multiple readers lock-free head atomic reading older index safe
- Broker never reads ring, only alpha, broker only gets final signal

Separate Data vs Execution adapters:

| Asset | Data Source Pre-Broker | Execution Broker | Latency |
| BTCUSD | Binance fstream depth20@100ms + trade + markPrice + OKX books5 | Binance fapi REST or WS order | Data 100ms Exec 90ms |
| ETHUSD | Same ETHUSDT + BTC leader | Same | Same + BTC lag |
| SPY | Polygon.io WS T.SPY Q.SPY FMV 20ms | Alpaca or IBKR | Data 20ms Exec 150ms |
| ES | CME via IBKR reqMktDepth / Databento | IBKR | Data 30ms Exec 50ms FIX |

Why separate: Alpaca bars 1Min delayed 15s at times. Polygon real-time. Don't use Alpaca for data ever.

**Your edge is time + flow, not pattern.** Retail trades candle close, MM trades L2 imbalance 400ms before, whale trades exchange flow 10min before. Reading WS before broker aggregation you move WITH maker not AFTER.

---

## OKX INTEGRATION SECURE + FAST

- `okx_live/feed.py` public NO KEY handles 6 channels pushes DataRing publishes WHALE_FLOW when single trade notional >50k usd, cluster >=3 in 5s, OI FUNDING LIQ events.
- `okx_live/config.py` frozen dataclass from_env fail-closed validates ccxt present keys length >=10 if LIVE_TRADING true else paper.
- `okx_live/trader.py` `engine.py` `runner.py` `safety.py` only need keys when LIVE_TRADING true.
- Never commit keys, `.env` gitignored.
- For BTCUSD ETHUSD use OKX feed + Binance depth for cross-validation.

**Setup:**

```bash
cp .env.example .env
chmod 600 .env
# edit with rotated keys
OKX_API_KEY=...
OKX_API_SECRET=...
OKX_PASSPHRASE=...
OKX_LIVE_TRADING=false
```

Test feed without keys:

```python
from okx_live.feed import OKXPreBrokerFeed
feed=OKXPreBrokerFeed()
feed.handle_frame({"arg":{"channel":"trades","instId":"BTC-USDT-SWAP"},"data":[{"px":"67000","sz":"0.5","side":"buy","ts":"1710000000000"}]})
print(feed.feed_status()) # frames=1 trades=1 whales maybe
```

---

## FINAL INSTRUCTION - MAKE MONEY NOT ASTROLOGY

You are NOT here to add more sacred_geometry or astrology. You are here to make money.

Delete fantasy, install microstructure, wire real data, prove edge on gold_set, make organism self-code adaptive params via shadow, move with whales via netflow + options sweep, move with market makers via OFI + absorption + spoof detection at L2 100ms BEFORE retail candle.

If you keep random generation, system will bleed fees and SEC flags backtest as fraud.

Start P0 wiring, then latency bench, then whale modules, then self-coding loop test.

**Folders to read in order:**

1. SECURITY_RED_ALERT_OKX_KEYS.md
2. INSTITUTIONAL_INDICATOR_DEEP_AUDIT_V4.md (this V4 companion) + INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md V1
3. INSTITUTIONAL_DATA_EDGES_PROFIT_KNOWLEDGE.md + WHALE_AND_MARKET_MAKER_LATENCY_STRATEGY.md
4. AUTO_SELF_CODING_ORGANISM_BLUEPRINT.md + core/base_module.py lines 300-622 full autonomous cycle
5. core/modules_template/institutional_template.py template
6. core/data_ring.py + okx_live/feed.py pre-broker
7. core/ofi_detector.py etc 8 modules examples
8. autonomy/organism.py + self_coding.py bounded engine

**NOW CODE - WIRED AS ONE BRAIN, SELF-FIXING, WHALE-ALIGNED, PRE-BROKER.**

