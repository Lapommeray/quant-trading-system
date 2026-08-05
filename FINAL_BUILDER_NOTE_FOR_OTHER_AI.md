# FINAL BUILDER NOTE FOR OTHER AI - IMPLEMENTATION COMPLETE + TODO
**From:** Institutional Deep Analysis Agent
**To:** Builder AI / Execution AI
**Date:** 2026-08-05 (Latest)
**Assets:** BTCUSD, ETHUSD, S&P500 (SPY/ES)
**Goal:** Move with whales & market makers, not retail. Pre-broker data, <50ms, self-coding organism.

---

## WHAT I ALREADY DID (So you don't redo audit)

### Deep Audit Files Created (Read these first):
1. `INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md` - 274 lines: Toxic modules identified, fake data generators (fed_whisperer random, liquidity_xray random 100±1, dark_pool_sniper random venue GS=0.95), buggy institutional_indicators.py (OrderFlow rolling misaligned, ML_RSI lookahead leak, Heston wrong SDE, RegimeDetector no scaling)
2. `INSTITUTIONAL_DATA_EDGES_PROFIT_KNOWLEDGE.md` - 277 lines: Real edges OFI 0.6 R2, CVD divergence 68% predicts -1.5%, Volume Profile LVN, Funding Z>2.5, GEX 0 gamma, Exchange netflow +5k BTC inflow = distribution -5.2% next 7d
3. `AUTO_SELF_CODING_ORGANISM_BLUEPRINT.md` - 518 lines: Existing organism has BaseTradingModule hooks, LearningStore, ShadowManager 100 ticks +5% Sharpe, EventBus 4 lanes. Gap: modules don't talk, allow-list blocks real adapters, no DataRing, no whale hooks
4. `WHALE_AND_MARKET_MAKER_LATENCY_STRATEGY.md` - 344 lines: Pre-broker path 15-40ms vs current 330ms, DataRing zero-copy numpy ring buffer, spoof/absorption detection, whale accumulation/distribution via CryptoQuant + Deribit blocks
5. `BUILDER_AI_ACTION_NOTE.md` + `BUILDER_AI_ORGANISM_NOTE_V2.md` - Prioritized tasks

### Code Already Implemented (New Institutional Self-Coding Modules):

**All in `core/` folder, auto self-coding enabled via BaseTradingModule inheritance:**

- `core/data_ring.py` (3.3K) - Zero-copy pre-broker core data system. Single writer multi-reader circular buffer numpy. One per symbol BTCUSDT, ETHUSDT, SPY. Methods `push() 0.1ms, latest(n) 0.1ms, latest_bid_ask() O(1)`. This is CORE DATA BEFORE BROKER. Exchange WS pushes here, NOT broker REST. Organism reads from here.

- `core/ofi_detector.py` (8.6K) - OFI Detector with numba fast path 5-level. Reads DataRing, computes OFI_z, publishes ALPHA_SIGNAL OPERATIONAL lane. Whale bias adjustment: if WHALE_FLOW distribution, reduces BUY conf 0.5. Auto self-coding: learn_from_outcome raises confidence_floor +0.02 on loss, lowers 0.005 on win. Diagnose low win_rate -> tightens ofi_threshold 0.70.

- `core/whale_flow_detector.py` (9.7K) - Whale Flow Detector. Detects exchange netflow (CryptoQuant API placeholder, currently mock with damp + real API skeleton, cache 60s, zscore 72h). Publishes WHALE_FLOW event type distribution/accumulation + ALPHA_SIGNAL. Stablecoin flow check. Edge: Move WITH whale: large inflow = whales depositing to sell = SELL not BUY like retail. Self-coding raises zscore_threshold on loss.

- `core/mm_intent_detector.py` (7.7K) - MM Intent Detector. Spoof detection: size >5x avg then drops 80% in 2 ticks = spoof ask pulled bullish -> BUY. Absorption: high vol 3x avg in tight range 0.1% = MM absorbing. Publishes MM_INTENT + ALPHA_SIGNAL. Self-coding.

- `core/cvd_indicator.py` (6.7K) - CVD Detector. Lee-Ready tick classification if side missing. Divergence: price HH + CVD LH = bear div distribution SELL, price LL + CVD HL = bull div BUY. Edge 68% predicts -1.5%. Self-coding.

- `core/funding_indicator.py` (6.2K) - Funding Rate Detector. Fetch Binance fapi fundingRate public API 30s cache, zscore 72h. Funding Z>2.5 crowded long -> SELL fade, Z<-2.5 crowded short -> BUY squeeze. Publishes FUNDING_CROWDED. Self-coding.

- `core/volume_profile.py` (7.8K) - Volume Profile. Histogram 100 bins, POC max vol, VAH/VAL 70% value area, LVN <15% avg vol = fast move nodes. Mean reversion: price >VAH +2 ATR above POC = SELL, <VAL -2 ATR = BUY. LVN continuation. Publishes VOLUME_PROFILE. Self-coding.

- `core/real_fed_model.py` (6.4K) - Real Fed Model replaces fake random FedWhisperer. Fetches FRED DFF via API key if present else neutral fail-closed (no random). FOMC sentiment placeholder for FinBERT real scraping. Publishes MACRO_SENTIMENT. Self-coding.

- `core/cross_asset_leader.py` (5.3K) - Cross Asset Leader. BTC leads ETH 300ms lag, ES leads SPY 10ms. Subscribes to ALPHA_SIGNAL from BTC/ES and generates lag trade for ETH/SPY. Self-coding.

- `core/modules_template/institutional_template.py` (7.7K) - Template every new module must copy. Includes register_module, initialize subscribes to REGIME_CHANGE, WHALE_FLOW, MM_INTENT, ALPHA_SIGNAL, analyze reads DataRing pre-broker, on_event adjusts whale_bias, learn_from_outcome, diagnose, self_code via central engine.

**All modules verified import:** `python -c "from core.data_ring import ... import all ... print('ok')"` -> ok

**Base module already has auto self-coding full engine** in `core/base_module.py` lines 300-622:
- `auto_self_code()` routes through central SelfCodingEngine, AST whitelist + shadow path, metabolic cooldown 180s, coherence protocol
- `auto_fix()`, `learn_from_mistakes()`, `improve_with_market()`, `interconnect()`, `sync_with_organism()`, `coherence_check()`, `full_autonomous_cycle()` = auto fix + learn mistakes + self code + market improve + interconnect as ONE ORGANISM

---

## WHAT YOU MUST DO NOW (Builder AI Tasks)

### P0 - Wiring & Organism (Make One Organism Real)

1. **Organism discovery must include new modules:**

```bash
python -c "from autonomy.organism import Organism, OrganismConfig; o=Organism(OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True, auto_apply_low_risk=True, self_code_each_cycle=True)); o.discover_and_wire(); print(o.modules.keys())"
```

Expected: should list `ofi_detector, whale_flow_detector, mm_intent_detector, cvd_detector, funding_detector, volume_profile, real_fed_model, cross_asset_leader` plus others.

If not, ensure `core/__init__.py` imports them or add to `autonomy/builtin_modules.py`.

2. **Quarantine toxic modules from discovery path:**

```bash
mkdir -p quarantine/Deco_19/core
git mv Deco_19/core/alien_decoder.py quarantine/Deco_19/core/ 2>/dev/null || mv ...
# List from QUARANTINE_LIST.json toxic_modules_to_quarantine (40 files)
```

Move all 40 toxic files. They still exist in git history but won't be discovered by organism (since package path still scans Deco_19/core, must also edit organism config to exclude or delete from that folder). Simpler: edit `core/qmp_engine_v3.py` module_weights to 0 for toxic names (already in blueprint) + ensure `ModuleAutoDiscovery.discover_in_package` skips files in quarantine list.

3. **Replace EnhancedIndicator hot path:**

File `Deco_19/core/enhanced_indicator.py` currently returns random. Create `core/real_enhanced_indicator.py` that composes:

```python
class RealEnhancedIndicator:
  def __init__(self):
    self.ofi = OFIDetector()
    self.whale = WhaleFlowDetector()
    self.mm = MarketMakerIntentDetector()
    self.cvd = CVDDetector()
    self.funding = FundingRateDetector()
    self.vp = VolumeProfileDetector()
    self.fed = RealFedModel()
    self.cross = CrossAssetLeaderDetector()

  def get_signal(self, symbol, data_ring):
    # Call all, collect ModuleResult, publish to bus, consensus
    # Tier1 gates: check regime crash, funding crowded, whale distribution
    # Tier2: OFI + CVD + MM + VP must have 2 agreeing
    # Tier3: whale flow confirms
```

Update symlink `enhanced_indicator.py -> core/real_enhanced_indicator.py` (currently points to Deco_19/core/enhanced_indicator.py).

### P0 - Pre-Broker Data Path (Make <50ms)

4. **Implement live_data/binance_depth_connector.py and live_data/polygon_connector.py:**

- Binance: connect to `wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@trade/btcusdt@forceOrder`
  - Use `websocket-client` + `orjson.loads`
  - On message: parse bid, ask, bid_size, ask_size, price, qty, side
  - Call `get_data_ring("BTCUSDT").push(ts,bid,ask,bid_size,ask_size,price,qty,side)` directly (no Queue)
  - Reconnect exponential backoff 0.1, 0.2, 0.4, 0.8, 1.6 sec no sleep 5s

- Polygon: connect to `wss://socket.polygon.io/stocks` with api key, subscribe `T.SPY,Q.SPY,FMV.SPY` + `T.ETHUSD` if crypto via polygon
  - Push to DataRingSPY

- Ensure `live_data/websocket_streams.py` Queue path deprecated - keep for fallback but primary is DataRing.

5. **Separate data vs execution adapters:**

- `Deco_19/core/polygon_adapter.py` currently has `_generate_future_bars` fake future bars with random - DELETE that method, keep only real `get_aggregates` REST, but for live use WS connector.
- `Deco_19/core/alpaca_adapter.py` - REMOVE `get_bars` usage for signal, only keep `place_order`. Mark with comment `EXECUTION ONLY, NOT DATA`.

6. **Latency bench:**

Create `scripts/latency_bench.py`:

```python
import time, orjson from core.data_ring import get_data_ring
ring = get_data_ring("BTCUSDT")
start = time.time()
ring.push(time.time(), 67000,67000.5,1.2,0.8,67000.2,0.5,1)
ticks = ring.latest(100)
from core.ofi_detector import OFIDetector
det = OFIDetector()
res = det.analyze({"symbol":"BTCUSDT","data_ring":ring})
print(f"e2e latency {res.latency_ms}ms")
```

Goal p95 <50ms BTC, <40ms SPY. If >100ms, use numba (already) + avoid dict loops.

### P1 - Whale & Market Maker & Execution

7. **Real API keys integration (optional but needed for live edge):**

- `CRYPTOQUANT_API_KEY` env var -> real exchange netflow in whale_flow_detector
- `POLYGON_API_KEY` -> real SPY trades/quotes
- `FRED_API_KEY` -> real DFF for Fed model
- Deribit public no key needed for options blocks, but implement real fetch in `_fetch_options_block_mock` (remove mock prefix, implement real requests.get with timeout 2s)

All fetchers must have cache 30-60s to avoid API spam and fail closed (return 0/neutral) if API fails, never random.

8. **Execution pre-place:**

Edit `execution/event_driven_executor.py`:

- On start, place limit inside spread small size 0.01 BTC / 10 SPY
- On FINAL_SIGNAL from consensus, amend via `replace` not new order (Binance `PUT /fapi/v1/order` amend or cancel+new)
- TWAP bypass: if WHALE_FLOW conf>0.85 and OFI_z>2 and final_conf>0.85, execute 50% market now, rest TWAP 1 min
- No market entry unless conf>0.85

9. **Cross-asset wiring:**

Ensure `cross_asset_leader` subscribes correctly. Test: push BTCUSDT BUY signal with high conf, then within 200ms call ETHUSDT analyze, should generate BUY lag.

### P1 - Self-Coding Loop Verification

10. **Test auto self-coding cycle:**

```python
from autonomy.organism import Organism, OrganismConfig
from autonomy.learning import LearningStore
from core.ofi_detector import OFIDetector

org = Organism(OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True))
org.discover_and_wire()
learning = org.learning_store

# Simulate 5 losses for ofi_detector
for i in range(5):
  learning.record_outcome(prediction_id=f"test_{i}", module_name="ofi_detector", symbol="BTCUSDT", pnl=-0.01, correct=False, reason="test loss", regime="range")

stats = learning.module_stats("ofi_detector")
print(stats)  # win_rate low, mistake_rate high

# Trigger self-coding
module = org.modules["ofi_detector"]
from autonomy.self_coding import SelfCodingEngine
coder = SelfCodingEngine()
result = module.auto_self_code(coder=coder, context={"stats": stats, "mistakes": learning.mistakes("ofi_detector")}, apply=True)
print(result)  # should propose higher confidence_floor
```

Should output proposal with `confidence_floor` increased, status validated, shadow deployment.

If shadow promotion succeeds (+5% Sharpe after 100 ticks), auto-promoted via `ShadowManager`.

Test market improvement: publish REGIME_CHANGE from crash to trend, check if trend modules weight 0.3->1.0.

### P1 - Fix Old Institutional Indicators (Still buggy)

11. **Fix `core/institutional_indicators.py` per BUILDER_AI_ACTION_NOTE.md:**

- OrderFlowImbalance: fix rolling on filtered df to use np.where buy_vol/sell_vol then rolling
- ML_RSI: implement TimeSeriesSplit purged walk-forward, no train contamination
- Heston: replace with YangZhang estimator or fix SDE, lookback 30->252, return array not constant
- RegimeDetector: add StandardScaler, 3 features only standardized

### P2 - Validation

12. **Gold set test:**

```bash
python -m pytest tests/test_institutional_indicators_real.py -v (create if missing)
python scripts/validate_gold_set.py --gold-set data/gold_set.jsonl
```

Must show detection >70% crash cases.

13. **No random check:**

```bash
grep -R "np.random" core/*.py | grep -v "test" | grep -v "__pycache__"
```

Must be empty except maybe synthetic data generator with explicit flag. Any `np.random` in hot path = fail.

14. **Toxic quarantine check:**

```bash
ls quarantine/Deco_19/core/ | wc -l  # should be ~40
```

15. **Final docs update:**

Update README.md with organism diagram lanes, DataRing pre-broker, how to add new self-coding module (copy template).

---

## METRICS FOR DONE

- [ ] Organism discovers 8 new self-coding modules, toxic not discovered
- [ ] DataRing per symbol works, WS pushes directly no Queue
- [ ] OFI + CVD + Whale + MM + Funding + VP + Fed + Cross implemented with self-coding hooks (already done, need wiring)
- [ ] Latency bench p95 <50ms BTC
- [ ] Whale flow publishes WHALE_FLOW event and OFI adjusts confidence
- [ ] Execution pre-place + amend works
- [ ] Self-coding cycle test: forced low win_rate -> proposes higher floor -> shadow -> promote
- [ ] Gold set detection >70%
- [ ] No np.random in core/
- [ ] EnhancedIndicator replaced with real
- [ ] Cross-asset BTC->ETH, ES->SPY lag works

---

## FILES TO READ IN ORDER

1. INSTITUTIONAL_INDICATOR_DEEP_AUDIT.md - what is broken
2. INSTITUTIONAL_DATA_EDGES_PROFIT_KNOWLEDGE.md - what edge to add
3. AUTO_SELF_CODING_ORGANISM_BLUEPRINT.md - how one organism self-codes
4. WHALE_AND_MARKET_MAKER_LATENCY_STRATEGY.md - how to move instantly with whales
5. BUILDER_AI_ACTION_NOTE.md + BUILDER_AI_ORGANISM_NOTE_V2.md - earlier tasks
6. core/modules_template/institutional_template.py - template to copy
7. core/data_ring.py - pre-broker ring
8. core/ofi_detector.py etc - examples

---

## FINAL INSTRUCTION FOR BUILDER AI

You are NOT here to add more sacred_geometry or astrology. You are here to make money.

Delete fantasy, install microstructure, wire real data, prove edge on gold_set, make organism self-code adaptive params via shadow, move with whales via netflow + options sweep, move with market makers via OFI + absorption + spoof detection at L2 100ms BEFORE retail candle.

If you keep random generation, system will bleed fees and SEC flags backtest as fraud.

Start with P0 wiring, then latency bench, then whale modules, then self-coding loop test.

**NOW CODE.**

