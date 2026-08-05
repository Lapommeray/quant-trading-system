#!/usr/bin/env python3
"""
Organism Smoke Test - proves the whole trading organism boots and breathes
as ONE system, fully offline (no network, no broker, fail-closed).

Verifies in one pass:
  1. Organism discovery + wiring (event bus lanes, module graph)
  2. Pre-broker data path: OKX v5 frames -> DataRing -> event bus
     (recorded real-format frames, parsed without network)
  3. SPX (S&P 500) feed ingestion into the same DataRing/bus
  4. Institutional modules (OFI, CVD) consuming the ring and emitting signals
  5. Learning from mistakes: record outcomes -> module stats -> mistake lessons
  6. Auto self-coding: bounded SelfCodingEngine proposal + validation
  7. ExecutionPlanner: maker/taker/wait decisions (move with MM/whales)

Run:  python scripts/organism_smoke.py
Exit: 0 = organism healthy, 1 = any subsystem failed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    print("=" * 72)
    print("ORGANISM SMOKE TEST - one organism, offline, fail-closed")
    print("=" * 72)

    # ------------------------------------------------------------ 1. organism
    print("\n[1] Organism discovery + wiring")
    from autonomy.organism import Organism, OrganismConfig

    org = Organism(
        OrganismConfig(
            self_coding_enabled=True,
            auto_approve_low_risk=True,
            self_code_each_cycle=False,  # we trigger one cycle explicitly
            log_dir="data/organism_logs/smoke",
            module_packages=("autonomy", "core"),
        )
    )
    wiring = org.discover_and_wire()
    active = list(org.modules)
    check("modules discovered", len(active) >= 8, f"{len(active)} active: {sorted(active)}")
    check("wiring failures", not wiring.get("failed"), json.dumps(wiring.get("failed", {}))[:200])

    for name in ("ofi_detector", "cvd_detector", "whale_flow_detector", "funding_detector", "volume_profile"):
        check(f"core module wired: {name}", name in org.modules)

    bus = org.event_bus
    lane_stats = bus.get_lane_stats()
    check("event bus 4 lanes", set(lane_stats) >= {"critical", "operational", "adaptive", "evolutionary"},
          json.dumps(lane_stats))

    # -------------------------------------------------- 2. pre-broker OKX path
    print("\n[2] Pre-broker data path (recorded OKX v5 frames -> DataRing -> bus)")
    from core.data_ring import get_data_ring
    from okx_live.feed import OKXPreBrokerFeed, OKXFeedConfig

    ring_btc = get_data_ring("BTCUSDT")
    events: list[dict] = []
    # NOTE: capture key must not collide with payload keys (liquidation frames
    # carry their own "type" field), so we use "event_type" here.
    bus.subscribe("WHALE_FLOW", lambda e: events.append({"event_type": e.event_type, **e.payload}))
    bus.subscribe("LIQUIDATION_EVENT", lambda e: events.append({"event_type": e.event_type, **e.payload}))

    feed = OKXPreBrokerFeed(
        OKXFeedConfig(
            symbols=["BTC-USDT-SWAP"],
            channels=["trades", "books5", "liquidation-orders"],
            whale_usd_threshold=50_000.0,
        ),
        event_bus=bus,
        ring_factory=lambda sym: get_data_ring(sym),
    )

    # Real-format OKX public WS frames (snapshot of the live protocol).
    frames = [
        {"arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"}, "data": [{
            "asks": [["109234.1", "1.02"], ["109235.0", "2.5"]],
            "bids": [["109200.0", "4.1"], ["109199.0", "0.8"]], "ts": "1722840000000"}]},
        {"arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"}, "data": [
            {"instId": "BTC-USDT-SWAP", "tradeId": "1", "px": "109210.0", "sz": "0.5", "side": "buy", "ts": "1722840000100"},
            {"instId": "BTC-USDT-SWAP", "tradeId": "2", "px": "109212.0", "sz": "0.9", "side": "buy", "ts": "1722840000200"},
            {"instId": "BTC-USDT-SWAP", "tradeId": "3", "px": "109215.0", "sz": "2.0", "side": "buy", "ts": "1722840000300"}]},
        {"arg": {"channel": "liquidation-orders", "instId": "BTC-USDT-SWAP"}, "data": [
            {"instId": "BTC-USDT-SWAP", "side": "sell", "posSide": "long", "bkPx": "108900.0", "sz": "12.5", "type": "filled", "ts": "1722840000400"}]},
    ]
    for frame in frames:
        feed.handle_frame(frame)

    ring_snapshot = ring_btc.latest(5)
    check("OKX frames parsed", feed.feed_status()["stats"]["frames"] == 3, json.dumps(feed.feed_status()["stats"]))
    check("DataRing populated (BTCUSDT)", len(ring_snapshot) >= 5,
          f"bid={ring_snapshot[-1]['bid']} ask={ring_snapshot[-1]['ask']}")
    whale_events = [e for e in events if e["event_type"] == "WHALE_FLOW"]
    check("whale flow event (0.5 BTC = $54k)", len(whale_events) >= 1,
          json.dumps(whale_events[0]) if whale_events else "none")
    check("whale cluster detected", bool(whale_events) and any(e.get("cluster") for e in whale_events))
    check("liquidation event", any(e["event_type"] == "LIQUIDATION_EVENT" for e in events))

    # ------------------------------------------------------------ 3. SPX path
    print("\n[3] S&P 500 pre-broker feed (fake fetcher, no network)")
    from quant_trading_system.data_feeds.spx_feed import SPXLiveFeed

    spx_ring = get_data_ring("SPX")
    spx = SPXLiveFeed(
        event_bus=bus,
        ring_factory=lambda sym: get_data_ring(sym),
        fetch_func=lambda: {
            "ts": time.time(), "price": 5530.25, "open": 5520.0,
            "high": 5540.0, "low": 5515.0, "volume": 12_345.0,
        },
    )
    spx._ingest(spx._fetch_func())
    spx_status = spx.feed_status()
    check("SPX ring populated", spx_status["last_price"] == 5530.25 and len(spx_ring.latest(2)) >= 1,
          f"price={spx_status['last_price']}")

    # ------------------------------------------ 4. modules consume the ring
    print("\n[4] Institutional modules consuming pre-broker ring")
    ofi = org.modules["ofi_detector"]
    cvd = org.modules["cvd_detector"]
    res_ofi = ofi.analyze({"symbol": "BTCUSDT", "data_ring": ring_btc})
    res_cvd = cvd.analyze({"symbol": "BTCUSDT", "data_ring": ring_btc})
    check("OFI analyze ran", res_ofi is not None, f"signal={res_ofi.signal} conf={res_ofi.confidence:.2f}")
    check("CVD analyze ran", res_cvd is not None, f"signal={res_cvd.signal} conf={res_cvd.confidence:.2f}")

    # --------------------------------------------- 5. learning from mistakes
    print("\n[5] Learning from past mistakes")
    learning = org.learning_store
    for i in range(6):
        learning.record_outcome(
            prediction_id=f"smoke_{i}", module_name="ofi_detector", symbol="BTCUSDT",
            pnl=-0.012 if i % 2 == 0 else 0.004, correct=(i % 2 == 1),
            reason="loss" if i % 2 == 0 else "win", regime="range",
        )
    stats = learning.module_stats("ofi_detector")
    mistakes = learning.mistakes("ofi_detector")
    check("outcomes recorded", stats.get("outcomes", stats.get("total_outcomes", 0)) >= 6,
          json.dumps(stats)[:160])
    check("mistake lessons exist", len(mistakes) >= 1, f"{len(mistakes)} lessons")

    # --------------------------------------------------- 6. auto self-coding
    print("\n[6] Auto self-coding (bounded, validated, governed)")
    from autonomy.self_coding import SelfCodingEngine

    coder = SelfCodingEngine()
    ctx = {"stats": stats, "mistakes": mistakes, "reason": "smoke_loss_streak"}
    result = ofi.auto_self_code(coder=coder, context=ctx, apply=True)
    check("self-code proposal produced", result.get("status") in ("validated", "approved", "applied", "pending_approval")
          or result.get("proposal_status") in ("VALIDATED", "APPROVED", "APPLIED", "PENDING_APPROVAL"),
          json.dumps({k: v for k, v in result.items() if k != "code"})[:240])
    check("self-code validated", result.get("validated", result.get("validation", {}).get("passed", False)) in (True, "passed", "validated"))

    # ------------------------------------------------- 7. execution planning
    print("\n[7] ExecutionPlanner - move with MM/whales, not retail")
    from core.execution_planner import ExecutionPlanner, ExecutionPlannerConfig

    planner = ExecutionPlanner(ExecutionPlannerConfig())
    base_market = {
        "symbol": "BTCUSDT", "bid": 109200.0, "ask": 109234.1,
        "bid_size": 4.1, "ask_size": 1.02, "feed_fresh": True,
    }

    p_retail = planner.plan("BTCUSDT", "buy", 0.60, {**base_market, "ofi_z": 0.3, "whale_conf": 0.1})
    check("weak signal aborts (no retail chasing)", p_retail.routing.value == "abort", p_retail.reason)

    p_flow = planner.plan("BTCUSDT", "buy", 0.72, {**base_market, "ofi_z": 0.9, "whale_conf": 0.3})
    check("flow OK -> maker inside spread", p_flow.routing.value == "maker" and p_flow.limit_price < 109234.1,
          f"limit={p_flow.limit_price} reason={p_flow.reason}")

    p_whale = planner.plan("BTCUSDT", "buy", 0.90, {**base_market, "ofi_z": 2.4, "whale_conf": 0.92, "cvd_align": True})
    check("whale stack -> immediate taker", p_whale.routing.value == "taker", p_whale.reason)

    p_recheck = planner.finalize_after_recheck(p_flow, {**base_market, "ofi_z": 1.8, "whale_conf": 0.6, "urgency": 0.7}, 210)
    check("recheck escalates with flow", p_recheck.routing.value == "taker", p_recheck.reason)

    p_dead = planner.finalize_after_recheck(p_flow, {**base_market, "ofi_z": -0.2, "whale_conf": 0.0, "urgency": 0.1}, 260)
    check("dead flow after timeout aborts", p_dead.routing.value == "abort", p_dead.reason)

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"ORGANISM SMOKE: {passed}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED:")
        for name, detail in failed:
            print(f"  - {name}: {detail}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
