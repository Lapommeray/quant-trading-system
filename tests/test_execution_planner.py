"""Tests for the maker-first ExecutionPlanner (move with MM/whales)."""

import pytest

from core.execution_planner import ExecutionPlanner, ExecutionPlannerConfig, Routing


def _market(**overrides):
    base = {
        "symbol": "BTCUSDT",
        "bid": 109200.0,
        "ask": 109234.1,
        "bid_size": 4.1,
        "ask_size": 1.02,
        "feed_fresh": True,
        "ofi_z": 0.0,
        "whale_conf": 0.0,
        "cvd_align": True,
        "urgency": 0.0,
    }
    base.update(overrides)
    return base


def test_abort_when_confidence_below_floor():
    p = ExecutionPlanner().plan("BTCUSDT", "buy", 0.40, _market())
    assert p.routing == Routing.ABORT
    assert p.reason == "confidence_below_floor"


def test_abort_on_stale_feed():
    p = ExecutionPlanner().plan("BTCUSDT", "buy", 0.90, _market(feed_fresh=False))
    assert p.routing == Routing.ABORT
    assert p.reason == "stale_feed"


def test_abort_on_wide_spread():
    p = ExecutionPlanner().plan(
        "BTCUSDT", "buy", 0.90, _market(bid=109000.0, ask=110000.0)
    )
    assert p.routing == Routing.ABORT
    assert p.reason.startswith("spread_too_wide")


def test_abort_on_missing_quote():
    p = ExecutionPlanner().plan("BTCUSDT", "buy", 0.90, _market(bid=0.0, ask=0.0))
    assert p.routing == Routing.ABORT


def test_whale_stack_escalates_to_taker():
    p = ExecutionPlanner().plan(
        "BTCUSDT",
        "buy",
        0.90,
        _market(ofi_z=2.4, whale_conf=0.92, cvd_align=True),
    )
    assert p.routing == Routing.TAKER
    assert p.urgency == 1.0
    assert p.reason == "whale_stack_escalation"


def test_whale_stack_requires_cvd_alignment():
    p = ExecutionPlanner().plan(
        "BTCUSDT",
        "buy",
        0.90,
        _market(ofi_z=2.4, whale_conf=0.92, cvd_align=False),
    )
    assert p.routing != Routing.TAKER  # flow disagrees -> never take


def test_maker_limit_inside_spread_with_queue():
    p = ExecutionPlanner().plan(
        "BTCUSDT",
        "buy",
        0.72,
        _market(ofi_z=0.9, whale_conf=0.3, bid_size=8.0, ask_size=1.0),
    )
    assert p.routing == Routing.MAKER
    assert p.limit_price is not None
    assert p.limit_price < 109234.1  # inside the spread
    assert p.limit_price > 109200.0
    assert p.recheck_ms == 200.0


def test_no_queue_share_waits_instead_of_chasing():
    p = ExecutionPlanner().plan(
        "BTCUSDT",
        "buy",
        0.72,
        _market(ofi_z=0.2, whale_conf=0.0, bid_size=0.2, ask_size=10.0),
    )
    assert p.routing == Routing.WAIT
    assert p.reason == "flow_not_confirming"


def test_recheck_escalates_when_flow_confirms():
    planner = ExecutionPlanner()
    p = planner.plan(
        "BTCUSDT",
        "buy",
        0.72,
        _market(ofi_z=0.9, whale_conf=0.3, bid_size=8.0, ask_size=1.0),
    )
    assert p.routing == Routing.MAKER

    p2 = planner.finalize_after_recheck(
        p,
        _market(ofi_z=1.8, whale_conf=0.6, urgency=0.7),
        elapsed_ms=210,
    )
    assert p2.routing == Routing.TAKER
    assert p2.reason == "recheck_escalation"


def test_recheck_holds_maker_while_flow_alive():
    planner = ExecutionPlanner()
    p = planner.plan(
        "BTCUSDT",
        "buy",
        0.72,
        _market(ofi_z=0.9, whale_conf=0.3, bid_size=8.0, ask_size=1.0),
    )
    p2 = planner.finalize_after_recheck(
        p, _market(ofi_z=0.8, whale_conf=0.2, urgency=0.3), elapsed_ms=100
    )
    assert p2.routing == Routing.MAKER
    assert p2.reason == "maker_hold_after_recheck"


def test_recheck_aborts_when_flow_dies_past_timeout():
    planner = ExecutionPlanner(ExecutionPlannerConfig(max_wait_ms=250.0))
    p = planner.plan(
        "BTCUSDT",
        "buy",
        0.72,
        _market(ofi_z=0.9, whale_conf=0.3, bid_size=8.0, ask_size=1.0),
    )
    p2 = planner.finalize_after_recheck(
        p,
        _market(ofi_z=-0.2, whale_conf=0.0, urgency=0.1),
        elapsed_ms=260,
    )
    assert p2.routing == Routing.ABORT
    assert p2.reason == "flow_died_timeout"


def test_sell_side_mirrors_buy_logic():
    p = ExecutionPlanner().plan(
        "ETHUSDT",
        "sell",
        0.90,
        _market(ofi_z=2.5, whale_conf=0.9, cvd_align=True),
    )
    assert p.routing == Routing.TAKER
