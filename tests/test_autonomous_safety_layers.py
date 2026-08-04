"""Tests for the systemic containment layers."""

from autonomy import AutonomousGuardrails, EventPriority, EventBus
from autonomy.gold_set import GoldSet
from autonomy.sentinel import MultiTimeframeSentinel


def test_guardrails_only_allow_tighter_limits():
    guard = AutonomousGuardrails()
    assert guard.limits.max_position_pct == 0.02
    assert guard.limits.max_leverage == 1.0
    allowed, reason = guard.validate_trade({"position_size_pct": 0.03, "leverage": 1.0})
    assert not allowed
    assert "Position" in reason


def test_sentinel_enters_and_exits_survival_mode():
    sentinel = MultiTimeframeSentinel()
    shock = [100.0] * 21 + [110.0]
    decision = sentinel.evaluate({"1m": shock, "15m": shock, "1h": shock})
    assert decision.survival_mode is True
    stable = [100.0] * 22
    for _ in range(3):
        decision = sentinel.evaluate({"1m": stable, "15m": stable, "1h": stable})
    assert decision.survival_mode is False


def test_sentinel_failure_fails_safe():
    sentinel = MultiTimeframeSentinel()
    sentinel._measure_spike = lambda frame: (_ for _ in ()).throw(RuntimeError("dead"))
    decision = sentinel.evaluate({"1m": [100.0] * 30})
    assert decision.survival_mode is True
    assert decision.heartbeat_ok is False


def test_priority_bus_dispatches_async_lane():
    bus = EventBus()
    received = []
    bus.subscribe(
        "SELF_IMPROVEMENT",
        lambda event: received.append(event.event_type),
        asynchronous=True,
    )
    event = bus.publish_async("SELF_IMPROVEMENT", {"ok": True})
    assert event.priority == int(EventPriority.EVOLUTIONARY)
    assert bus.drain(2.0)
    assert received == ["SELF_IMPROVEMENT"]
    bus.stop()


def test_gold_set_has_crash_and_volatility_cases():
    summary = GoldSet.load().summary()
    assert summary["cases"] >= 20
    assert summary["crashes"] >= 10
    assert summary["volatile"] >= 10
