from datetime import datetime, timedelta, timezone

from core.module_mesh_orchestrator import (
    ModuleMeshConfig,
    ModuleMeshOrchestrator,
    UnifiedAction,
)


class TrendModule:
    def analyze(self, context):
        assert "price" in context
        return {"action": "BUY", "confidence": 0.8, "reason": "trend_up"}


class LiquidityModule:
    def process(self, context):
        trend_output = context["module_outputs"]["trend"]
        assert trend_output["reason"] == "trend_up"
        context["liquidity_saw_trend"] = True
        return {"score": 0.7, "confidence": 0.7, "reason": "liquidity_supports"}


class RiskModule:
    def run(self, context):
        assert context["liquidity_saw_trend"] is True
        return {"action": "BUY", "confidence": 0.75, "reason": "risk_clear"}


def test_modules_communicate_and_return_one_unified_result():
    mesh = ModuleMeshOrchestrator(
        {"trend": TrendModule(), "liquidity": LiquidityModule(), "risk": RiskModule()},
        config=ModuleMeshConfig(
            confidence_threshold=0.6, disagreement_hold_threshold=0.0
        ),
    )

    result = mesh.run({"symbol": "SPY", "price": 500.0})

    assert result.action is UnifiedAction.BUY
    assert result.tradable is True
    assert result.reason == "module_consensus"
    assert list(result.module_results) == ["trend", "liquidity", "risk"]
    assert result.context["module_outputs"]["trend"]["reason"] == "trend_up"
    assert len(result.context["event_log"]) == 3
    as_dict = result.to_dict()
    assert as_dict["action"] == "BUY"
    assert set(as_dict["module_results"]) == {"trend", "liquidity", "risk"}


def test_module_failure_fails_closed_to_hold():
    def broken(context):
        raise RuntimeError("boom")

    mesh = ModuleMeshOrchestrator(
        {"trend": TrendModule(), "broken": broken},
        config=ModuleMeshConfig(confidence_threshold=0.1),
    )

    result = mesh.run({"symbol": "SPY", "price": 500.0})

    assert result.action is UnifiedAction.HOLD
    assert result.tradable is False
    assert "module_failure_fail_closed" in result.reason
    assert result.module_results["broken"].ok is False
    assert "boom" in result.module_results["broken"].error


def test_module_veto_fails_closed_to_hold():
    mesh = ModuleMeshOrchestrator(
        {
            "trend": TrendModule(),
            "risk": lambda context: {"veto": True, "reason": "drawdown_limit"},
        },
        config=ModuleMeshConfig(confidence_threshold=0.1),
    )

    result = mesh.run({"symbol": "SPY", "price": 500.0})

    assert result.action is UnifiedAction.HOLD
    assert result.tradable is False
    assert result.reason == "module_veto: risk"


def test_disagreement_returns_one_hold_result():
    mesh = ModuleMeshOrchestrator(
        {
            "trend": lambda context: {"action": "BUY", "confidence": 0.9},
            "flow": lambda context: {"action": "SELL", "confidence": 0.8},
        },
        config=ModuleMeshConfig(
            confidence_threshold=0.1, disagreement_hold_threshold=0.1
        ),
    )

    result = mesh.run({"symbol": "SPY", "price": 500.0})

    assert result.action is UnifiedAction.HOLD
    assert result.tradable is False
    assert result.reason.startswith("module_disagreement")


def test_stale_context_never_runs_modules_and_holds():
    stale = datetime.now(timezone.utc) - timedelta(hours=1)
    called = {"value": False}

    def module(context):
        called["value"] = True
        return {"action": "BUY", "confidence": 1.0}

    mesh = ModuleMeshOrchestrator(
        {"module": module},
        config=ModuleMeshConfig(max_context_age_seconds=10),
    )

    result = mesh.run({"timestamp": stale, "symbol": "SPY", "price": 500.0})

    assert result.action is UnifiedAction.HOLD
    assert result.reason == "stale_or_future_context"
    assert called["value"] is False
