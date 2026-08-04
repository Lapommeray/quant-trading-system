"""Standard-library tests for the canonical bounded autonomy loop."""

from pathlib import Path

from autonomy import (
    ApprovalPolicy,
    Organism,
    OrganismConfig,
    ProposalStatus,
    SafeCodeValidator,
    SelfCodingEngine,
)
from autonomy.learning import LearningStore
from autonomy.market import MarketRegimeDetector
from core.event_bus import EventBus


def test_learning_store_persists_negative_lesson(tmp_path):
    path = tmp_path / "learning.jsonl"
    store = LearningStore(path)
    prediction_id = store.record_prediction(
        module_name="alpha",
        symbol="BTC/USDT",
        signal="BUY",
        confidence=0.8,
        regime="high_range",
    )
    store.record_outcome(prediction_id=prediction_id, pnl=-10, reason="slippage")

    reloaded = LearningStore(path)
    assert reloaded.module_stats("alpha")["losses"] == 1
    assert reloaded.mistakes("alpha")[0]["lesson"] == "slippage"


def test_market_regime_is_data_driven():
    detector = MarketRegimeDetector(window=30, minimum_samples=5)
    regime = detector.detect({"close": [100 + index for index in range(30)]})
    assert regime.label == "low_bull"
    assert regime.direction == "bull"


def test_low_risk_code_is_approved_but_not_live_source(tmp_path):
    class Module:
        module_name = "safe_alpha"
        config = {}

        class Health:
            status = "ok"
            error_count = 0
            last_error = None

        health = Health()

    engine = SelfCodingEngine(
        project_root=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        policy=ApprovalPolicy(auto_approve_low_risk=True, auto_apply_low_risk=True),
    )
    result = engine.run_for_module(Module(), context={"regime": "low_bull"})
    assert result["status"] == ProposalStatus.APPLIED.value
    assert Path(result["artifact_path"]).is_file()
    assert Path(result["test_path"]).is_file()
    assert result["test_execution"]["passed"] is True
    assert result["source_path"] != result["artifact_path"]


def test_validator_rejects_dangerous_generated_code():
    report = SafeCodeValidator().validate("import os\nos.system('bad')")
    assert not report.passed
    assert report.policy_valid is False


def test_organism_wires_modules_and_emits_regime(tmp_path):
    bus = EventBus()
    events = []
    bus.subscribe("MARKET_REGIME", lambda event: events.append(event.payload))
    organism = Organism(
        OrganismConfig(
            module_packages=("autonomy",),
            enable_self_improvement=False,
            learning_path=str(tmp_path / "learning.jsonl"),
            self_coding_dir=str(tmp_path / "artifacts"),
            log_dir=str(tmp_path / "logs"),
        ),
        event_bus=bus,
    )
    wired = organism.discover_and_wire()
    result = organism.generate_consensus_signal(
        "BTC/USDT", {"close": [100 + index for index in range(40)]}
    )
    assert "momentum_alpha" in wired["active"]
    assert result["regime"] == "low_bull"
    assert events
    organism.stop()
