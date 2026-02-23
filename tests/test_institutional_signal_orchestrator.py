from core.institutional_signal_orchestrator import InstitutionalSignalOrchestrator


def test_decide_buy_signal_with_risk_levels():
    orchestrator = InstitutionalSignalOrchestrator(confidence_threshold=0.6)
    decision = orchestrator.decide(price=100.0, trend_score=0.6, vol_score=0.2, liquidity_score=0.5)
    assert decision.action == "BUY"
    assert 0.6 <= decision.confidence <= 1.0
    assert decision.stop_loss < decision.entry < decision.take_profit


def test_decide_hold_when_confidence_low():
    orchestrator = InstitutionalSignalOrchestrator(confidence_threshold=0.9)
    decision = orchestrator.decide(price=100.0, trend_score=0.1, vol_score=0.8, liquidity_score=0.0)
    assert decision.action == "HOLD"
    assert decision.stop_loss == decision.entry == decision.take_profit


def test_learn_adjusts_threshold():
    orchestrator = InstitutionalSignalOrchestrator(confidence_threshold=0.62)
    orchestrator.learn({"sharpe": 2.0, "max_drawdown": 0.05})
    assert orchestrator.confidence_threshold < 0.62
    orchestrator.learn({"sharpe": 0.4, "max_drawdown": 0.2})
    assert orchestrator.confidence_threshold >= 0.61
