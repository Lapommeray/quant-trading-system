from core.self_coder import StrategyGenerator


def test_self_coder_generates_institutional_strategy_code():
    generator = StrategyGenerator(algorithm=None)
    code = generator._synthesize_strategy_code({"volatility": "high", "trend": "strong_bull"})
    assert "class GeneratedInstitutionalStrategy" in code
    assert "stop = avg * (1 - 0.015)" in code
    assert "take = avg * (1 + 0.03)" in code
    assert "self.MarketOrder" in code


def test_backward_compat_method_is_local():
    generator = StrategyGenerator(algorithm=None)
    code = generator._call_openai_api("legacy")
    assert "GeneratedInstitutionalStrategy" in code
