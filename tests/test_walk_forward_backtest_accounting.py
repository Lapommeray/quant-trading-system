import pandas as pd

from backtest.walk_forward_quantum_backtest import WalkForwardQuantumBacktester


def _close_frame(values):
    return pd.DataFrame(
        {"Close": values},
        index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"),
    )


def _force_signal(backtester: WalkForwardQuantumBacktester, signal: int) -> None:
    backtester._get_tremor_scanner_prediction = lambda data: signal
    backtester._get_spectral_fusion_prediction = lambda data: signal
    backtester._get_dna_breath_prediction = lambda data: signal


def test_no_trade_signal_is_not_counted_as_loss():
    backtester = WalkForwardQuantumBacktester()
    _force_signal(backtester, 0)

    results = backtester._test_quantum_models(_close_frame([100, 110, 90]))
    aggregate = backtester._aggregate_results(results)

    assert results == []
    assert aggregate["total_trades"] == 0
    assert aggregate["losing_trades"] == 0


def test_short_winner_uses_signed_return_for_profit():
    backtester = WalkForwardQuantumBacktester()
    _force_signal(backtester, -1)

    results = backtester._test_quantum_models(_close_frame([100, 90]))
    aggregate = backtester._aggregate_results(results)

    assert results[0]["price_change"] == -0.1
    assert results[0]["signed_return"] == 0.1
    assert aggregate["winning_trades"] == 1
    assert aggregate["avg_profit"] == 0.1


def test_losses_are_signed_and_not_reported_as_profit():
    backtester = WalkForwardQuantumBacktester()
    _force_signal(backtester, 1)

    results = backtester._test_quantum_models(_close_frame([100, 90]))
    aggregate = backtester._aggregate_results(results)

    assert results[0]["signed_return"] == -0.1
    assert aggregate["winning_trades"] == 0
    assert aggregate["losing_trades"] == 1
    assert aggregate["avg_loss"] == -0.1
