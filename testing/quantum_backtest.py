def run_parallel_backtest(strategy):
    """Tests across 9 parallel market timelines"""
    results = {}
    for timeline in range(1, 10):
        with QuantumTunnel(timeline) as qt:
            data = qt.fetch_history()
            score = strategy.test(data)
            results[f"Timeline_{timeline}"] = score
    return results
