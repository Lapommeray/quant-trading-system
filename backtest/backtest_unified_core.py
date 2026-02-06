"""
Offline Verification of Unified AI Indicator Accuracy

Runs the full intelligence pipeline against historical data and measures:
- Signal accuracy (predicted direction vs actual price movement)
- Regime detection accuracy
- Self-learning weight evolution over time
- Per-indicator contribution scores

Usage:
    python backtest/backtest_unified_core.py
    python backtest/backtest_unified_core.py --symbol XAUUSD --period 1y
    python backtest/backtest_unified_core.py --symbol EURUSD --period 6mo --verbose
"""

import os
import sys
import argparse
import logging
import importlib.util
import time

import numpy as np
import pandas as pd

CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core")


def _load(filename, classname):
    path = os.path.join(CORE_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


FeatureEngineer = _load("feature_engineering.py", "FeatureEngineer")
DataPipeline = _load("data_pipeline.py", "DataPipeline")
UnifiedIntelligenceCore = _load("unified_core.py", "UnifiedIntelligenceCore")

logger = logging.getLogger("Backtest")


def run_backtest(
    symbol: str = "XAUUSD",
    period: str = "1y",
    interval: str = "1d",
    window: int = 50,
    verbose: bool = False,
):
    """
    Run full backtest of the unified intelligence core.

    Steps:
    1. Load historical OHLCV data
    2. Slide a window across the data
    3. At each position, feed the window to the unified core
    4. Compare the predicted direction to the actual next-bar movement
    5. Track accuracy, regime transitions, and weight evolution
    """
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST: {symbol} | period={period} | interval={interval}")
    print(f"{'=' * 60}\n")

    pipeline = DataPipeline()
    features = FeatureEngineer()
    data_dir = os.path.dirname(CORE_DIR)
    core = UnifiedIntelligenceCore(data_dir=data_dir)

    print(f"Fetching historical data for {symbol}...")
    df = pipeline.fetch_historical(symbol, period=period, interval=interval)

    if df is None or len(df) < window + 10:
        print(f"ERROR: Not enough data for {symbol}. Got {len(df) if df is not None else 0} bars, need {window + 10}")
        return None

    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Window size: {window} bars")
    print(f"Tradeable bars: {len(df) - window - 1}")
    print()

    total = 0
    correct = 0
    signals_emitted = 0
    buy_correct = 0
    buy_total = 0
    sell_correct = 0
    sell_total = 0
    hold_count = 0

    regime_counts = {"BULL": 0, "BEAR": 0, "RANGE": 0, "UNKNOWN": 0}
    results = []

    for i in range(window, len(df) - 1):
        window_df = df.iloc[i - window:i].copy()

        market_data = {
            "symbol": symbol,
            "source": "backtest",
            "ohlcv": window_df.reset_index(drop=False).to_dict("records"),
            "ohlcv_daily": window_df.reset_index(drop=False).to_dict("records"),
            "close": float(window_df["Close"].iloc[-1]),
        }

        result = core.generate_signal(symbol, market_data)
        signal = result.get("final_signal", "HOLD")
        confidence = result.get("confidence", 0.0)
        regime = result.get("regime", "UNKNOWN")

        current_close = float(df["Close"].iloc[i])
        next_close = float(df["Close"].iloc[i + 1])

        if current_close == 0:
            continue

        actual_change = (next_close - current_close) / current_close
        if actual_change > 0.0002:
            true_dir = "BUY"
        elif actual_change < -0.0002:
            true_dir = "SELL"
        else:
            true_dir = "HOLD"

        total += 1
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        is_correct = False
        if signal == "HOLD":
            hold_count += 1
            if true_dir == "HOLD":
                is_correct = True
                correct += 1
        elif signal == "BUY":
            signals_emitted += 1
            buy_total += 1
            if actual_change > 0:
                is_correct = True
                correct += 1
                buy_correct += 1
        elif signal == "SELL":
            signals_emitted += 1
            sell_total += 1
            if actual_change < 0:
                is_correct = True
                correct += 1
                sell_correct += 1

        results.append({
            "bar": i,
            "signal": signal,
            "confidence": confidence,
            "true_dir": true_dir,
            "change_pct": actual_change * 100,
            "correct": is_correct,
            "regime": regime,
        })

        if verbose and signal != "HOLD":
            mark = "WIN" if is_correct else "LOSS"
            print(
                f"  Bar {i:4d} | {signal:4s} conf={confidence:.3f} | "
                f"actual={actual_change * 100:+.3f}% | {mark} | regime={regime}"
            )

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {symbol}")
    print(f"{'=' * 60}")

    accuracy = 100 * correct / total if total > 0 else 0
    signal_accuracy = 0
    if signals_emitted > 0:
        directional_correct = (buy_correct + sell_correct)
        signal_accuracy = 100 * directional_correct / signals_emitted

    print(f"\n  Total bars evaluated:    {total}")
    print(f"  Signals emitted:         {signals_emitted} ({100 * signals_emitted / max(total, 1):.1f}%)")
    print(f"  HOLD decisions:          {hold_count}")
    print(f"  Overall accuracy:        {accuracy:.2f}%")
    print(f"  Directional accuracy:    {signal_accuracy:.2f}% (of emitted signals)")
    print(f"  BUY accuracy:            {100 * buy_correct / max(buy_total, 1):.2f}% ({buy_correct}/{buy_total})")
    print(f"  SELL accuracy:           {100 * sell_correct / max(sell_total, 1):.2f}% ({sell_correct}/{sell_total})")

    print(f"\n  Regime distribution:")
    for regime, count in sorted(regime_counts.items()):
        if count > 0:
            print(f"    {regime:8s}: {count:4d} ({100 * count / total:.1f}%)")

    core_stats = core.get_stats()
    if "learning" in core_stats:
        learn = core_stats["learning"]
        print(f"\n  Self-learning stats:")
        print(f"    Total evaluated:       {learn.get('total_evaluated', 0)}")
        print(f"    Correct predictions:   {learn.get('correct', 0)}")
        print(f"    Accuracy (self-track): {learn.get('accuracy_pct', 0):.2f}%")
        print(f"    Weights: {learn.get('weights', {})}")

    if "rl" in core_stats:
        rl = core_stats["rl"]
        print(f"\n  RL agent stats:")
        print(f"    States learned:        {rl.get('states_learned', 0)}")
        print(f"    Total decisions:       {rl.get('total_decisions', 0)}")
        print(f"    Epsilon:               {rl.get('epsilon', 'N/A')}")

    if "genetic" in core_stats:
        gen = core_stats["genetic"]
        print(f"\n  Genetic evolver stats:")
        print(f"    Generation:            {gen.get('generation', 0)}")
        print(f"    Best fitness:          {gen.get('best_fitness', 'N/A')}")

    print(f"\n{'=' * 60}")
    print(f"  [RESULT] {symbol} backtest accuracy: {accuracy:.3f}%")
    print(f"  [RESULT] Directional accuracy: {signal_accuracy:.3f}%")
    print(f"{'=' * 60}\n")

    return {
        "symbol": symbol,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "signal_accuracy": signal_accuracy,
        "signals_emitted": signals_emitted,
        "buy_accuracy": 100 * buy_correct / max(buy_total, 1),
        "sell_accuracy": 100 * sell_correct / max(sell_total, 1),
        "regime_counts": regime_counts,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Unified Intelligence Core")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--period", default="1y", help="Historical period (1mo, 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--interval", default="1d", help="Bar interval (1h, 1d)")
    parser.add_argument("--window", type=int, default=50, help="Lookback window size")
    parser.add_argument("--verbose", action="store_true", help="Print every signal")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_backtest(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        window=args.window,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
