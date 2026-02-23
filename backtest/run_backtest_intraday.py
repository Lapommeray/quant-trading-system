"""
Intraday Walk-Forward Backtest

Sliding-window walk-forward validation for the unified intelligence core.
Trains on N days, tests on next 1 day, slides forward.

Measures:
- Directional accuracy per window
- Cumulative PnL with risk guard
- Sharpe/Sortino/max drawdown
- Self-learning weight evolution over time

Usage:
    python backtest/run_backtest_intraday.py
    python backtest/run_backtest_intraday.py --symbol XAUUSD --interval 5m
    python backtest/run_backtest_intraday.py --symbol XAUUSD --train-days 5 --test-days 1
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
TRAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")


def _load(directory, filename, classname):
    path = os.path.join(directory, filename)
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


DataPipeline = _load(CORE_DIR, "data_pipeline.py", "DataPipeline")
UnifiedIntelligenceCore = _load(CORE_DIR, "unified_core.py", "UnifiedIntelligenceCore")
PerformanceMetrics = _load(CORE_DIR, "metrics.py", "PerformanceMetrics")
RiskGuard = _load(CORE_DIR, "risk_guard.py", "RiskGuard")

logger = logging.getLogger("IntradayBacktest")


def run_walk_forward(
    symbol: str = "XAUUSD",
    interval: str = "1h",
    train_days: int = 5,
    test_days: int = 1,
    window: int = 50,
    verbose: bool = False,
):
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD INTRADAY BACKTEST")
    print(f"  {symbol} | interval={interval} | train={train_days}d | test={test_days}d")
    print(f"{'=' * 70}\n")

    pipeline = DataPipeline()
    data_dir = os.path.dirname(CORE_DIR)

    period_map = {
        "1m": "5d", "5m": "1mo", "15m": "1mo",
        "30m": "3mo", "1h": "3mo", "1d": "1y",
    }
    period = period_map.get(interval, "3mo")

    print(f"Fetching historical data ({period}, {interval})...")
    df = pipeline.fetch_historical(symbol, period=period, interval=interval)

    if df is None or len(df) < window + 20:
        print(f"ERROR: Not enough data. Got {len(df) if df is not None else 0} bars")
        return None

    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    bars_per_day_map = {"1m": 390, "5m": 78, "15m": 26, "30m": 13, "1h": 7, "1d": 1}
    bars_per_day = bars_per_day_map.get(interval, 7)
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day

    if train_bars + test_bars > len(df):
        train_bars = max(window + 5, len(df) // 3)
        test_bars = max(5, len(df) // 6)
        print(f"Adjusted: train={train_bars} bars, test={test_bars} bars")

    metrics = PerformanceMetrics()
    risk_guard = RiskGuard(max_risk_pct=1.0, max_dd_pct=5.0)

    all_results = []
    fold = 0
    start_idx = 0

    while start_idx + train_bars + test_bars <= len(df):
        fold += 1
        train_end = start_idx + train_bars
        test_end = train_end + test_bars

        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]

        core = UnifiedIntelligenceCore(data_dir=data_dir)

        for i in range(window, len(train_df) - 1):
            w = train_df.iloc[i - window:i].copy()
            market_data = {
                "symbol": symbol, "source": "backtest",
                "ohlcv": w.reset_index(drop=False).to_dict("records"),
                "ohlcv_daily": w.reset_index(drop=False).to_dict("records"),
                "close": float(w["Close"].iloc[-1]),
            }
            core.generate_signal(symbol, market_data)

        fold_correct = 0
        fold_total = 0
        fold_signals = 0

        for i in range(window, len(test_df) - 1):
            w = test_df.iloc[i - window:i].copy()
            market_data = {
                "symbol": symbol, "source": "backtest",
                "ohlcv": w.reset_index(drop=False).to_dict("records"),
                "ohlcv_daily": w.reset_index(drop=False).to_dict("records"),
                "close": float(w["Close"].iloc[-1]),
            }

            result = core.generate_signal(symbol, market_data)
            signal = result.get("final_signal", "HOLD")
            confidence = result.get("confidence", 0.0)

            current = float(test_df["Close"].iloc[i])
            next_close = float(test_df["Close"].iloc[i + 1])

            if current == 0:
                continue

            actual_change = (next_close - current) / current
            if actual_change > 0.0002:
                true_dir = "BUY"
            elif actual_change < -0.0002:
                true_dir = "SELL"
            else:
                true_dir = "HOLD"

            fold_total += 1
            pnl = 0.0

            if signal in ("BUY", "SELL"):
                fold_signals += 1
                metrics.record_prediction(signal, true_dir)

                if signal == "BUY":
                    pnl = actual_change * 100
                else:
                    pnl = -actual_change * 100

                if pnl > 0:
                    fold_correct += 1

                metrics.record_return(pnl / 100)
                metrics.record_trade({"pnl": pnl, "signal": signal, "true": true_dir})
                risk_guard.record_trade(pnl, symbol)
            elif true_dir == "HOLD":
                fold_correct += 1

            if verbose and signal != "HOLD":
                mark = "WIN" if pnl > 0 else "LOSS"
                print(
                    f"  Fold {fold} | Bar {i:4d} | {signal:4s} conf={confidence:.3f} | "
                    f"actual={actual_change * 100:+.3f}% | {mark}"
                )

        fold_acc = (fold_correct / fold_total * 100) if fold_total > 0 else 0
        signal_rate = (fold_signals / fold_total * 100) if fold_total > 0 else 0

        all_results.append({
            "fold": fold,
            "train_bars": len(train_df),
            "test_bars": fold_total,
            "signals": fold_signals,
            "accuracy": fold_acc,
            "signal_rate": signal_rate,
        })

        print(
            f"  Fold {fold}: test={fold_total} bars, signals={fold_signals}, "
            f"accuracy={fold_acc:.1f}%, signal_rate={signal_rate:.1f}%"
        )

        start_idx += test_bars

    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD RESULTS: {symbol}")
    print(f"{'=' * 70}")

    if all_results:
        accuracies = [r["accuracy"] for r in all_results]
        print(f"\n  Folds completed:      {len(all_results)}")
        print(f"  Mean accuracy:        {np.mean(accuracies):.2f}%")
        print(f"  Std accuracy:         {np.std(accuracies):.2f}%")
        print(f"  Min/Max accuracy:     {np.min(accuracies):.2f}% / {np.max(accuracies):.2f}%")

    summary = metrics.get_summary()
    print(f"\n  Performance Metrics:")
    print(f"    Sharpe ratio:       {summary['sharpe_ratio']}")
    print(f"    Sortino ratio:      {summary['sortino_ratio']}")
    print(f"    Max drawdown:       {summary['max_drawdown']}")
    print(f"    Win rate:           {summary['win_rate']}")
    print(f"    Profit factor:      {summary['profit_factor']}")
    print(f"    Expectancy:         {summary['expectancy']}")
    print(f"    BUY precision:      {summary['buy_precision']}")
    print(f"    SELL precision:     {summary['sell_precision']}")

    risk_status = risk_guard.get_status()
    print(f"\n  Risk Status:")
    print(f"    Equity:             {risk_status['equity']}")
    print(f"    Total PnL:          {risk_status['total_pnl']}")
    print(f"    Max drawdown pct:   {risk_status['current_dd_pct']}%")
    print(f"    Total trades:       {risk_status['total_trades']}")
    print(f"    Win rate:           {risk_status['win_rate']}%")

    print(f"\n{'=' * 70}\n")

    return {
        "folds": all_results,
        "metrics": summary,
        "risk": risk_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Intraday Backtest")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--interval", default="1h", help="Bar interval (1m, 5m, 15m, 1h)")
    parser.add_argument("--train-days", type=int, default=5, help="Training window in days")
    parser.add_argument("--test-days", type=int, default=1, help="Testing window in days")
    parser.add_argument("--window", type=int, default=50, help="Lookback window for features")
    parser.add_argument("--verbose", action="store_true", help="Print every signal")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run_walk_forward(
        symbol=args.symbol,
        interval=args.interval,
        train_days=args.train_days,
        test_days=args.test_days,
        window=args.window,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
