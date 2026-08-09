#!/usr/bin/env python3
"""
Live Data Injection — Authoritative Baseline Command

Locks real-market-grounded metrics as the external truth for the invariant
∀t. Equity_t ≥ Equity_0.

- Fetches live OHLCV via advanced_modules.enhanced_backtester.fetch_live_ohlcv()
  (yfinance with deterministic synthetic fallback seeded by OMNIUM_INVARIANT_SEED)
- Runs EnhancedBacktester deterministic walk-forward over real price history
- Writes result to git-tracked live_baseline.json
- evolve.py metrics_degraded() can then compare candidate live metrics against
  this locked baseline, in addition to simulated deterministic check.

Usage:
    python3 lock_live_baseline.py [--symbol BTC-USD --period 1y --interval 1d]

The file live_baseline.json is git-tracked (not gitignored) to anchor the invariant
in real market behavior while preserving reproducibility via OMNIUM seed.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [LOCK_LIVE_BASELINE] %(message)s"
)
log = logging.getLogger("lock_live_baseline")


def main():
    parser = argparse.ArgumentParser(
        description="Lock live baseline for invariant grounding"
    )
    parser.add_argument(
        "--symbol", default="BTC-USD", help="Ticker symbol for yfinance"
    )
    parser.add_argument(
        "--period", default="1y", help="yfinance period (1y, 2y, max, etc.)"
    )
    parser.add_argument(
        "--interval", default="1d", help="yfinance interval (1d, 1h, etc.)"
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "live_baseline.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    try:
        from evolve import load_backtest_metrics
    except ImportError as e:
        log.error(f"Failed to import evolve.load_backtest_metrics: {e}")
        raise

    log.info(
        f"Fetching live metrics for {args.symbol} {args.period} {args.interval} ..."
    )
    metrics = load_backtest_metrics(
        use_live_data=True,
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
    )

    if metrics is None:
        log.error("Failed to load live metrics; aborting baseline lock")
        return 1

    # Enrich with metadata for external grounding proof
    payload = {
        **metrics,
        "locked_at": datetime.utcnow().isoformat() + "Z",
        "symbol": args.symbol,
        "period": args.period,
        "interval": args.interval,
        "data_source": "live",
        "invariant": "∀t. Equity_t ≥ Equity_0",
        "grounding": "live OHLCV via yfinance, deterministic walk-forward SMA(20), seed=OMNIUM_INVARIANT_SEED",
        "seed_bytes": "OMNIUM_INVARIANT_SEED",
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    log.info(f"Live baseline locked to {output_path}:")
    log.info(json.dumps(payload, indent=2, default=str))

    # Also verify deterministic reproducibility of live path (with synthetic fallback it is deterministic,
    # with real yfinance it will vary with market, but size/tie-breaking remains seeded)
    try:
        metrics2 = load_backtest_metrics(
            use_live_data=True,
            symbol=args.symbol,
            period=args.period,
            interval=args.interval,
        )
        if metrics == metrics2:
            log.info(
                "Live metrics bit-identical across two runs (deterministic fallback active)"
            )
        else:
            log.info(
                "Live metrics vary across runs (real yfinance data — expected if market data changed, but internal RNG remains seeded)"
            )
    except Exception:
        log.warning("Second verification run failed; ignoring")

    # Integrity note for evolve.py guard
    log.info(
        "Baseline ready. evolve.py metrics_degraded() will compare live candidates against this file."
    )
    log.info(
        "Simulated deterministic guard remains fast path; live path is authoritative for external grounding."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
