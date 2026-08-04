#!/usr/bin/env python3
"""
Quant Trading System - Main Entry Point (Real Trading, Fail-Closed)

Post PR #117 autonomous organism with OKX live execution.
This entry point REQUIRES:
- Real market data (yfinance or OKX ticker) - no synthetic fallback
- Real OKX credentials for live trading (ccxt + API keys)
- No QuantConnect dependency

Architecture:
  autonomy/organism.py  -> auto-discovery, self-improvement, health
  autonomy/consensus.py -> weighted voting
  autonomy/execution.py -> event-driven routing to okx_live
  okx_live/engine.py    -> real OKX engine, fail-closed, no simulation
  okx_live/safety.py    -> eternal guardrails, kill switch

For simulation/paper testing, use execution/okx_engine.py explicitly (marked simulation).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.event_bus import get_event_bus, reset_event_bus
from quant_trading_system.config import settings
from quant_trading_system.logging import configure_root_logger

log = logging.getLogger(__name__)


def validate_real_trading_prereqs():
    """Fails closed if real trading prereqs missing."""
    missing = []

    # ccxt required for real OKX
    try:
        import ccxt  # noqa: F401
    except ImportError:
        missing.append("ccxt (pip install ccxt)")

    # Check credentials if live trading requested
    live_requested = os.getenv("OKX_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    if live_requested:
        if not os.getenv("OKX_API_KEY"):
            missing.append("OKX_API_KEY env var")
        if not os.getenv("OKX_API_SECRET"):
            missing.append("OKX_API_SECRET env var")
        if not os.getenv("OKX_PASSPHRASE"):
            missing.append("OKX_PASSPHRASE env var")

    if missing:
        raise RuntimeError(
            f"Real trading prerequisites missing (fail-closed): {', '.join(missing)} - "
            f"For simulation, use execution/okx_engine.py explicitly or run with OKX_ALLOW_PAPER_FOR_TEST=true"
        )


def get_real_market_data(symbol: str) -> dict:
    """Fetch real market data - fails closed, no synthetic fallback."""
    try:
        from quant_trading_system.data_feeds.yfinance_feed import get_price_history
        import pandas as pd

        # Map crypto to yfinance
        yf_symbol = symbol
        if "/USDT" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"
        elif "/USD" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"

        df = get_price_history(yf_symbol, start="2024-01-01", end="2024-12-31")
        if df.empty:
            raise RuntimeError(f"yfinance empty for {symbol} ({yf_symbol}) - fail-closed")

        if hasattr(df.columns, "levels") or "MultiIndex" in str(type(df.columns)):
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            except Exception:
                pass

        col_map = {c.lower(): c for c in df.columns}
        rename = {}
        for k in ["open", "high", "low", "close", "volume"]:
            if k in col_map:
                rename[col_map[k]] = k.capitalize()
        df = df.rename(columns=rename)

        for req in ["Open", "High", "Low", "Close", "Volume"]:
            if req not in df.columns:
                raise RuntimeError(f"Missing required column {req} for {symbol} - fail-closed")

        history = {
            "1m": df.tail(1000),
            "5m": df.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 5 else df,
            "10m": df.resample("10min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 10 else df,
            "15m": df.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 15 else df,
            "20m": df.resample("20min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 20 else df,
            "25m": df.resample("25min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 25 else df,
        }

        return history

    except Exception as exc:
        raise RuntimeError(f"Real market data fetch failed for {symbol} - fail-closed, no synthetic fallback: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Quant Trading System - Real Trading (Fail-Closed)")
    parser.add_argument("--symbols", type=str, default=",".join(settings.default_symbols), help="Comma separated, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Single cycle")
    parser.add_argument("--live", action="store_true", help="Require real OKX live trading (fail-closed if credentials missing)")
    args = parser.parse_args()

    configure_root_logger(settings.log_level)

    if args.live:
        os.environ["OKX_LIVE_TRADING"] = "true"

    # Validate prereqs - fails closed
    try:
        validate_real_trading_prereqs()
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        log.error("No symbols provided - fail-closed")
        sys.exit(1)

    # Import autonomy/okx_live after validation - these modules themselves fail closed
    try:
        from autonomy import Organism, OrganismConfig
        from autonomy.execution import AutonomousExecutor, ExecutorConfig
        from okx_live.engine import OKXLiveEngine
    except Exception as exc:
        log.error(f"Failed to import autonomy/okx_live (fail-closed): {exc}")
        sys.exit(1)

    event_bus = get_event_bus()

    try:
        okx_engine = OKXLiveEngine(event_bus=event_bus)
        okx_engine.connect()
    except Exception as exc:
        log.error(f"OKXLiveEngine connect failed (fail-closed): {exc}")
        sys.exit(1)

    organism = Organism(config=OrganismConfig(auto_discover=True, enable_self_improvement=True), event_bus=event_bus)
    wired = organism.discover_and_wire(project_root=ROOT)
    log.info("Organism wired: %s", wired)

    if not wired["active"]:
        log.warning("No active modules discovered - will produce no signals (fail-closed safe)")

    executor = AutonomousExecutor(okx_engine=okx_engine, event_bus=event_bus, config=ExecutorConfig(min_confidence=0.65, allowed_symbols=symbols))

    organism.start()
    executor.start()

    log.info("=== REAL TRADING MAIN STARTED (fail-closed) ===")
    log.info("Symbols: %s Interval: %d Live: %s", symbols, args.interval, args.live)

    try:
        cycle = 0
        while True:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)
            for sym in symbols:
                try:
                    history = get_real_market_data(sym)
                    consensus = organism.generate_consensus_signal(sym, history)
                    log.info(
                        "Consensus %s => %s conf=%.3f weighted=%.3f votes=%s",
                        sym,
                        consensus.get("final_signal"),
                        consensus.get("confidence", 0),
                        consensus.get("weighted_confidence", 0),
                        consensus.get("votes", {}),
                    )
                except Exception as exc:
                    log.error(f"Processing {sym} failed (fail-closed): {exc}")
                    # In real trading, we fail closed on data failure - do not continue with synthetic
                    if args.live:
                        raise

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted, shutting down")
    finally:
        organism.stop()
        executor.stop()


if __name__ == "__main__":
    main()
