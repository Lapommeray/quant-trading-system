#!/usr/bin/env python3
"""
Quant Trading System - Main Entry Point (Real Trading, Fail-Closed)

Architecture (per expected checkout):
  autonomy/organism.py
  autonomy/consensus.py
  autonomy/execution.py
  autonomy/events.py
  okx_live/config.py
  okx_live/trader.py

No QuantConnect. Fails closed if real market data or credentials missing.
No synthetic fallback, no --paper flag.
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

from autonomy import Organism, OrganismConfig
from okx_live.config import OKXLiveConfig, get_okx_config
from okx_live.trader import OKXLiveTrader
from autonomy.organism import get_event_bus

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_real_market_data(symbol: str) -> dict:
    """Real market data only - fails closed, no synthetic."""
    try:
        from quant_trading_system.data_feeds.yfinance_feed import get_price_history
        import pandas as pd

        yf_symbol = symbol
        if "/USDT" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"
        elif "/USD" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"

        df = get_price_history(yf_symbol, start="2024-01-01", end="2024-12-31")
        if df.empty:
            raise RuntimeError(f"yfinance empty for {symbol} ({yf_symbol}) - fail-closed, no synthetic")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        col_map = {c.lower(): c for c in df.columns}
        rename = {}
        for k in ["open", "high", "low", "close", "volume"]:
            if k in col_map:
                rename[col_map[k]] = k.capitalize()
        df = df.rename(columns=rename)

        for req in ["Open", "High", "Low", "Close", "Volume"]:
            if req not in df.columns:
                raise RuntimeError(f"Missing column {req} - fail-closed")

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
        raise RuntimeError(f"Real market data failed for {symbol} - fail-closed: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Quant Trading System - Real Trading (Fail-Closed)")
    parser.add_argument("--symbols", type=str, required=True, help="Comma separated, e.g. BTC/USDT,ETH/USDT - required, fail-closed if missing")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Single cycle")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        log.error("No symbols provided - fail-closed")
        sys.exit(1)

    # Validate real trading prereqs - fail-closed
    try:
        cfg = OKXLiveConfig.from_env()
        cfg.validate_for_real_trading()
    except Exception as exc:
        log.error(f"Prereq validation failed (fail-closed): {exc}")
        sys.exit(1)

    event_bus = get_event_bus()

    try:
        trader = OKXLiveTrader(config=cfg, event_bus=event_bus)
        trader.connect()
    except Exception as exc:
        log.error(f"OKX trader connect failed (fail-closed): {exc}")
        sys.exit(1)

    organism = Organism(event_bus=event_bus)
    wired = organism.discover_and_wire()
    log.info("Organism wired: %s", wired)

    from autonomy.execution import AutonomousExecutor, ExecutorConfig

    executor = AutonomousExecutor(okx_engine=trader, event_bus=event_bus, config=ExecutorConfig(min_confidence=0.65, allowed_symbols=symbols))

    organism.start()
    executor.start()

    log.info("=== REAL TRADING MAIN STARTED (fail-closed) ===")
    log.info("Symbols: %s Interval: %d", symbols, args.interval)

    try:
        cycle = 0
        while True:
            cycle += 1
            log.info("--- Cycle %d ---", cycle)
            for sym in symbols:
                try:
                    history = get_real_market_data(sym)
                    consensus = organism.generate_consensus_signal(sym, history)
                    log.info("Consensus %s => %s conf=%.3f weighted=%.3f", sym, consensus.get("final_signal"), consensus.get("confidence", 0), consensus.get("weighted_confidence", 0))
                except Exception as exc:
                    log.error(f"Processing {sym} failed (fail-closed): {exc}")
                    raise

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        organism.stop()
        executor.stop()


if __name__ == "__main__":
    main()
