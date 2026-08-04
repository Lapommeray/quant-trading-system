"""
OKX Live Runner - Real trading, fails closed.

No --paper flag, no synthetic fallback. If real data or credentials missing, exits with error.

Usage:
  OKX_API_KEY=... OKX_API_SECRET=... OKX_PASSPHRASE=... python -m okx_live.runner --symbols BTC/USDT,ETH/USDT --interval 60
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomy import Organism, OrganismConfig
from autonomy.execution import AutonomousExecutor, ExecutorConfig
from okx_live.engine import OKXLiveEngine
from core.event_bus import get_event_bus

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_real_history(symbol: str) -> dict:
    """
    Fetch real history - fails closed if not available.
    No synthetic fallback.
    """
    try:
        from quant_trading_system.data_feeds.yfinance_feed import get_price_history

        # Map crypto to yfinance format
        yf_symbol = symbol
        if "/USDT" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"
        elif "/USD" in symbol:
            yf_symbol = f"{symbol.split('/')[0]}-USD"

        df = get_price_history(yf_symbol, start="2024-01-01", end="2024-12-31")
        if df.empty:
            raise RuntimeError(f"yfinance returned empty data for {yf_symbol} - fail-closed, no synthetic fallback")

        if hasattr(df.columns, "get_level_values") and isinstance(df.columns, type(df.columns)):
            try:
                import pandas as pd

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            except Exception:
                pass

        # Normalize columns
        col_map = {c.lower(): c for c in df.columns}
        rename = {}
        for k in ["open", "high", "low", "close", "volume"]:
            if k in col_map:
                rename[col_map[k]] = k.capitalize()
        df = df.rename(columns=rename)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            raise RuntimeError(f"yfinance data missing required columns {required} - fail-closed")

        import pandas as pd

        history = {
            "1m": df.tail(1000),
            "5m": df.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(df) > 5
            else df,
            "10m": df.resample("10min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(df) > 10
            else df,
            "15m": df.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(df) > 15
            else df,
            "20m": df.resample("20min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(df) > 20
            else df,
            "25m": df.resample("25min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            if len(df) > 25
            else df,
        }

        return history

    except Exception as exc:
        raise RuntimeError(f"Failed to get real market data for {symbol} - fail-closed, no synthetic fallback: {exc}") from exc


class OKXLiveRunner:
    """Real trading runner - fails closed."""

    def __init__(self, symbols: List[str], interval_sec: int, once: bool = False):
        self.symbols = symbols
        self.interval_sec = interval_sec
        self.once = once
        self.event_bus = get_event_bus()
        # Real engine, fails closed if credentials/ccxt missing
        self.okx_engine = OKXLiveEngine(event_bus=self.event_bus)
        self.executor = AutonomousExecutor(
            okx_engine=self.okx_engine, event_bus=self.event_bus, config=ExecutorConfig(min_confidence=0.65)
        )
        self.organism = Organism(event_bus=self.event_bus)
        self.running = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        log.warning("Signal %s received, shutting down", signum)
        self.running = False

    def start(self):
        log.info("=== OKX LIVE REAL TRADING RUNNER (fail-closed) ===")
        log.info("Symbols: %s Interval: %d", self.symbols, self.interval_sec)

        # Will raise if credentials or ccxt missing
        self.okx_engine.connect()

        wired = self.organism.discover_and_wire()
        log.info("Organism wired: %s", wired)

        self.organism.start()
        self.executor.start()

        self.running = True
        try:
            cycle = 0
            while self.running:
                cycle += 1
                log.info("--- Cycle %d @ %s ---", cycle, datetime.now().isoformat())
                for sym in self.symbols:
                    try:
                        history = get_real_history(sym)
                        consensus = self.organism.generate_consensus_signal(sym, history)
                        log.info(
                            "Consensus %s => %s conf=%.3f weighted=%.3f",
                            sym,
                            consensus.get("final_signal"),
                            consensus.get("confidence", 0),
                            consensus.get("weighted_confidence", 0),
                        )
                    except Exception as exc:
                        log.error("Failed processing %s: %s - fail-closed will abort if critical", sym, exc)
                        raise

                if self.once:
                    break
                time.sleep(self.interval_sec)
        finally:
            self.organism.stop()
            self.executor.stop()
            log.info("Runner stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="OKX Live Real Trading Runner (fail-closed)")
    parser.add_argument("--symbols", type=str, required=True, help="Comma separated, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Single cycle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided - fail-closed")

    runner = OKXLiveRunner(symbols=symbols, interval_sec=args.interval, once=args.once)
    runner.start()
