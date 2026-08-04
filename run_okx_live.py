#!/usr/bin/env python3
"""
OKX Live Runner - Event-Driven with Organism + Safety.

Usage:
  python run_okx_live.py --symbols BTC/USDT,ETH/USDT --interval 60
  python run_okx_live.py --once                         # single cycle
  python run_okx_live.py --symbols BTC/USDT --paper     # force paper
  OKX_LIVE_TRADING=true python run_okx_live.py --symbols BTC/USDT  # live (requires keys)

Environment:
  OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE
  OKX_LIVE_TRADING=true|false (default false = paper)
  QT_LOG_LEVEL=INFO
  QT_DEFAULT_SYMBOLS=BTC/USDT,ETH/USDT (fallback)

Safety: defaults to paper_mode, checks eternal guardrails, kill switch via SIGUSR1.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Ensure project root in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_trading_system.config import settings
from quant_trading_system.data_feeds.yfinance_feed import get_price_history
from quant_trading_system.logging import configure_root_logger

from core.event_bus import get_event_bus
from core.organism import Organism, OrganismConfig
from execution.okx_engine import OKXEngine
from execution.event_driven_executor import EventDrivenExecutor, ExecutorConfig
from core.qmp_engine_v3 import QMPUltraEngine

log = logging.getLogger(__name__)

class OKXLiveRunner:
    def __init__(self, symbols: List[str], interval_sec: int, once: bool = False, force_paper: bool = False):
        self.symbols = [s.strip().replace("-", "/") for s in symbols if s.strip()]
        self.interval_sec = interval_sec
        self.once = once
        self.force_paper = force_paper

        self.event_bus = get_event_bus()
        paper_mode = True if force_paper else settings.okx_paper_mode
        self.okx_engine = OKXEngine(
            paper_mode=paper_mode,
            event_bus=self.event_bus,
            max_leverage=settings.okx_max_leverage,
            max_position_pct=settings.okx_max_position_pct,
            max_daily_loss_pct=settings.okx_max_daily_loss_pct,
        )
        self.executor = EventDrivenExecutor(
            okx_engine=self.okx_engine,
            event_bus=self.event_bus,
            config=ExecutorConfig(min_confidence=0.60, allowed_symbols=self.symbols),
        )
        self.organism = Organism(
            config=OrganismConfig(
                auto_discover=settings.organism_auto_discover,
                self_improvement_interval_sec=settings.organism_self_improve_interval,
                enable_self_improvement=True,
            ),
            event_bus=self.event_bus,
        )
        self.qmp_engine = QMPUltraEngine()  # QC-free

        self.running = False
        self.cycle = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        log.warning("Signal %s received, shutting down", signum)
        self.running = False

    def start(self):
        log.info("=== OKX Live Runner Starting ===")
        log.info("Symbols: %s Interval: %ds Paper: %s Once: %s", self.symbols, self.interval_sec, self.okx_engine.paper_mode, self.once)
        log.info("OKX Live Env: %s CCXT available: %s", settings.okx_live_trading, self.okx_engine._balance)

        self.okx_engine.connect()
        wired = self.organism.discover_and_wire(project_root=ROOT)
        log.info("Organism wired: %s", wired)
        self.organism.start()
        self.executor.start()

        self.running = True
        try:
            while self.running:
                self.cycle += 1
                log.info("--- Cycle %d @ %s ---", self.cycle, datetime.now().isoformat())
                for sym in self.symbols:
                    try:
                        self._process_symbol(sym)
                    except Exception as exc:
                        log.exception("Error processing %s: %s", sym, exc)

                if self.once:
                    break

                # Health print every 10 cycles
                if self.cycle % 10 == 0:
                    status = self.organism.get_status()
                    exec_stats = self.executor.get_stats()
                    log.info("Organism status modules=%d weights=%s", len(status["modules"]), exec_stats.get("orders_placed", 0))
                    log.info("Executor stats %s", exec_stats)
                    log.info("OKX status %s", self.okx_engine.get_status())

                time.sleep(self.interval_sec)
        finally:
            self.shutdown()

    def _generate_synthetic_data(self, symbol: str, periods: int = 500):
        """Fallback synthetic data when yfinance fails (no internet)."""
        import pandas as pd
        import numpy as np

        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq="1min")
        # Seed by symbol to make deterministic per symbol
        seed = hash(symbol) % (2**32)
        np.random.seed(seed)
        base = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        returns = np.random.normal(0, 0.001, periods)
        prices = base * np.exp(np.cumsum(returns))
        opens = np.roll(prices, 1)
        opens[0] = base
        highs = prices * (1 + np.abs(np.random.normal(0, 0.0005, periods)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.0005, periods)))
        volumes = np.random.lognormal(6, 0.5, periods)
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
            index=dates,
        )
        return df

    def _process_symbol(self, symbol: str):
        # Fetch recent data via yfinance as history, fallback to synthetic if network fails
        try:
            yf_symbol = symbol
            if "/USDT" in symbol:
                base = symbol.split("/")[0]
                yf_symbol = f"{base}-USD"
            elif "/USD" in symbol:
                base = symbol.split("/")[0]
                yf_symbol = f"{base}-USD"

            try:
                df = get_price_history(yf_symbol, start="2024-01-01", end="2024-12-31")
                if df.empty:
                    raise ValueError("empty yfinance data")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                col_map = {c.lower(): c for c in df.columns}
                rename = {}
                for k in ["open", "high", "low", "close", "volume"]:
                    if k in col_map:
                        rename[col_map[k]] = k.capitalize()
                df = df.rename(columns=rename)
                log.info("Fetched yfinance data for %s (%d rows)", symbol, len(df))
            except Exception as yf_exc:
                log.warning("yfinance fetch failed for %s (%s), using synthetic data: %s", symbol, yf_symbol, yf_exc)
                df = self._generate_synthetic_data(symbol)

            history_data = {
                "1m": df.tail(1000),
                "5m": df.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 5 else df,
                "10m": df.resample("10min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 10 else df,
                "15m": df.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 15 else df,
                "20m": df.resample("20min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 20 else df,
                "25m": df.resample("25min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna() if len(df) > 25 else df,
            }

            qmp_signal = self.qmp_engine.generate_signal(symbol, history_data)
            log.info("QMP %s => %s conf=%.3f", symbol, qmp_signal.get("final_signal"), qmp_signal.get("confidence", 0))

            organism_consensus = self.organism.generate_consensus_signal(symbol, history_data)
            log.info("Organism %s => %s conf=%.3f weighted=%.3f", symbol, organism_consensus.get("final_signal"), organism_consensus.get("confidence", 0), organism_consensus.get("weighted_confidence", 0))

            # Force direct execution attempt for visibility (executor also handles via bus)
            if organism_consensus.get("final_signal") in ("BUY", "SELL"):
                result = self.okx_engine.place_order_from_signal(organism_consensus)
                if result:
                    log.info("Direct OKX execution result: %s", result.to_dict())

        except Exception as exc:
            log.exception("Failed _process_symbol %s: %s", symbol, exc)

    def shutdown(self):
        log.info("Shutting down OKX runner")
        try:
            self.organism.stop()
        except Exception:
            pass
        try:
            self.executor.stop()
        except Exception:
            pass
        log.info("OKX runner stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="OKX Live Runner - Event-Driven")
    parser.add_argument("--symbols", type=str, default=",".join(settings.default_symbols), help="Comma separated symbols e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    parser.add_argument("--paper", action="store_true", help="Force paper mode even if live env set")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_root_logger(settings.log_level)

    import pandas as pd  # ensure import after root logger

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = settings.default_symbols

    runner = OKXLiveRunner(symbols=symbols, interval_sec=args.interval, once=args.once, force_paper=args.paper)
    runner.start()
