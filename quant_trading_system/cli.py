"""CLI for quick backtest-style data access."""

from __future__ import annotations

import argparse

from quant_trading_system.config import settings
from quant_trading_system.data_feeds.yfinance_feed import get_price_history
from quant_trading_system.logging import configure_root_logger


def main() -> None:
    parser = argparse.ArgumentParser(prog="quant-trading")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("fetch", help="Fetch and print OHLCV row count")
    bt.add_argument("symbol")
    bt.add_argument("start")
    bt.add_argument("end")

    args = parser.parse_args()
    configure_root_logger(settings.log_level)

    if args.cmd == "fetch":
        df = get_price_history(args.symbol, args.start, args.end)
        print(f"rows={len(df)} symbol={args.symbol}")


if __name__ == "__main__":
    main()
