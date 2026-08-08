"""Compatibility entry point for the fail-closed OKX live runner.

Use ``python -m okx_live.runner`` for the canonical module entry point.  This
script intentionally delegates to that runner and does not provide a paper or
synthetic-data fallback.
"""

from okx_live.runner import OKXLiveRunner, get_real_history, parse_args

if __name__ == "__main__":
    args = parse_args()
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided - fail-closed")
    OKXLiveRunner(symbols=symbols, interval_sec=args.interval, once=args.once).start()
