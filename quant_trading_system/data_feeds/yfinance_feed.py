"""YFinance feed with local CSV caching and retry/backoff."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import time

import pandas as pd
import yfinance as yf

from quant_trading_system.config import settings, ensure_cache_dir

log = logging.getLogger(__name__)
CACHE_ROOT = settings.data_cache_dir


def _cache_path(symbol: str, start: str, end: str) -> Path:
    return CACHE_ROOT / f"{symbol}_{start}_{end}.csv"


def _is_fresh(path: Path, max_age_days: int) -> bool:
    if not path.is_file():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age < timedelta(days=max_age_days)


def _download(symbol: str, start: str, end: str) -> pd.DataFrame:
    delay = 1
    for attempt in range(1, 4):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df.empty:
                raise ValueError(f"yfinance returned no data for {symbol}")
            return df
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise RuntimeError(f"Failed to download {symbol} after 3 attempts") from exc
            log.warning(
                "yfinance download failed for %s attempt %d/3: %s; retrying in %ss",
                symbol,
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2


def get_price_history(
    symbol: str,
    start: str,
    end: str,
    *,
    max_age_days: int | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    max_age = settings.max_price_age_days if max_age_days is None else max_age_days
    cache_file = _cache_path(symbol, start, end)

    if not force_download and _is_fresh(cache_file, max_age):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    data = _download(symbol, start, end)
    if getattr(data.index, "tz", None) is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    ensure_cache_dir()
    data.to_csv(cache_file)
    return data
