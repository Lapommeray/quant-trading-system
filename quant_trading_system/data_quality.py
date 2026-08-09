"""Fail-closed market-data validation helpers.

Trading code must never silently consume malformed prices.  These utilities
normalize provider output only where the transformation is lossless
(e.g. timezone conversion) and raise :class:`DataQualityError` for any data
that could create false signals, look-ahead leakage, or bad risk decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    """Raised when market data is missing, impossible, stale, or malformed."""


_CANONICAL_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


def _canonical_column_name(column: Any) -> str:
    """Return a stable OHLCV column name, including yfinance MultiIndex cols."""
    if isinstance(column, tuple):
        # yfinance can emit either ('Close', 'AAPL') or ('AAPL', 'Close').
        for part in column:
            key = str(part).strip().lower()
            if key in _CANONICAL_COLUMNS:
                return _CANONICAL_COLUMNS[key]
        return "_".join(str(part).strip() for part in column)

    key = str(column).strip().lower()
    return _CANONICAL_COLUMNS.get(key, str(column).strip())


def _require_finite_numeric(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        bad = int(numeric.isna().sum())
        raise DataQualityError(f"{name} contains {bad} missing/non-numeric value(s)")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataQualityError(f"{name} contains non-finite value(s)")
    return numeric


def normalize_ohlcv_frame(
    df: pd.DataFrame,
    *,
    symbol: str = "UNKNOWN",
    required_columns: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
    max_future_skew_seconds: float = 300.0,
) -> pd.DataFrame:
    """Validate and return a UTC-indexed OHLCV frame.

    Checks performed:
    * non-empty ``DatetimeIndex``;
    * ascending, duplicate-free timestamps to prevent look-ahead leakage;
    * no timestamps materially in the future;
    * required OHLCV columns present;
    * finite numeric prices/volume;
    * strictly positive OHLC prices;
    * ``High >= max(Open, Close)``, ``Low <= min(Open, Close)``, ``High >= Low``;
    * non-negative volume.

    The returned frame is a copy.  The only normalization is canonical column
    naming and UTC timezone conversion/localization.
    """
    if df is None or df.empty:
        raise DataQualityError(f"{symbol}: empty OHLCV data")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError(f"{symbol}: OHLCV index must be a DatetimeIndex")

    out = df.copy()
    out.columns = [_canonical_column_name(col) for col in out.columns]

    missing = [col for col in required_columns if col not in out.columns]
    if missing:
        raise DataQualityError(f"{symbol}: missing required column(s): {', '.join(missing)}")

    if not out.index.is_monotonic_increasing:
        raise DataQualityError(f"{symbol}: timestamps must be strictly ascending")
    if out.index.has_duplicates:
        raise DataQualityError(f"{symbol}: duplicate timestamps detected")

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    now = pd.Timestamp(datetime.now(timezone.utc))
    latest = out.index.max()
    if latest > now + pd.Timedelta(seconds=max_future_skew_seconds):
        raise DataQualityError(f"{symbol}: latest timestamp {latest} is in the future")

    for col in required_columns:
        out[col] = _require_finite_numeric(out[col], f"{symbol}.{col}")

    price_cols = [col for col in ("Open", "High", "Low", "Close") if col in out.columns]
    if (out[price_cols] <= 0).any().any():
        raise DataQualityError(f"{symbol}: OHLC prices must be strictly positive")

    if "Volume" in out.columns and (out["Volume"] < 0).any():
        raise DataQualityError(f"{symbol}: volume must be non-negative")

    if {"Open", "High", "Low", "Close"}.issubset(out.columns):
        if (out["High"] < out[["Open", "Close"]].max(axis=1)).any():
            raise DataQualityError(f"{symbol}: high is below open/close")
        if (out["Low"] > out[["Open", "Close"]].min(axis=1)).any():
            raise DataQualityError(f"{symbol}: low is above open/close")
        if (out["High"] < out["Low"]).any():
            raise DataQualityError(f"{symbol}: high is below low")

    return out


def validate_market_tick(row: Mapping[str, Any], *, symbol: str = "UNKNOWN") -> dict[str, float]:
    """Validate a single live quote/bar row and return numeric fields.

    Raises ``DataQualityError`` instead of letting an impossible tick reach a
    DataRing, event bus, broker adapter, or risk engine.
    """
    def as_float(key: str, default: float | None = None) -> float:
        raw = row.get(key, default)
        if raw is None:
            raise DataQualityError(f"{symbol}: missing {key}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise DataQualityError(f"{symbol}: {key} is non-numeric") from exc
        if not np.isfinite(value):
            raise DataQualityError(f"{symbol}: {key} is non-finite")
        return value

    price = as_float("price")
    if price <= 0:
        raise DataQualityError(f"{symbol}: price must be positive")

    open_ = as_float("open", price)
    high = as_float("high", price)
    low = as_float("low", price)
    volume = as_float("volume", 0.0)
    ts = as_float("ts", datetime.now(timezone.utc).timestamp())

    if min(open_, high, low) <= 0:
        raise DataQualityError(f"{symbol}: OHLC fields must be positive")
    if high < max(open_, price):
        raise DataQualityError(f"{symbol}: high is below open/price")
    if low > min(open_, price):
        raise DataQualityError(f"{symbol}: low is above open/price")
    if high < low:
        raise DataQualityError(f"{symbol}: high is below low")
    if volume < 0:
        raise DataQualityError(f"{symbol}: volume must be non-negative")

    now_ts = datetime.now(timezone.utc).timestamp()
    if ts > now_ts + 300.0:
        raise DataQualityError(f"{symbol}: tick timestamp is in the future")

    return {"ts": ts, "price": price, "open": open_, "high": high, "low": low, "volume": volume}


__all__ = ["DataQualityError", "normalize_ohlcv_frame", "validate_market_tick"]
