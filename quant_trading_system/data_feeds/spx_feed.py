"""
S&P 500 (^GSPC) live feed - pre-broker equity-index data for the organism.

OKX does not list the S&P 500 index, so the equity leg comes from a separate
feed.  Two providers are supported:

* TwelveData (recommended, low-latency) when ``TWELVEDATA_API_KEY`` is set;
* yfinance ``^GSPC`` 1-minute polling otherwise (slower - 10s poll default,
  latency-dominated; fine for regime/risk context, NOT for tick timing).

Like the OKX feed, this layer is read-only and pre-broker: data lands in the
core DataRing (symbol ``SPX``) and on the event bus as ``MARKET_DATA`` /
``SPX_UPDATE`` before any broker order path can consume it.  Fail-closed:
if the fetch fails repeatedly the feed reports unhealthy and publishes
``RISK_ALERT``; no synthetic data is ever injected.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from quant_trading_system.data_quality import validate_market_tick

log = logging.getLogger(__name__)

try:
    from core.data_ring import DataRing, get_data_ring  # type: ignore
    from core.event_bus import EventBus, get_event_bus  # type: ignore

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    get_data_ring = None  # type: ignore
    get_event_bus = None  # type: ignore

RING_SYMBOL = "SPX"


def _fetch_yfinance() -> Dict[str, Any]:
    """Fetch the latest ^GSPC quote row from yfinance (1m bar)."""
    import yfinance as yf  # deferred import: optional dependency

    df = yf.download("^GSPC", period="1d", interval="1m", progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no S&P 500 data")
    last = df.iloc[-1]
    ts = last.name
    if hasattr(ts, "timestamp"):
        ts_epoch = ts.timestamp()
    else:
        ts_epoch = time.time()
    return {
        "ts": float(ts_epoch),
        "price": float(last["Close"]),
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "volume": float(last["Volume"]),
    }


def _fetch_twelvedata(api_key: str, poll_sec: int) -> Dict[str, Any]:
    """Fetch the latest S&P 500 price from TwelveData (1min interval)."""
    import requests  # deferred import

    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": "SPX",
            "interval": "1min",
            "outputsize": "1",
            "apikey": api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()
    values = payload.get("values") or []
    if not values:
        raise RuntimeError("TwelveData returned no S&P 500 data")
    row = values[0]
    return {
        "ts": time.time(),
        "price": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": float(row.get("volume") or 0.0),
    }


class SPXLiveFeed:
    """Polling live feed for the S&P 500 index, writing into the DataRing."""

    def __init__(
        self,
        event_bus: Optional[Any] = None,
        ring_factory: Callable[[str], Any] = None,
        fetch_func: Optional[Callable[[], Dict[str, Any]]] = None,
        poll_seconds: Optional[float] = None,
        api_key: Optional[str] = None,
        stale_after_sec: float = 60.0,
    ):
        self.event_bus = event_bus
        if self.event_bus is None and CORE_AVAILABLE and get_event_bus:
            self.event_bus = get_event_bus()
        self._ring_factory = ring_factory or (
            (lambda sym: get_data_ring(sym)) if CORE_AVAILABLE and get_data_ring else {}
        )
        self._api_key = (
            api_key if api_key is not None else os.getenv("TWELVEDATA_API_KEY", "")
        )
        self._poll_seconds = (
            poll_seconds
            if poll_seconds is not None
            else float(os.getenv("SPX_POLL_SECONDS", "10"))
        )
        self.stale_after_sec = stale_after_sec

        if fetch_func is not None:
            self._fetch_func = fetch_func
        elif self._api_key:
            self._fetch_func = lambda: _fetch_twelvedata(
                self._api_key, int(self._poll_seconds)
            )
        else:
            self._fetch_func = _fetch_yfinance

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_ok_ts = 0.0
        self._last_price = 0.0
        self._consecutive_failures = 0
        self._stats = {"updates": 0, "failures": 0, "alerts": 0}

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="SPXLiveFeed", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                row = self._fetch_func()
                self._ingest(row)
                self._last_ok_ts = time.time()
                self._consecutive_failures = 0
                self._stats["updates"] += 1
            except Exception as exc:  # noqa: BLE001
                self._stats["failures"] += 1
                self._consecutive_failures += 1
                log.warning("SPX feed fetch failed: %s", exc)
                if self._consecutive_failures >= 3 and self.event_bus is not None:
                    self._stats["alerts"] += 1
                    self.event_bus.publish(
                        "RISK_ALERT",
                        {
                            "reason": "spx_feed_failing",
                            "symbol": RING_SYMBOL,
                            "consecutive_failures": self._consecutive_failures,
                        },
                        source="SPXLiveFeed",
                    )
            time.sleep(self._poll_seconds)

    def _ingest(self, row: Dict[str, Any]) -> None:
        tick = validate_market_tick(row, symbol=RING_SYMBOL)
        price = tick["price"]
        self._last_price = price
        self._last_ok_ts = time.time()
        self._consecutive_failures = 0
        ring = self._ring_factory(RING_SYMBOL) if self._ring_factory else None
        if ring is not None and hasattr(ring, "push"):
            ring.push(
                tick["ts"],
                float(row.get("bid", price)),
                float(row.get("ask", price)),
                0.0,
                0.0,
                price,
                tick["volume"],
                0,
            )
        if self.event_bus is not None:
            self.event_bus.publish(
                "MARKET_DATA",
                {
                    "symbol": RING_SYMBOL,
                    "inst_id": "^GSPC",
                    "price": price,
                    "open": tick["open"],
                    "high": tick["high"],
                    "low": tick["low"],
                    "volume": tick["volume"],
                    "asset_class": "equity_index",
                    "source": "twelvedata" if self._api_key else "yfinance",
                    "pre_broker": True,
                },
                source="SPXLiveFeed",
            )
            self.event_bus.publish(
                "SPX_UPDATE",
                {"symbol": RING_SYMBOL, "price": price, "ts": tick["ts"]},
                source="SPXLiveFeed",
            )

    def feed_status(self) -> Dict[str, Any]:
        age = time.time() - self._last_ok_ts if self._last_ok_ts else float("inf")
        return {
            "running": self._running,
            "healthy": age <= self.stale_after_sec,
            "last_price": self._last_price,
            "data_age_sec": round(age, 2),
            "poll_seconds": self._poll_seconds,
            "provider": "twelvedata" if self._api_key else "yfinance",
            "stats": dict(self._stats),
        }


__all__ = ["SPXLiveFeed", "RING_SYMBOL"]
