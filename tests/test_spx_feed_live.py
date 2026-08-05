"""Tests for the S&P 500 pre-broker feed (offline, fake fetcher)."""

import time

from quant_trading_system.data_feeds.spx_feed import RING_SYMBOL, SPXLiveFeed


class _RingRegistry:
    def __init__(self):
        self.rings = {}

    def __call__(self, symbol):
        if symbol not in self.rings:
            from core.data_ring import get_data_ring

            self.rings[symbol] = get_data_ring(symbol, size=1000)
        return self.rings[symbol]


class _CaptureBus:
    def __init__(self):
        self.events = []

    def publish(self, event_type, payload, source="test", **kwargs):
        self.events.append({"event_type": event_type, "payload": payload, "source": source})


def _fake_fetch(price=5530.25):
    return {
        "ts": time.time(), "price": price, "open": 5520.0,
        "high": 5540.0, "low": 5515.0, "volume": 12_345.0,
    }


def test_ingest_writes_ring_and_publishes_market_data():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = SPXLiveFeed(event_bus=bus, ring_factory=registry, fetch_func=_fake_fetch)

    feed._ingest(feed._fetch_func())

    ring = registry.rings[RING_SYMBOL]
    last = ring.latest(1)[-1]
    assert float(last["price"]) == 5530.25

    md = [e for e in bus.events if e["event_type"] == "MARKET_DATA"]
    assert md and md[0]["payload"]["asset_class"] == "equity_index"
    assert md[0]["payload"]["pre_broker"] is True
    spx_updates = [e for e in bus.events if e["event_type"] == "SPX_UPDATE"]
    assert spx_updates and spx_updates[0]["payload"]["price"] == 5530.25


def test_rejects_non_positive_price():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = SPXLiveFeed(event_bus=bus, ring_factory=registry, fetch_func=lambda: _fake_fetch(0.0))
    try:
        feed._ingest(feed._fetch_func())
        assert False, "should have raised"
    except ValueError:
        pass
    assert feed.feed_status()["last_price"] == 0.0


def test_feed_status_healthy():
    registry = _RingRegistry()
    feed = SPXLiveFeed(event_bus=_CaptureBus(), ring_factory=registry, fetch_func=_fake_fetch)
    feed._ingest(feed._fetch_func())
    status = feed.feed_status()
    assert status["healthy"] is True
    assert status["last_price"] == 5530.25
    assert status["provider"] in ("yfinance", "twelvedata")
