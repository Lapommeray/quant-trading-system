"""Tests for the OKX pre-broker WebSocket feed (offline frame parsing)."""

import pytest

from core.data_ring import get_data_ring
from okx_live.feed import OKXFeedConfig, OKXPreBrokerFeed, ring_symbol_for


class _RingRegistry:
    def __init__(self):
        self.rings = {}

    def __call__(self, symbol):
        if symbol not in self.rings:
            self.rings[symbol] = get_data_ring(symbol, size=2000)
        return self.rings[symbol]


class _CaptureBus:
    def __init__(self):
        self.events = []

    def publish(self, event_type, payload, source="test", **kwargs):
        self.events.append(
            {"event_type": event_type, "payload": payload, "source": source}
        )
        return None


def _feed(registry, bus=None):
    return OKXPreBrokerFeed(
        OKXFeedConfig(
            symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            channels=[
                "trades",
                "books5",
                "tickers",
                "open-interest",
                "funding-rate",
                "liquidation-orders",
            ],
            whale_usd_threshold=50_000.0,
            whale_cluster_window_sec=5.0,
        ),
        event_bus=bus or _CaptureBus(),
        ring_factory=registry,
    )


def test_ring_symbol_mapping():
    assert ring_symbol_for("BTC-USDT-SWAP") == "BTCUSDT"
    assert ring_symbol_for("ETH-USDT-SWAP") == "ETHUSDT"
    assert ring_symbol_for("BTC-USDT") == "BTCUSDT"
    assert ring_symbol_for("BTC-USD-SWAP") == "BTCUSD"


def test_trade_frame_writes_ring_and_publishes_market_data():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = _feed(registry, bus)

    feed.handle_frame(
        {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "tradeId": "1",
                    "px": "109210.0",
                    "sz": "0.5",
                    "side": "buy",
                    "ts": "1722840000100",
                }
            ],
        }
    )

    ring = registry.rings["BTCUSDT"]
    last = ring.latest(1)[-1]
    assert float(last["price"]) == pytest.approx(109210.0)
    assert int(last["side"]) == 1
    md = [e for e in bus.events if e["event_type"] == "MARKET_DATA"]
    assert md and md[0]["payload"]["pre_broker"] is True
    assert md[0]["payload"]["symbol"] == "BTCUSDT"


def test_sell_trade_side_is_negative():
    registry = _RingRegistry()
    feed = _feed(registry)
    feed.handle_frame(
        {
            "arg": {"channel": "trades", "instId": "ETH-USDT-SWAP"},
            "data": [
                {
                    "instId": "ETH-USDT-SWAP",
                    "tradeId": "2",
                    "px": "3500.0",
                    "sz": "10.0",
                    "side": "sell",
                    "ts": "1722840000200",
                }
            ],
        }
    )
    last = registry.rings["ETHUSDT"].latest(1)[-1]
    assert int(last["side"]) == -1


def test_whale_flow_and_cluster_detection():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = _feed(registry, bus)

    # 3 buys of 0.5 BTC at ~109k = $54.5k each -> whales + cluster
    for i, px in enumerate(["109210.0", "109212.0", "109215.0"]):
        feed.handle_frame(
            {
                "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "tradeId": str(i),
                        "px": px,
                        "sz": "0.5",
                        "side": "buy",
                        "ts": f"1722840000{100 + i}00",
                    }
                ],
            }
        )

    whales = [e for e in bus.events if e["event_type"] == "WHALE_FLOW"]
    assert len(whales) == 3
    assert whales[0]["payload"]["notional_usd"] == pytest.approx(109210.0 * 0.5)
    assert whales[-1]["payload"]["cluster"] is True


def test_book_frame_populates_best_bid_ask():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = _feed(registry, bus)
    feed.handle_frame(
        {
            "arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "asks": [["109234.1", "1.02"], ["109235.0", "2.5"]],
                    "bids": [["109200.0", "4.1"], ["109199.0", "0.8"]],
                    "ts": "1722840000000",
                }
            ],
        }
    )
    last = registry.rings["BTCUSDT"].latest(1)[-1]
    assert float(last["bid"]) == pytest.approx(109200.0)
    assert float(last["ask"]) == pytest.approx(109234.1)
    assert float(last["bid_size"]) == pytest.approx(4.1)

    books = [e for e in bus.events if e["event_type"] == "ORDER_BOOK_UPDATE"]
    assert books and books[0]["payload"]["levels5"]["bids"][0][0] == "109200.0"


def test_liquidation_and_funding_frames():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = _feed(registry, bus)

    feed.handle_frame(
        {
            "arg": {"channel": "liquidation-orders", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "side": "sell",
                    "posSide": "long",
                    "bkPx": "108900.0",
                    "sz": "12.5",
                    "type": "filled",
                    "ts": "1722840000400",
                }
            ],
        }
    )
    liq = [e for e in bus.events if e["event_type"] == "LIQUIDATION_EVENT"]
    assert liq and liq[0]["payload"]["symbol"] == "BTCUSDT"
    assert liq[0]["payload"]["sz"] == pytest.approx(12.5)

    feed.handle_frame(
        {
            "arg": {"channel": "funding-rate", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "fundingRate": "0.0001",
                    "nextFundingRate": "0.00015",
                    "fundingTime": "1722844800000",
                    "ts": "1722840000500",
                }
            ],
        }
    )
    fund = [e for e in bus.events if e["event_type"] == "FUNDING_UPDATE"]
    assert fund and fund[0]["payload"]["funding_rate"] == pytest.approx(0.0001)


def test_oi_and_ticker_frames():
    registry = _RingRegistry()
    bus = _CaptureBus()
    feed = _feed(registry, bus)

    feed.handle_frame(
        {
            "arg": {"channel": "open-interest", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "oi": "185234",
                    "oiCcy": "1.69",
                    "ts": "1722840000600",
                }
            ],
        }
    )
    oi = [e for e in bus.events if e["event_type"] == "OI_UPDATE"]
    assert oi and oi[0]["payload"]["oi"] == pytest.approx(185234.0)

    feed.handle_frame(
        {
            "arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "last": "109210.0",
                    "open24h": "105000.0",
                    "high24h": "110000.0",
                    "low24h": "104500.0",
                    "vol24h": "45231.0",
                    "ts": "1722840000700",
                }
            ],
        }
    )
    tk = [e for e in bus.events if e["event_type"] == "TICKER_UPDATE"]
    assert tk and tk[0]["payload"]["last"] == pytest.approx(109210.0)


def test_feed_status_and_stale_detection():
    registry = _RingRegistry()
    feed = _feed(registry)
    status = feed.feed_status()
    assert status["running"] is False
    assert status["stats"]["frames"] == 0
    # No frames ever -> stale
    assert feed.is_stale() is True
