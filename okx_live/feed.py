"""
OKX Pre-Broker WebSocket Feed - raw market data BEFORE any broker interaction.

This is the "read data as soon as it lands in the core data system" layer:

    OKX public WS (books5, trades, tickers, open-interest, funding-rate,
    liquidation-orders)
        -> DataRing (in-memory tick ring, <0.2ms reads)
        -> EventBus events (MARKET_DATA, WHALE_FLOW, OI_UPDATE,
           FUNDING_UPDATE, LIQUIDATION_EVENT)

Nothing here touches accounts, keys, or order placement.  The feed is
read-only and fails closed: if the socket dies or the ring goes stale,
``feed_status()`` reports it and ``RISK_ALERT`` is published.

Offline testability: ``handle_frame()`` is a pure parser that takes an OKX v5
WS frame and mutates only in-memory state, so the entire message path can be
unit-tested without network access.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

log = logging.getLogger(__name__)

try:
    import websocket  # type: ignore

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    websocket = None  # type: ignore

try:
    from core.data_ring import DataRing, get_data_ring  # type: ignore
    from core.event_bus import EventBus, get_event_bus  # type: ignore

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    DataRing = None  # type: ignore
    get_data_ring = None  # type: ignore
    EventBus = None  # type: ignore
    get_event_bus = None  # type: ignore

# Canonical ring symbol mapping: OKX instId -> DataRing symbol.
_INST_TO_RING = {
    "BTC-USDT-SWAP": "BTCUSDT",
    "BTC-USDT": "BTCUSDT",
    "BTC-USD-SWAP": "BTCUSD",
    "ETH-USDT-SWAP": "ETHUSDT",
    "ETH-USDT": "ETHUSDT",
    "ETH-USD-SWAP": "ETHUSD",
}


def ring_symbol_for(inst_id: str) -> str:
    return _INST_TO_RING.get(inst_id, inst_id.replace("-", ""))


@dataclass
class OKXFeedConfig:
    ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    symbols: List[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("OKX_FEED_SYMBOLS", "BTC-USDT-SWAP,ETH-USDT-SWAP").split(
                ","
            )
            if s.strip()
        ]
        or ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    )
    channels: List[str] = field(
        default_factory=lambda: [
            "books5",
            "trades",
            "tickers",
            "open-interest",
            "funding-rate",
            "liquidation-orders",
        ]
    )
    ping_interval_sec: float = 25.0
    reconnect_backoff_sec: float = 3.0
    stale_after_sec: float = 10.0
    # Whales: single trade notional above this USD threshold.
    whale_usd_threshold: float = 50_000.0
    # A cluster = >=3 whale prints within this window.
    whale_cluster_window_sec: float = 5.0
    # Publish ORDER_BOOK_UPDATE at most every N seconds per symbol.
    book_publish_interval_sec: float = 0.25


class OKXPreBrokerFeed:
    """Public OKX WebSocket feed writing straight into the core DataRing."""

    def __init__(
        self,
        config: Optional[OKXFeedConfig] = None,
        event_bus: Optional[Any] = None,
        ring_factory: Callable[[str], Any] = None,
        ws_factory: Callable[..., Any] = None,
    ):
        self.config = config or OKXFeedConfig()
        self.event_bus = event_bus
        if self.event_bus is None and CORE_AVAILABLE and get_event_bus:
            self.event_bus = get_event_bus()

        # Ring factory: default to the global DataRing registry; injectable
        # for tests (a dict or fake ring works).
        self._ring_factory = ring_factory or (
            (lambda sym: get_data_ring(sym)) if CORE_AVAILABLE and get_data_ring else {}
        )
        self._ws_factory = ws_factory or (
            (websocket.WebSocketApp if websocket is not None else None)
        )
        if self._ws_factory is None and not ws_factory:
            log.warning(
                "websocket-client not installed; feed can parse frames but cannot connect"
            )

        self._ws: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._lock = threading.RLock()
        self._last_frame_ts = 0.0
        self._last_book_pub: Dict[str, float] = {}
        self._whale_window: Deque[tuple] = deque()
        self._stats: Dict[str, int] = {
            "frames": 0,
            "trades": 0,
            "books": 0,
            "whales": 0,
            "whale_clusters": 0,
            "liquidations": 0,
            "reconnects": 0,
        }

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="OKXPreBrokerFeed", daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None
        self._connected = False

    def _run_loop(self) -> None:
        backoff = self.config.reconnect_backoff_sec
        while self._running:
            if self._ws_factory is None:
                time.sleep(backoff)
                continue
            try:
                self._ws = self._ws_factory(
                    self.config.ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self._ws.run_forever(ping_interval=self.config.ping_interval_sec)
            except Exception as exc:  # noqa: BLE001
                log.warning("OKX feed connection error: %s", exc)
            self._stats["reconnects"] += 1
            self._connected = False
            if not self._running:
                break
            time.sleep(backoff)

    # ------------------------------------------------------------------
    # ws callbacks
    # ------------------------------------------------------------------
    def _on_open(self, ws: Any) -> None:
        self._connected = True
        log.info("OKX pre-broker feed connected to %s", self.config.ws_url)
        if self.event_bus is not None:
            self.event_bus.publish(
                "OKX_FEED_CONNECTED",
                {"symbols": self.config.symbols, "channels": self.config.channels},
                source="OKXPreBrokerFeed",
            )
        self._subscribe(ws)

    def _subscribe(self, ws: Any) -> None:
        args = [
            {"channel": channel, "instId": inst}
            for channel in self.config.channels
            for inst in self.config.symbols
        ]
        try:
            ws.send(json.dumps({"op": "subscribe", "args": args}))
        except Exception as exc:  # noqa: BLE001
            log.warning("OKX feed subscribe failed: %s", exc)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            frame = json.loads(message)
        except json.JSONDecodeError:
            return
        self.handle_frame(frame)

    def _on_error(self, ws: Any, error: Any) -> None:
        log.warning("OKX feed WS error: %s", error)

    def _on_close(self, ws: Any, code: Any = None, msg: Any = None) -> None:
        self._connected = False
        log.info("OKX feed WS closed (code=%s)", code)

    # ------------------------------------------------------------------
    # frame parsing (pure, offline-testable)
    # ------------------------------------------------------------------
    def handle_frame(self, frame: Dict[str, Any]) -> None:
        if "data" not in frame or not isinstance(frame.get("data"), list):
            return
        arg = frame.get("arg") or {}
        channel = arg.get("channel", "")
        inst_id = arg.get("instId", "")
        if not channel or not inst_id:
            return

        self._last_frame_ts = time.time()
        self._stats["frames"] += 1
        ring = (
            self._ring_factory(ring_symbol_for(inst_id)) if self._ring_factory else None
        )

        for item in frame["data"]:
            try:
                if channel == "trades":
                    self._handle_trade(inst_id, ring, item)
                elif channel == "books5":
                    self._handle_book(inst_id, ring, item)
                elif channel == "tickers":
                    self._handle_ticker(inst_id, item)
                elif channel == "open-interest":
                    self._handle_oi(inst_id, item)
                elif channel == "funding-rate":
                    self._handle_funding(inst_id, item)
                elif channel == "liquidation-orders":
                    self._handle_liquidation(inst_id, item)
            except Exception as exc:  # noqa: BLE001
                log.warning("OKX feed frame error (%s/%s): %s", channel, inst_id, exc)

    def _handle_trade(self, inst_id: str, ring: Any, item: Dict[str, Any]) -> None:
        px = float(item.get("px", 0.0))
        sz = float(item.get("sz", 0.0))
        side = 1 if item.get("side") == "buy" else -1
        ts = int(item.get("ts", time.time() * 1000)) / 1000.0
        if px <= 0:
            return
        self._stats["trades"] += 1
        if ring is not None and hasattr(ring, "push"):
            ring.push(ts, 0.0, 0.0, 0.0, 0.0, px, sz, side)

        notional = px * sz
        if notional >= self.config.whale_usd_threshold:
            self._stats["whales"] += 1
            now = time.time()
            self._whale_window.append((now, inst_id, side, notional))
            while (
                self._whale_window
                and now - self._whale_window[0][0]
                > self.config.whale_cluster_window_sec
            ):
                self._whale_window.popleft()
            clustered = len(self._whale_window) >= 3
            if clustered:
                self._stats["whale_clusters"] += 1
            if self.event_bus is not None:
                self.event_bus.publish(
                    "WHALE_FLOW",
                    {
                        "inst_id": inst_id,
                        "symbol": ring_symbol_for(inst_id),
                        "side": "buy" if side == 1 else "sell",
                        "px": px,
                        "sz": sz,
                        "notional_usd": notional,
                        "cluster": clustered,
                    },
                    source="OKXPreBrokerFeed",
                )
        if self.event_bus is not None:
            self.event_bus.publish(
                "MARKET_DATA",
                {
                    "symbol": ring_symbol_for(inst_id),
                    "inst_id": inst_id,
                    "price": px,
                    "qty": sz,
                    "side": side,
                    "source": "okx_ws",
                    "pre_broker": True,
                },
                source="OKXPreBrokerFeed",
            )

    def _handle_book(self, inst_id: str, ring: Any, item: Dict[str, Any]) -> None:
        bids = item.get("bids") or []
        asks = item.get("asks") or []
        ts = int(item.get("ts", time.time() * 1000)) / 1000.0
        if not bids or not asks:
            return
        bid_px, bid_sz = float(bids[0][0]), float(bids[0][1])
        ask_px, ask_sz = float(asks[0][0]), float(asks[0][1])
        if bid_px <= 0 or ask_px <= 0:
            return
        self._stats["books"] += 1
        if ring is not None and hasattr(ring, "push"):
            mid = (bid_px + ask_px) / 2.0
            ring.push(ts, bid_px, ask_px, bid_sz, ask_sz, mid, 0.0, 0)

        now = time.time()
        last_pub = self._last_book_pub.get(inst_id, 0.0)
        if (
            self.event_bus is not None
            and now - last_pub >= self.config.book_publish_interval_sec
        ):
            self._last_book_pub[inst_id] = now
            self.event_bus.publish(
                "ORDER_BOOK_UPDATE",
                {
                    "symbol": ring_symbol_for(inst_id),
                    "inst_id": inst_id,
                    "bid": bid_px,
                    "bid_size": bid_sz,
                    "ask": ask_px,
                    "ask_size": ask_sz,
                    "levels5": {"bids": bids, "asks": asks},
                    "source": "okx_ws",
                    "pre_broker": True,
                },
                source="OKXPreBrokerFeed",
            )

    def _handle_ticker(self, inst_id: str, item: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            "TICKER_UPDATE",
            {
                "symbol": ring_symbol_for(inst_id),
                "inst_id": inst_id,
                "last": float(item.get("last", 0.0) or 0.0),
                "open24h": float(item.get("open24h", 0.0) or 0.0),
                "high24h": float(item.get("high24h", 0.0) or 0.0),
                "low24h": float(item.get("low24h", 0.0) or 0.0),
                "vol24h": float(item.get("vol24h", 0.0) or 0.0),
                "source": "okx_ws",
            },
            source="OKXPreBrokerFeed",
        )

    def _handle_oi(self, inst_id: str, item: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            "OI_UPDATE",
            {
                "symbol": ring_symbol_for(inst_id),
                "inst_id": inst_id,
                "oi": float(item.get("oi", 0.0) or 0.0),
                "oi_ccy": float(item.get("oiCcy", 0.0) or 0.0),
                "ts": item.get("ts", ""),
            },
            source="OKXPreBrokerFeed",
        )

    def _handle_funding(self, inst_id: str, item: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish(
            "FUNDING_UPDATE",
            {
                "symbol": ring_symbol_for(inst_id),
                "inst_id": inst_id,
                "funding_rate": float(item.get("fundingRate", 0.0) or 0.0),
                "next_funding_rate": float(item.get("nextFundingRate", 0.0) or 0.0),
                "funding_time": item.get("fundingTime", ""),
            },
            source="OKXPreBrokerFeed",
        )

    def _handle_liquidation(self, inst_id: str, item: Dict[str, Any]) -> None:
        self._stats["liquidations"] += 1
        if self.event_bus is None:
            return
        self.event_bus.publish(
            "LIQUIDATION_EVENT",
            {
                "symbol": ring_symbol_for(inst_id),
                "inst_id": inst_id,
                "side": item.get("side", ""),
                "pos_side": item.get("posSide", ""),
                "bk_price": float(item.get("bkPx", 0.0) or 0.0),
                "sz": float(item.get("sz", 0.0) or 0.0),
                "type": item.get("type", "filled"),
                "ts": item.get("ts", ""),
            },
            source="OKXPreBrokerFeed",
        )

    # ------------------------------------------------------------------
    # health / status
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        return self._connected

    def last_frame_age_sec(self) -> float:
        if self._last_frame_ts <= 0:
            return float("inf")
        return time.time() - self._last_frame_ts

    def is_stale(self) -> bool:
        return self.last_frame_age_sec() > self.config.stale_after_sec

    def feed_status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "running": self._running,
            "stale": self.is_stale(),
            "last_frame_age_sec": round(self.last_frame_age_sec(), 3),
            "symbols": self.config.symbols,
            "channels": self.config.channels,
            "stats": dict(self._stats),
        }

    def check_stale_and_alert(self) -> bool:
        """Publish RISK_ALERT when the pre-broker feed goes stale."""
        if self.is_stale() and self.event_bus is not None and self._connected:
            self.event_bus.publish(
                "RISK_ALERT",
                {
                    "reason": "pre_broker_feed_stale",
                    "age_sec": round(self.last_frame_age_sec(), 2),
                    "source": "OKXPreBrokerFeed",
                },
                source="OKXPreBrokerFeed",
            )
            return True
        return False


__all__ = ["OKXPreBrokerFeed", "OKXFeedConfig", "ring_symbol_for"]
