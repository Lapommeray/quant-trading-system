"""
DataRing - Zero-copy pre-broker core data system
Single source of truth IN MEMORY before broker sees anything.
Designed for <0.2ms push/read, 50ms end-to-end signal path.
"""

import numpy as np
import time
import threading
from typing import Dict, Optional


class DataRing:
    """
    Lock-free single-writer multi-reader circular buffer for L2 + trade ticks.
    One per symbol: BTCUSDT, ETHUSDT, SPY, ES, etc.
    """

    def __init__(self, symbol: str, size: int = 200000):
        self.symbol = symbol
        self.size = size
        # dtype: ts float64, bid/ask float32, sizes float32, last price/qty float32, side int8 (1 buy, -1 sell, 0 unknown)
        self.dtype = np.dtype(
            [
                ("ts", "f8"),
                ("bid", "f4"),
                ("ask", "f4"),
                ("bid_size", "f4"),
                ("ask_size", "f4"),
                ("price", "f4"),
                ("qty", "f4"),
                ("side", "i1"),
            ]
        )
        self.buf = np.zeros(size, dtype=self.dtype)
        self.head = 0
        self._lock = (
            threading.Lock()
        )  # only for head update, readers lock-free reading old data
        self.last_update_ts = time.time()

    def push(
        self,
        ts: float,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        price: float,
        qty: float,
        side: int = 0,
    ):
        """Push tick from WS thread - must be ultra fast."""
        # No allocation, direct assignment
        idx = self.head
        self.buf[idx]["ts"] = ts
        self.buf[idx]["bid"] = bid
        self.buf[idx]["ask"] = ask
        self.buf[idx]["bid_size"] = bid_size
        self.buf[idx]["ask_size"] = ask_size
        self.buf[idx]["price"] = price
        self.buf[idx]["qty"] = qty
        self.buf[idx]["side"] = side
        # atomic head increment (single writer, so no race)
        self.head = (idx + 1) % self.size
        self.last_update_ts = ts

    def latest(self, n: int = 200):
        """Zero-copy view of last n ticks. No copy unless wrapped. Up to 0.1ms."""
        if n <= 0:
            return self.buf[:0]
        head = self.head
        if head >= n:
            return self.buf[head - n : head]  # contiguous, no copy view
        else:
            # wrapped
            part1 = self.buf[self.size - (n - head) :]
            part2 = self.buf[:head]
            # This case needs copy (wrap). Acceptable occasionally. For speed, caller should avoid requesting n > head when head small.
            return np.concatenate([part1, part2])

    def latest_bid_ask(self):
        """Get latest best bid/ask in O(1)."""
        idx = (self.head - 1) % self.size
        entry = self.buf[idx]
        return (
            float(entry["bid"]),
            float(entry["ask"]),
            float(entry["bid_size"]),
            float(entry["ask_size"]),
        )

    def latency_ms(self):
        """Time since last update - if >5 sec, feed stale."""
        return (time.time() - self.last_update_ts) * 1000


# Global registry - one ring per symbol
_global_rings: Dict[str, DataRing] = {}
_global_lock = threading.Lock()


def get_data_ring(symbol: str, size: int = 200000) -> DataRing:
    with _global_lock:
        if symbol not in _global_rings:
            _global_rings[symbol] = DataRing(symbol, size=size)
        return _global_rings[symbol]


def get_global_ring(symbol: str) -> DataRing:
    return get_data_ring(symbol)


__all__ = ["DataRing", "get_data_ring", "get_global_ring"]
