"""
Event bus for autonomous module communication.

Implements a thread-safe pub/sub system used by the Organism to wire
modules together without tight coupling.

Event types (canonical):
- SIGNAL_GENERATED: qmp engine produced a signal
- MARKET_DATA: new tick/candle
- ORDER_REQUEST: request to place order
- ORDER_FILLED: exchange confirmed fill
- RISK_ALERT: risk manager alert
- MODULE_HEALTH: module health update
- SELF_IMPROVEMENT: self-improvement cycle result
- KILL_SWITCH: emergency halt
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EventBus:
    """Thread-safe event bus with sync and async consumption."""

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._wildcard_subscribers: List[Callable[[Event], None]] = []
        self._history: deque[Event] = deque(maxlen=max_history)
        self._queue: queue.Queue[Event] = queue.Queue()
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stats: Dict[str, int] = defaultdict(int)

    # ---- subscription ----
    def subscribe(self, event_type: str, callback: Callable[[Event], None]):
        """Subscribe to specific event type. Use '*' for all."""
        with self._lock:
            if event_type == "*":
                self._wildcard_subscribers.append(callback)
            else:
                self._subscribers[event_type].append(callback)
            log.debug("Subscribed to %s: %s", event_type, callback)

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]):
        with self._lock:
            if event_type == "*":
                if callback in self._wildcard_subscribers:
                    self._wildcard_subscribers.remove(callback)
            else:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)

    # ---- publishing ----
    def publish(self, event_type: str, payload: Dict[str, Any], source: str = "unknown") -> Event:
        ev = Event(event_type=event_type, payload=payload, source=source)
        self._publish_event(ev)
        return ev

    def publish_event(self, event: Event):
        self._publish_event(event)

    def _publish_event(self, event: Event):
        with self._lock:
            self._history.append(event)
            self._stats[event.event_type] += 1

        # Immediate synchronous dispatch to subscribers (fast path)
        callbacks: List[Callable] = []
        with self._lock:
            callbacks.extend(self._subscribers.get(event.event_type, []))
            callbacks.extend(self._wildcard_subscribers)

        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.exception("EventBus callback error %s for %s: %s", cb, event.event_type, exc)

        # Also enqueue for async worker if running
        if self._running:
            self._queue.put(event)

    # ---- async worker ----
    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, name="EventBus", daemon=True)
        self._worker_thread.start()
        log.info("EventBus started")

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        log.info("EventBus stopped")

    def _worker_loop(self):
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
                # Could add persistence, additional side channels
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log.exception("EventBus worker error: %s", exc)

    # ---- introspection ----
    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        with self._lock:
            if event_type:
                filtered = [e for e in self._history if e.event_type == event_type]
            else:
                filtered = list(self._history)
        return filtered[-limit:]

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def clear_history(self):
        with self._lock:
            self._history.clear()
            self._stats.clear()


# Singleton
_global_bus: Optional[EventBus] = None
_global_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _global_bus
    with _global_lock:
        if _global_bus is None:
            _global_bus = EventBus()
            _global_bus.start()
        return _global_bus


def reset_event_bus():
    global _global_bus
    with _global_lock:
        if _global_bus:
            _global_bus.stop()
        _global_bus = None
