"""Priority-aware event bus for autonomous module communication.

Execution and safety events are lane 0.  Evolutionary work (learning,
self-coding and persistence) is lane 3.  Normal ``publish`` calls retain the
historical synchronous behavior for compatibility, while asynchronous
subscriptions and ``publish_async`` use a priority queue.  A slow learning
callback therefore cannot hold up an execution callback in the supported
priority path.

No event bus can interrupt Python code that is already executing.  The bus
therefore guarantees ordering at queue boundaries and always dispatches lane 0
synchronously before lower-priority work; trade execution still has its own
fail-closed guardrails.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Lower number means a more urgent event lane."""

    CRITICAL = 0
    OPERATIONAL = 1
    ADAPTIVE = 2
    EVOLUTIONARY = 3


_EVENT_PRIORITY: Dict[str, EventPriority] = {
    # Lane 0: execution and safety must win over everything else.
    "KILL_SWITCH": EventPriority.CRITICAL,
    "EMERGENCY_STOP": EventPriority.CRITICAL,
    "STOP_LOSS": EventPriority.CRITICAL,
    "RISK_ALERT": EventPriority.CRITICAL,
    "ORDER_REQUEST": EventPriority.CRITICAL,
    "ORDER_FILLED": EventPriority.CRITICAL,
    "SURVIVAL_MODE": EventPriority.CRITICAL,
    "CLEAR_SURVIVAL_MODE": EventPriority.CRITICAL,
    # Lane 1: data and signal flow.
    "MARKET_DATA": EventPriority.OPERATIONAL,
    "SIGNAL_GENERATED": EventPriority.OPERATIONAL,
    "CONSENSUS_REACHED": EventPriority.OPERATIONAL,
    "MODULE_REGISTERED": EventPriority.OPERATIONAL,
    "ORGANISM_WIRED": EventPriority.OPERATIONAL,
    "ORGANISM_STARTED": EventPriority.OPERATIONAL,
    "ORGANISM_STOPPED": EventPriority.OPERATIONAL,
    "EXECUTOR_STARTED": EventPriority.OPERATIONAL,
    "OKX_CONNECTED": EventPriority.OPERATIONAL,
    # Lane 2: adaptation and health.
    "MARKET_REGIME": EventPriority.ADAPTIVE,
    "MODULE_HEALTH": EventPriority.ADAPTIVE,
    "WEIGHT_UPDATE": EventPriority.ADAPTIVE,
    "LEARNING_FEEDBACK": EventPriority.ADAPTIVE,
    "SHADOW_PROMOTED": EventPriority.ADAPTIVE,
    # Lane 3: evolutionary work.
    "TRADE_OUTCOME": EventPriority.EVOLUTIONARY,
    "MEMORY_UPDATE": EventPriority.EVOLUTIONARY,
    "SELF_IMPROVEMENT": EventPriority.EVOLUTIONARY,
    "CODE_PROPOSED": EventPriority.EVOLUTIONARY,
    "CODE_VALIDATED": EventPriority.EVOLUTIONARY,
    "CODE_APPROVED": EventPriority.EVOLUTIONARY,
    "CODE_PENDING_APPROVAL": EventPriority.EVOLUTIONARY,
    "CODE_APPLIED": EventPriority.EVOLUTIONARY,
    "CODE_PENALTY_BOX": EventPriority.EVOLUTIONARY,
    "MODULE_REPAIRED": EventPriority.EVOLUTIONARY,
    "MODULE_REPAIR_FAILED": EventPriority.EVOLUTIONARY,
}


def event_priority(event_type: str) -> EventPriority:
    """Return the configured lane for an event type."""

    return _EVENT_PRIORITY.get(str(event_type), EventPriority.OPERATIONAL)


@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    priority: int = -1

    def __post_init__(self) -> None:
        if self.priority < 0:
            self.priority = int(event_priority(self.event_type))

    @property
    def lane(self) -> int:
        return int(self.priority)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "lane": self.lane,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class _Subscription:
    callback: Callable[[Event], None]
    asynchronous: bool = False


@dataclass
class _QueuedCallback:
    priority: int
    sequence: int
    callback: Callable[[Event], None]
    event: Event

    def __lt__(self, other: "_QueuedCallback") -> bool:
        return (self.priority, self.sequence) < (other.priority, other.sequence)


class EventBus:
    """Thread-safe pub/sub bus with critical-to-evolutionary priority lanes."""

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[_Subscription]] = defaultdict(list)
        self._wildcard_subscribers: List[_Subscription] = []
        self._history: deque[Event] = deque(maxlen=max_history)
        self._queue: queue.PriorityQueue[_QueuedCallback] = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._sequence = 0
        self._stats: Dict[str, int] = defaultdict(int)
        self._lane_stats: Dict[int, int] = defaultdict(int)

    # ---- subscription -------------------------------------------------
    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        *,
        asynchronous: bool = False,
        lane: Optional[int | EventPriority] = None,
    ) -> None:
        """Subscribe to an event; use ``*`` for all event types.

        ``lane`` is accepted as an explicit priority-lane spelling.  Lane 2/3
        subscriptions default to asynchronous dispatch unless the caller
        explicitly chooses otherwise.


        ``asynchronous=True`` places that callback on the priority worker.
        This is intended for persistence, analytics and evolutionary work,
        never for an order or kill-switch callback.
        """

        if lane is not None and int(lane) >= int(EventPriority.ADAPTIVE):
            asynchronous = True
        subscription = _Subscription(callback, asynchronous)
        with self._lock:
            subscribers = (
                self._wildcard_subscribers
                if event_type == "*"
                else self._subscribers[event_type]
            )
            if not any(item.callback == callback for item in subscribers):
                subscribers.append(subscription)
            log.debug("Subscribed to lane-aware event %s: %s", event_type, callback)
        if asynchronous:
            self._ensure_worker()

    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            subscribers = (
                self._wildcard_subscribers
                if event_type == "*"
                else self._subscribers[event_type]
            )
            subscribers[:] = [item for item in subscribers if item.callback != callback]

    # ---- publishing ---------------------------------------------------
    def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "unknown",
        *,
        priority: Optional[int | EventPriority] = None,
        asynchronous: bool = False,
    ) -> Event:
        event = Event(
            event_type=event_type,
            payload=payload,
            source=source,
            priority=int(event_priority(event_type) if priority is None else priority),
        )
        self._record(event)
        if asynchronous:
            self._enqueue_event(event)
        else:
            self._dispatch(event, include_async=False)
        return event

    def publish_async(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str = "unknown",
        *,
        priority: Optional[int | EventPriority] = None,
    ) -> Event:
        """Queue an event for priority-lane dispatch without blocking."""

        return self.publish(
            event_type,
            payload,
            source,
            priority=priority,
            asynchronous=True,
        )

    def publish_event(self, event: Event, *, asynchronous: bool = False) -> None:
        self._record(event)
        if asynchronous:
            self._enqueue_event(event)
        else:
            self._dispatch(event, include_async=False)

    def _record(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            self._stats[event.event_type] += 1
            self._lane_stats[int(event.priority)] += 1

    def _subscriptions_for(self, event: Event) -> List[_Subscription]:
        with self._lock:
            return list(self._subscribers.get(event.event_type, [])) + list(
                self._wildcard_subscribers
            )

    def _dispatch(self, event: Event, *, include_async: bool) -> None:
        for subscription in self._subscriptions_for(event):
            if subscription.asynchronous and not include_async:
                self._enqueue_callback(subscription.callback, event)
                continue
            try:
                subscription.callback(event)
            except Exception as exc:
                # A module callback can never crash the bus or prevent the
                # next critical callback from running.
                log.exception(
                    "EventBus callback error %s for %s: %s",
                    subscription.callback,
                    event.event_type,
                    exc,
                )

    def _enqueue_event(self, event: Event) -> None:
        self._ensure_worker()
        for subscription in self._subscriptions_for(event):
            self._enqueue_callback(subscription.callback, event)

    def _enqueue_callback(
        self, callback: Callable[[Event], None], event: Event
    ) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._queue.put(_QueuedCallback(int(event.priority), sequence, callback, event))

    # ---- async worker -------------------------------------------------
    def _ensure_worker(self) -> None:
        if not self._running:
            self.start()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="EventBusPriorityWorker",
            daemon=True,
        )
        self._worker_thread.start()
        log.info("Priority EventBus started")

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        self._worker_thread = None
        log.info("Priority EventBus stopped")

    def _worker_loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                queued = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # PriorityQueue ensures lane 0 work is selected before lane 3
                # work that is waiting.  A callback already running cannot be
                # interrupted, hence critical callbacks should stay short.
                queued.callback(queued.event)
            except Exception as exc:
                log.exception("Priority EventBus worker error: %s", exc)
            finally:
                self._queue.task_done()

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait for queued evolutionary/async callbacks to finish."""

        deadline = time.monotonic() + max(0.0, timeout)
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    # ---- introspection ------------------------------------------------
    def get_history(
        self, event_type: Optional[str] = None, limit: int = 100
    ) -> List[Event]:
        # The legacy autonomy bus accepted ``get_history(10)`` as a limit.
        if isinstance(event_type, int) and limit == 100:
            limit, event_type = event_type, None
        with self._lock:
            if event_type:
                filtered = [e for e in self._history if e.event_type == event_type]
            else:
                filtered = list(self._history)
        return filtered[-max(0, int(limit)) :]

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def get_lane_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "critical": self._lane_stats[int(EventPriority.CRITICAL)],
                "operational": self._lane_stats[int(EventPriority.OPERATIONAL)],
                "adaptive": self._lane_stats[int(EventPriority.ADAPTIVE)],
                "evolutionary": self._lane_stats[int(EventPriority.EVOLUTIONARY)],
                "queued": self._queue.qsize(),
            }

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._stats.clear()
            self._lane_stats.clear()


# Singleton -------------------------------------------------------------
_global_bus: Optional[EventBus] = None
_global_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _global_bus
    with _global_lock:
        if _global_bus is None:
            _global_bus = EventBus()
            _global_bus.start()
        return _global_bus


def reset_event_bus() -> None:
    global _global_bus
    with _global_lock:
        if _global_bus:
            _global_bus.stop()
        _global_bus = None


PriorityEventBus = EventBus

__all__ = [
    "Event",
    "EventBus",
    "EventPriority",
    "PriorityEventBus",
    "event_priority",
    "get_event_bus",
    "reset_event_bus",
]
