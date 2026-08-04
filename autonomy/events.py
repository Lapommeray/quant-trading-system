"""
Autonomy events - canonical event definitions for organism communication.

This module defines event types and payload schemas used by autonomy/organism.py,
autonomy/consensus.py, autonomy/execution.py and okx_live/.

No QuantConnect dependency. Fail-closed: unknown events are dropped.

Event types (canonical):
- SIGNAL_GENERATED
- CONSENSUS_REACHED
- ORDER_REQUEST
- ORDER_FILLED
- RISK_ALERT
- MODULE_HEALTH
- SELF_IMPROVEMENT
- KILL_SWITCH
- ORGANISM_WIRED
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AutonomyEvent:
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


# Canonical event type constants
class EventTypes:
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    CONSENSUS_REACHED = "CONSENSUS_REACHED"
    ORDER_REQUEST = "ORDER_REQUEST"
    ORDER_FILLED = "ORDER_FILLED"
    RISK_ALERT = "RISK_ALERT"
    MODULE_HEALTH = "MODULE_HEALTH"
    SELF_IMPROVEMENT = "SELF_IMPROVEMENT"
    KILL_SWITCH = "KILL_SWITCH"
    ORGANISM_WIRED = "ORGANISM_WIRED"
    ORGANISM_STARTED = "ORGANISM_STARTED"
    ORGANISM_STOPPED = "ORGANISM_STOPPED"
    EXECUTOR_STARTED = "EXECUTOR_STARTED"
    OKX_CONNECTED = "OKX_CONNECTED"


def make_signal_event(symbol: str, final_signal: Optional[str], confidence: float, weighted_confidence: float, votes: Dict[str, float], source: str = "AutonomyOrganism") -> AutonomyEvent:
    return AutonomyEvent(
        event_type=EventTypes.SIGNAL_GENERATED,
        source=source,
        payload={
            "symbol": symbol,
            "final_signal": final_signal,
            "confidence": confidence,
            "weighted_confidence": weighted_confidence,
            "votes": votes,
            "timestamp": time.time(),
        },
    )


def make_order_request_event(symbol: str, side: str, quantity: float, order_type: str = "market", real_trading: bool = True) -> AutonomyEvent:
    return AutonomyEvent(
        event_type=EventTypes.ORDER_REQUEST,
        source="AutonomyExecutor",
        payload={
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "real_trading": real_trading,
            "timestamp": time.time(),
        },
    )


def make_risk_alert_event(reason: str, order_id: Optional[str] = None) -> AutonomyEvent:
    return AutonomyEvent(
        event_type=EventTypes.RISK_ALERT,
        source="SafetyGuard",
        payload={"reason": reason, "order_id": order_id, "timestamp": time.time()},
    )
