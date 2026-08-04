"""Append-only JSONL audit trail for autonomous safety events."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class AuditTrail:
    """Best-effort durable audit writer that never loosens safety decisions."""

    def __init__(self, path: str | Path = "audit_logs/autonomous_events.jsonl") -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.write_failures = 0

    def record(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        source: str = "system",
    ) -> bool:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "event_type": event_type,
            "source": source,
            "payload": self._safe(payload or {}),
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except OSError:
                        pass
            return True
        except (OSError, TypeError, ValueError):
            self.write_failures += 1
            return False

    def callback(self, event: Any) -> None:
        payload = getattr(event, "payload", {})
        self.record(
            getattr(event, "event_type", "UNKNOWN"),
            payload if isinstance(payload, dict) else {"value": str(payload)},
            source=getattr(event, "source", "unknown"),
        )

    def subscribe(self, event_bus: Any) -> None:
        if hasattr(event_bus, "subscribe"):
            try:
                event_bus.subscribe("*", self.callback, asynchronous=True)
            except TypeError:
                event_bus.subscribe("*", self.callback)

    @staticmethod
    def _safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {
                str(key): AuditTrail._safe(item)
                for key, item in list(value.items())[:100]
            }
        if isinstance(value, (list, tuple)):
            return [AuditTrail._safe(item) for item in list(value)[:100]]
        if hasattr(value, "to_dict"):
            try:
                return AuditTrail._safe(value.to_dict())
            except Exception:
                pass
        return str(value)

    def status(self) -> Dict[str, Any]:
        return {"path": str(self.path), "write_failures": self.write_failures}


__all__ = ["AuditTrail"]
