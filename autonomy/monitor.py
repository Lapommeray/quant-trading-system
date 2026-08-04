"""Lightweight autonomous health/resource monitor."""

from __future__ import annotations

import gc
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class AutonomousMonitor:
    """Track repeated errors and resource pressure without optional psutil."""

    def __init__(
        self,
        *,
        max_module_errors: int = 5,
        max_memory_mb: int = 500,
        max_cpu_percent: float = 25.0,
        error_callback: Optional[Callable[[str, str], None]] = None,
        resource_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        clock=time.time,
    ) -> None:
        self.max_module_errors = max(1, int(max_module_errors))
        self.max_memory_mb = max(1, int(max_memory_mb))
        self.max_cpu_percent = max(1.0, float(max_cpu_percent))
        self.error_callback = error_callback
        self.resource_callback = resource_callback
        self.clock = clock
        self._last_resource_callback = 0.0
        self.metrics: Dict[str, Any] = {
            "error_count": 0,
            "last_errors": deque(maxlen=50),
            "memory_samples_mb": deque(maxlen=60),
            "cpu_samples_pct": deque(maxlen=60),
            "module_errors": Counter(),
        }
        self.running = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def log_error(self, error: str, module: str) -> None:
        self.metrics["error_count"] += 1
        self.metrics["module_errors"][module] += 1
        self.metrics["last_errors"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "module": module,
                "error": str(error),
            }
        )
        if (
            self.metrics["module_errors"][module] >= self.max_module_errors
            and self.error_callback
        ):
            self.error_callback(module, str(error))

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, name="AutonomousMonitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def _monitor_loop(self) -> None:
        previous_cpu = time.process_time()
        previous_wall = time.monotonic()
        while self.running and not self._stop.wait(1.0):
            now_cpu = time.process_time()
            now_wall = time.monotonic()
            cpu = (now_cpu - previous_cpu) / max(now_wall - previous_wall, 1e-6) * 100.0
            previous_cpu, previous_wall = now_cpu, now_wall
            memory = self._memory_mb()
            self.metrics["cpu_samples_pct"].append(cpu)
            self.metrics["memory_samples_mb"].append(memory)
            if memory > self.max_memory_mb or cpu > self.max_cpu_percent:
                # Resource pressure only creates an operational signal.  It
                # never deletes learning memory or loosens financial limits.
                self.metrics["resource_pressure"] = True
                self.metrics["throttled"] = True
                gc.collect()
                now = time.monotonic()
                if self.resource_callback and now - self._last_resource_callback >= 5.0:
                    self._last_resource_callback = now
                    self.resource_callback({"cpu_percent": cpu, "memory_mb": memory})

    @staticmethod
    def _memory_mb() -> float:
        try:
            import resource

            # Linux reports KB; macOS reports bytes.  Detect the common case.
            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value / (1024.0 if value > 10_000 else 1.0)
        except (ImportError, OSError, ValueError):
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "error_count": self.metrics["error_count"],
            "module_errors": dict(self.metrics["module_errors"]),
            "resource_pressure": bool(self.metrics.get("resource_pressure", False)),
            "throttled": bool(self.metrics.get("throttled", False)),
            "last_errors": list(self.metrics["last_errors"]),
            "cpu_samples_pct": list(self.metrics["cpu_samples_pct"]),
            "memory_samples_mb": list(self.metrics["memory_samples_mb"]),
        }


__all__ = ["AutonomousMonitor"]
