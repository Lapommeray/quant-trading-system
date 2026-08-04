"""
Autonomy Organism - canonical self-wiring organism (no QuantConnect).

This file is the primary organism implementation referenced by the handoff.
It re-uses core/event_bus and core/base_module but lives in autonomy/.

Features:
- Auto-discovery of modules via @register_module and package scan
- Self-improvement loop adjusting weights
- Health monitoring and isolation
- Consensus via autonomy/consensus
- EventBus wiring

Fails closed: if no modules discovered, consensus returns None (no trade).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

log = logging.getLogger(__name__)

# Local imports - no QC
from core.event_bus import EventBus, get_event_bus
from core.base_module import BaseTradingModule, get_registered_modules
from .consensus import ConsensusEngine, ConsensusResult

@dataclass
class OrganismConfig:
    auto_discover: bool = True
    self_improvement_interval_sec: int = 300
    health_check_interval_sec: int = 30
    enable_self_improvement: bool = True
    isolated_on_failure: bool = True
    max_module_failures: int = 5
    log_dir: str = "data/organism_logs"
    min_confidence: float = 0.60
    min_weighted_confidence: float = 0.30


class ModuleAutoDiscovery:
    """Discovers BaseTradingModule subclasses."""

    @staticmethod
    def discover_decorated() -> Dict[str, Type[BaseTradingModule]]:
        return get_registered_modules()

    @staticmethod
    def discover_in_package(package_name: str) -> Dict[str, Type[BaseTradingModule]]:
        discovered: Dict[str, Type[BaseTradingModule]] = {}
        try:
            pkg = importlib.import_module(package_name)
        except ImportError:
            return discovered

        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:
            return discovered

        prefix = pkg.__name__ + "."
        for _, mod_name, is_pkg in pkgutil.iter_modules(pkg_path, prefix):
            if is_pkg:
                continue
            try:
                mod = importlib.import_module(mod_name)
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if (
                        issubclass(obj, BaseTradingModule)
                        and obj is not BaseTradingModule
                        and obj.__module__ == mod.__name__
                    ):
                        key = getattr(obj, "module_name", obj.__name__)
                        discovered[key] = obj
            except Exception as exc:
                log.debug("Failed to import %s: %s", mod_name, exc)
                continue
        return discovered


class Organism:
    """Canonical autonomous organism."""

    def __init__(self, config: Optional[OrganismConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or OrganismConfig()
        self.event_bus = event_bus or get_event_bus()
        self.modules: Dict[str, BaseTradingModule] = {}
        self.module_weights: Dict[str, float] = {}
        self.performance_history: deque[Dict[str, Any]] = deque(maxlen=500)
        self._feedback_scores: Dict[str, List[float]] = defaultdict(list)
        self._running = False
        self._lock = threading.RLock()
        self._improvement_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self.consensus_engine = ConsensusEngine(min_weighted_confidence=self.config.min_weighted_confidence)

        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        self.event_bus.subscribe("ORDER_FILLED", self._on_order_filled)

    # Discovery & wiring
    def discover_and_wire(self, project_root: Optional[Path] = None) -> Dict[str, Any]:
        root = project_root or Path(__file__).resolve().parents[2] if Path(__file__).resolve().parents else Path(__file__).resolve().parent.parent

        # Decorated
        decorated = ModuleAutoDiscovery.discover_decorated()

        # Package scans - autonomy has priority over advanced_modules
        for pkg in ["autonomy", "advanced_modules", "core"]:
            pkg_mods = ModuleAutoDiscovery.discover_in_package(pkg)
            # Don't overwrite already discovered with same name from lower priority
            for k, v in pkg_mods.items():
                if k not in decorated:
                    decorated[k] = v

        total = 0
        with self._lock:
            for name, cls in decorated.items():
                if name in self.modules:
                    continue
                try:
                    instance = cls(config={}, event_bus=self.event_bus)
                    if instance.initialize():
                        self.modules[name] = instance
                        self.module_weights[name] = 1.0 / max(len(decorated), 1)
                        total += 1
                        self._wire_module_events(instance)
                    else:
                        log.warning("Module %s initialize() False", name)
                except Exception as exc:
                    log.warning("Failed to instantiate %s: %s", name, exc)

        self._normalize_weights()

        self.event_bus.publish(
            "ORGANISM_WIRED",
            {"modules": list(self.modules.keys()), "total_active": total},
            source="AutonomyOrganism",
        )

        return {"active": list(self.modules.keys()), "total_active": total}

    def _wire_module_events(self, module: BaseTradingModule):
        def make_handler(m):
            def handler(event):
                try:
                    m.on_event(event.event_type, event.payload)
                except Exception as exc:
                    m.record_failure(str(exc))
            return handler

        self.event_bus.subscribe("*", make_handler(module))

    def _normalize_weights(self):
        with self._lock:
            total = sum(self.module_weights.values())
            if total > 0:
                for k in self.module_weights:
                    self.module_weights[k] /= total

    # Lifecycle
    def start(self):
        if self._running:
            return
        self._running = True
        log.info("Autonomy Organism starting with %d modules", len(self.modules))

        if self.config.enable_self_improvement:
            self._improvement_thread = threading.Thread(
                target=self._self_improvement_loop, name="OrganismSelfImprove", daemon=True
            )
            self._improvement_thread.start()

        self._health_thread = threading.Thread(
            target=self._health_check_loop, name="OrganismHealth", daemon=True
        )
        self._health_thread.start()

        self.event_bus.publish("ORGANISM_STARTED", {"modules": len(self.modules)}, source="AutonomyOrganism")

    def stop(self):
        self._running = False
        if self._improvement_thread:
            self._improvement_thread.join(timeout=2)
        if self._health_thread:
            self._health_thread.join(timeout=2)
        self.event_bus.publish("ORGANISM_STOPPED", {}, source="AutonomyOrganism")

    # Signal generation
    def generate_consensus_signal(self, symbol: str, history_data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        confidences: Dict[str, float] = {}
        directions: Dict[str, str] = {}

        with self._lock:
            snapshot = dict(self.modules)

        start = time.time()
        for name, mod in snapshot.items():
            if not mod.enabled:
                continue
            try:
                t0 = time.time()
                res = mod.generate_signal(symbol, history_data)
                latency_ms = (time.time() - t0) * 1000
                mod.record_success(latency_ms)
                results[name] = res
                conf = getattr(res, "confidence", 0.0) if hasattr(res, "confidence") else res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
                direction = (
                    getattr(res, "signal", None)
                    if hasattr(res, "signal")
                    else res.get("direction") or res.get("signal")
                    if isinstance(res, dict)
                    else None
                )
                if direction:
                    dir_up = str(direction).upper()
                    if dir_up in ("BUY", "SELL", "NEUTRAL"):
                        directions[name] = dir_up
                        confidences[name] = float(conf)
            except Exception as exc:
                mod.record_failure(str(exc))
                log.warning("Module %s signal failed: %s", name, exc)

        with self._lock:
            weights = dict(self.module_weights)

        latency_ms = (time.time() - start) * 1000

        consensus_result: ConsensusResult = self.consensus_engine.compute(
            symbol=symbol,
            directions=directions,
            confidences=confidences,
            weights=weights,
            module_results={k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in results.items()},
            latency_ms=latency_ms,
        )

        payload = {
            "symbol": consensus_result.symbol,
            "final_signal": consensus_result.final_signal,
            "confidence": consensus_result.confidence,
            "weighted_confidence": consensus_result.weighted_confidence,
            "votes": consensus_result.votes,
            "module_results": consensus_result.module_results,
            "latency_ms": consensus_result.latency_ms,
            "timestamp": consensus_result.timestamp,
        }

        self.event_bus.publish("SIGNAL_GENERATED", payload, source="AutonomyOrganism")
        return payload

    # Self-improvement
    def _self_improvement_loop(self):
        while self._running:
            try:
                time.sleep(self.config.self_improvement_interval_sec)
                if not self._running:
                    break
                self.run_self_improvement_cycle()
            except Exception as exc:
                log.exception("Self-improve loop error: %s", exc)

    def run_self_improvement_cycle(self) -> Dict[str, Any]:
        with self._lock:
            mods = dict(self.modules)

        avg_scores: Dict[str, float] = {}
        for mod_name, scores in self._feedback_scores.items():
            if scores:
                avg_scores[mod_name] = sum(scores[-20:]) / len(scores[-20:])

        with self._lock:
            for name in self.module_weights:
                score = avg_scores.get(name, 0.5)
                if score > 0.6:
                    self.module_weights[name] *= 1.05
                elif score < 0.4:
                    self.module_weights[name] *= 0.95
            self._normalize_weights()

        improvements: Dict[str, Any] = {}
        for name, mod in mods.items():
            try:
                hist = list(self.performance_history)[-50:]
                res = mod.self_improve(hist)
                improvements[name] = res
            except Exception as exc:
                improvements[name] = {"error": str(exc)}

        result = {
            "timestamp": time.time(),
            "weights": dict(self.module_weights),
            "avg_scores": avg_scores,
            "improvements": improvements,
        }

        self.performance_history.append(result)
        self.event_bus.publish("SELF_IMPROVEMENT", result, source="AutonomyOrganism")

        try:
            log_path = Path(self.config.log_dir) / f"self_improve_{int(time.time())}.json"
            import json

            with open(log_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            pass

        return result

    def feedback(self, module_name: str, reward: float):
        self._feedback_scores[module_name].append(max(0.0, min(1.0, reward)))
        if len(self._feedback_scores[module_name]) > 100:
            self._feedback_scores[module_name] = self._feedback_scores[module_name][-100:]

    # Health
    def _health_check_loop(self):
        while self._running:
            try:
                time.sleep(self.config.health_check_interval_sec)
                if not self._running:
                    break
                self._run_health_check()
            except Exception as exc:
                log.exception("Health check error: %s", exc)

    def _run_health_check(self):
        with self._lock:
            mods = dict(self.modules)
        for name, mod in mods.items():
            if (
                mod.health.error_count >= self.config.max_module_failures
                and mod.enabled
                and self.config.isolated_on_failure
            ):
                mod.enabled = False
                mod.health.status = "isolated"
                log.warning("Module %s isolated after %d failures", name, mod.health.error_count)
                self.event_bus.publish(
                    "MODULE_HEALTH",
                    {"module": name, "status": "isolated", "reason": mod.health.last_error},
                    source="AutonomyOrganism",
                )

    def _on_order_filled(self, event):
        payload = event.payload if hasattr(event, "payload") else event
        if isinstance(payload, dict):
            mod_name = payload.get("source_module")
            pnl = payload.get("pnl")
            if mod_name and pnl is not None:
                reward = 1.0 if pnl > 0 else 0.0
                self.feedback(mod_name, reward)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "modules": {name: mod.to_dict() for name, mod in self.modules.items()},
                "weights": dict(self.module_weights),
                "performance_history_len": len(self.performance_history),
                "feedback_scores": {
                    k: sum(v[-10:]) / len(v[-10:]) if v else 0.0 for k, v in self._feedback_scores.items()
                },
            }


# Singleton
_global_organism: Optional[Organism] = None
_global_lock = threading.Lock()


def get_organism() -> Organism:
    global _global_organism
    with _global_lock:
        if _global_organism is None:
            _global_organism = Organism()
            _global_organism.discover_and_wire()
        return _global_organism


def reset_organism():
    global _global_organism
    with _global_lock:
        if _global_organism:
            _global_organism.stop()
        _global_organism = None
