"""
Autonomy Organism - self-wiring organism, no QuantConnect, no core dependency.

Fail-closed: if no modules, consensus returns None (no trade).
Self-contained: includes local EventBus and BaseModule registry to avoid core/ dependency.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable

log = logging.getLogger(__name__)

# ---- Minimal EventBus (self-contained) ----
class _LocalEvent:
    def __init__(self, event_type: str, payload: Dict[str, Any], source: str = "unknown"):
        self.event_type = event_type
        self.payload = payload
        self.source = source
        self.timestamp = time.time()

class _LocalEventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._wildcard: List[Callable] = []
        self._lock = threading.RLock()
        self._history = deque(maxlen=1000)

    def subscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type == "*":
                self._wildcard.append(callback)
            else:
                self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: Dict[str, Any], source: str = "unknown"):
        ev = _LocalEvent(event_type, payload, source)
        with self._lock:
            self._history.append(ev)
            cbs = list(self._subscribers.get(event_type, [])) + list(self._wildcard)
        for cb in cbs:
            try:
                cb(ev)
            except Exception as exc:
                log.debug("EventBus callback error: %s", exc)
        return ev

    def get_history(self, limit=100):
        with self._lock:
            return list(self._history)[-limit:]

_global_bus: Optional[_LocalEventBus] = None
_bus_lock = threading.Lock()

def get_event_bus() -> _LocalEventBus:
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = _LocalEventBus()
        return _global_bus

# ---- Base Module (self-contained) ----
MODULE_REGISTRY: Dict[str, Type["BaseModule"]] = {}

def register_module(cls=None, *, name: str | None = None):
    def decorator(inner_cls):
        reg_name = name or getattr(inner_cls, "module_name", inner_cls.__name__)
        MODULE_REGISTRY[reg_name] = inner_cls
        return inner_cls
    if cls is None:
        return decorator
    else:
        return decorator(cls)

@dataclass
class ModuleHealth:
    module_name: str
    status: str = "ok"
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_error: Optional[str] = None

@dataclass
class ModuleResult:
    module_name: str
    signal: Optional[str] = None
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {"module_name": self.module_name, "signal": self.signal, "confidence": self.confidence, "features": self.features}

class BaseModule:
    module_name: str = "base"
    category: str = "general"
    version: str = "1.0.0"
    dependencies: List[str] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None, event_bus: Any | None = None):
        self.config = config or {}
        self.event_bus = event_bus
        self.health = ModuleHealth(module_name=self.module_name)
        self.enabled = True

    def initialize(self) -> bool:
        return True

    def generate_signal(self, symbol: str, history_data: Dict[str, Any]) -> ModuleResult:
        return ModuleResult(module_name=self.module_name, signal="NEUTRAL", confidence=0.0)

    def on_event(self, event_type: str, payload: Dict[str, Any]):
        pass

    def self_improve(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"module": self.module_name, "improved": False}

    def record_success(self, latency_ms: float):
        self.health.success_count += 1
        self.health.status = "ok"
        if self.health.avg_latency_ms == 0:
            self.health.avg_latency_ms = latency_ms
        else:
            self.health.avg_latency_ms = 0.9 * self.health.avg_latency_ms + 0.1 * latency_ms
        self.health.last_heartbeat = time.time()

    def record_failure(self, error: str):
        self.health.error_count += 1
        self.health.last_error = error
        if self.health.error_count > 5:
            self.health.status = "failed"
        self.health.last_heartbeat = time.time()

    def to_dict(self):
        return {"module_name": self.module_name, "category": self.category, "enabled": self.enabled, "health": {"status": self.health.status, "error_count": self.health.error_count}}


# For backwards compat, alias BaseTradingModule to BaseModule
BaseTradingModule = BaseModule

def get_registered_modules():
    return dict(MODULE_REGISTRY)

# ---- Consensus ----
from .consensus import ConsensusEngine

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
    @staticmethod
    def discover_decorated() -> Dict[str, Type[BaseModule]]:
        return get_registered_modules()

    @staticmethod
    def discover_in_package(package_name: str) -> Dict[str, Type[BaseModule]]:
        discovered: Dict[str, Type[BaseModule]] = {}
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
                    if issubclass(obj, BaseModule) and obj is not BaseModule and obj.__module__ == mod.__name__:
                        key = getattr(obj, "module_name", obj.__name__)
                        discovered[key] = obj
            except Exception as exc:
                log.debug("Failed import %s: %s", mod_name, exc)
        return discovered

class Organism:
    def __init__(self, config: Optional[OrganismConfig] = None, event_bus: Optional[Any] = None):
        self.config = config or OrganismConfig()
        self.event_bus = event_bus or get_event_bus()
        self.modules: Dict[str, BaseModule] = {}
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

    def discover_and_wire(self, project_root: Optional[Path] = None) -> Dict[str, Any]:
        decorated = ModuleAutoDiscovery.discover_decorated()
        for pkg in ["autonomy", "advanced_modules"]:
            pkg_mods = ModuleAutoDiscovery.discover_in_package(pkg)
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
                except Exception as exc:
                    log.warning("Failed instantiate %s: %s", name, exc)
        self._normalize_weights()
        self.event_bus.publish("ORGANISM_WIRED", {"modules": list(self.modules.keys()), "total_active": total}, source="AutonomyOrganism")
        return {"active": list(self.modules.keys()), "total_active": total}

    def _wire_module_events(self, module: BaseModule):
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

    def start(self):
        if self._running:
            return
        self._running = True
        if self.config.enable_self_improvement:
            self._improvement_thread = threading.Thread(target=self._self_improvement_loop, name="OrganismSelfImprove", daemon=True)
            self._improvement_thread.start()
        self._health_thread = threading.Thread(target=self._health_check_loop, name="OrganismHealth", daemon=True)
        self._health_thread.start()
        self.event_bus.publish("ORGANISM_STARTED", {"modules": len(self.modules)}, source="AutonomyOrganism")

    def stop(self):
        self._running = False
        if self._improvement_thread:
            self._improvement_thread.join(timeout=2)
        if self._health_thread:
            self._health_thread.join(timeout=2)
        self.event_bus.publish("ORGANISM_STOPPED", {}, source="AutonomyOrganism")

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
                direction = getattr(res, "signal", None) if hasattr(res, "signal") else res.get("direction") or res.get("signal") if isinstance(res, dict) else None
                if direction:
                    dir_up = str(direction).upper()
                    if dir_up in ("BUY", "SELL", "NEUTRAL"):
                        directions[name] = dir_up
                        confidences[name] = float(conf)
            except Exception as exc:
                mod.record_failure(str(exc))
        with self._lock:
            weights = dict(self.module_weights)
        latency_ms = (time.time() - start) * 1000
        consensus_result = self.consensus_engine.compute(symbol=symbol, directions=directions, confidences=confidences, weights=weights, module_results={k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in results.items()}, latency_ms=latency_ms)
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

    def _self_improvement_loop(self):
        while self._running:
            try:
                time.sleep(self.config.self_improvement_interval_sec)
                if not self._running:
                    break
                self.run_self_improvement_cycle()
            except Exception as exc:
                log.exception("Self-improve error: %s", exc)

    def run_self_improvement_cycle(self):
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
        improvements = {}
        for name, mod in mods.items():
            try:
                hist = list(self.performance_history)[-50:]
                res = mod.self_improve(hist)
                improvements[name] = res
            except Exception as exc:
                improvements[name] = {"error": str(exc)}
        result = {"timestamp": time.time(), "weights": dict(self.module_weights), "avg_scores": avg_scores, "improvements": improvements}
        self.performance_history.append(result)
        self.event_bus.publish("SELF_IMPROVEMENT", result, source="AutonomyOrganism")
        return result

    def feedback(self, module_name: str, reward: float):
        self._feedback_scores[module_name].append(max(0.0, min(1.0, reward)))
        if len(self._feedback_scores[module_name]) > 100:
            self._feedback_scores[module_name] = self._feedback_scores[module_name][-100:]

    def _health_check_loop(self):
        while self._running:
            try:
                time.sleep(self.config.health_check_interval_sec)
                if not self._running:
                    break
                self._run_health_check()
            except Exception as exc:
                log.exception("Health error: %s", exc)

    def _run_health_check(self):
        with self._lock:
            mods = dict(self.modules)
        for name, mod in mods.items():
            if mod.health.error_count >= self.config.max_module_failures and mod.enabled and self.config.isolated_on_failure:
                mod.enabled = False
                mod.health.status = "isolated"
                self.event_bus.publish("MODULE_HEALTH", {"module": name, "status": "isolated"}, source="AutonomyOrganism")

    def _on_order_filled(self, event):
        payload = event.payload if hasattr(event, "payload") else event
        if isinstance(payload, dict):
            mod_name = payload.get("source_module")
            pnl = payload.get("pnl")
            if mod_name and pnl is not None:
                reward = 1.0 if pnl > 0 else 0.0
                self.feedback(mod_name, reward)

    def get_status(self):
        with self._lock:
            return {"modules": {name: mod.to_dict() for name, mod in self.modules.items()}, "weights": dict(self.module_weights)}

_global_organism = None
_global_lock = threading.Lock()

def get_organism():
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
