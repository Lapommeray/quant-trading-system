"""
Organism - autonomous self-wiring trading organism.

Responsibilities:
- Automatic module discovery & registration (scans advanced_modules + core)
- Dependency graph resolution & wiring via EventBus
- Self-improvement loop (periodic evaluation, weight adjustment, feedback)
- Health monitoring & isolation of failing modules
- Integration point for event-driven OKX execution
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base_module import BaseTradingModule, get_registered_modules
from .event_bus import EventBus, get_event_bus

log = logging.getLogger(__name__)


@dataclass
class OrganismConfig:
    auto_discover: bool = True
    self_improvement_interval_sec: int = 300  # 5 min
    health_check_interval_sec: int = 30
    enable_self_improvement: bool = True
    isolated_on_failure: bool = True
    max_module_failures: int = 5
    log_dir: str = "data/organism_logs"


class ModuleAutoDiscovery:
    """Discover modules implementing BaseTradingModule or legacy protocol."""

    @staticmethod
    def discover_decorated() -> Dict[str, Type[BaseTradingModule]]:
        """Return modules registered via @register_module."""
        return get_registered_modules()

    @staticmethod
    def discover_in_package(package_name: str) -> Dict[str, Type[BaseTradingModule]]:
        """Import all submodules of package_name and collect BaseTradingModule subclasses."""
        discovered: Dict[str, Type[BaseTradingModule]] = {}
        try:
            pkg = importlib.import_module(package_name)
        except ImportError as exc:
            log.debug("Package %s not importable: %s", package_name, exc)
            return discovered

        # Also include already decorated modules
        discovered.update(ModuleAutoDiscovery.discover_decorated())

        # Walk through package
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
                    if issubclass(obj, BaseTradingModule) and obj is not BaseTradingModule:
                        mod_key = getattr(obj, "module_name", obj.__name__)
                        discovered[mod_key] = obj
            except Exception as exc:
                log.debug("Failed to import %s: %s", mod_name, exc)
                continue
        return discovered

    @staticmethod
    def discover_legacy_advanced_modules(base_path: Path) -> Dict[str, Any]:
        """Legacy discovery for advanced_modules that don't inherit BaseTradingModule but have initialize()."""
        legacy: Dict[str, Any] = {}
        adv_path = base_path / "advanced_modules"
        if not adv_path.exists():
            return legacy
        for py_file in adv_path.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name == "module_interface.py":
                continue
            mod_name = py_file.stem
            # Try to import via importlib spec
            try:
                spec = importlib.util.spec_from_file_location(
                    f"advanced_modules.{mod_name}", str(py_file)
                )
                if not spec or not spec.loader:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                # Find class with initialize method
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and hasattr(attr, "initialize") and hasattr(attr, "__init__"):
                        legacy[mod_name] = attr
                        break
            except Exception as exc:
                log.debug("Legacy discovery failed for %s: %s", mod_name, exc)
                continue
        return legacy


class Organism:
    """Central autonomous organism wiring all modules."""

    def __init__(self, config: Optional[OrganismConfig] = None, event_bus: Optional[EventBus] = None):
        self.config = config or OrganismConfig()
        self.event_bus = event_bus or get_event_bus()
        self.modules: Dict[str, BaseTradingModule] = {}
        self.legacy_modules: Dict[str, Any] = {}
        self.module_weights: Dict[str, float] = {}
        self.performance_history: deque[Dict[str, Any]] = deque(maxlen=500)
        self._running = False
        self._lock = threading.RLock()
        self._improvement_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._feedback_scores: Dict[str, List[float]] = defaultdict(list)

        # Subscribe to relevant bus events
        self.event_bus.subscribe("MODULE_HEALTH", self._on_module_health)
        self.event_bus.subscribe("ORDER_FILLED", self._on_order_filled)

        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    # ---------- discovery & wiring ----------
    def discover_and_wire(self, project_root: Optional[Path] = None) -> Dict[str, Any]:
        root = project_root or Path(__file__).resolve().parents[1]
        log.info("Organism discovery in %s", root)

        # 1. Decorated modules
        decorated = ModuleAutoDiscovery.discover_decorated()
        log.info("Found %d decorated modules", len(decorated))

        # 2. Package scan
        for pkg in ["advanced_modules", "core", "alpha_intelligence_modules"]:
            pkg_mods = ModuleAutoDiscovery.discover_in_package(pkg)
            decorated.update(pkg_mods)

        # 3. Legacy advanced modules fallback
        legacy = ModuleAutoDiscovery.discover_legacy_advanced_modules(root)
        log.info("Found %d legacy advanced modules", len(legacy))

        # Instantiate
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
                        # Wire event handlers
                        self._wire_module_events(instance)
                    else:
                        log.warning("Module %s failed initialize()", name)
                except Exception as exc:
                    log.warning("Failed to instantiate module %s: %s", name, exc)

            self.legacy_modules.update(legacy)

        # Normalize weights
        self._normalize_weights()

        self.event_bus.publish("ORGANISM_WIRED", {"modules": list(self.modules.keys()), "legacy": list(legacy.keys())}, source="Organism")

        return {"active": list(self.modules.keys()), "legacy": list(legacy.keys()), "total_active": total}

    def _wire_module_events(self, module: BaseTradingModule):
        """Wire module's on_event to bus if implemented."""
        # Subscribe module to all events, let it filter internally
        # Avoid infinite loop: don't wire if same
        def make_handler(m):
            def handler(event):
                try:
                    m.on_event(event.event_type, event.payload)
                except Exception as exc:
                    log.exception("Module %s on_event error: %s", m.module_name, exc)
                    m.record_failure(str(exc))
            return handler

        self.event_bus.subscribe("*", make_handler(module))

    def _normalize_weights(self):
        with self._lock:
            total_weight = sum(self.module_weights.values())
            if total_weight > 0:
                for k in self.module_weights:
                    self.module_weights[k] /= total_weight

    # ---------- lifecycle ----------
    def start(self):
        if self._running:
            return
        self._running = True
        log.info("Organism starting with %d modules", len(self.modules))

        if self.config.enable_self_improvement:
            self._improvement_thread = threading.Thread(target=self._self_improvement_loop, name="OrganismSelfImprove", daemon=True)
            self._improvement_thread.start()

        self._health_thread = threading.Thread(target=self._health_check_loop, name="OrganismHealth", daemon=True)
        self._health_thread.start()

        self.event_bus.publish("ORGANISM_STARTED", {"modules": len(self.modules)}, source="Organism")

    def stop(self):
        self._running = False
        if self._improvement_thread:
            self._improvement_thread.join(timeout=2)
        if self._health_thread:
            self._health_thread.join(timeout=2)
        self.event_bus.publish("ORGANISM_STOPPED", {}, source="Organism")
        log.info("Organism stopped")

    # ---------- signal path ----------
    def generate_consensus_signal(self, symbol: str, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect signals from all modules, produce consensus."""
        results: Dict[str, Any] = {}
        confidences: Dict[str, float] = {}
        directions: Dict[str, str] = {}

        with self._lock:
            modules_snapshot = dict(self.modules)

        start = time.time()
        for name, mod in modules_snapshot.items():
            if not mod.enabled:
                continue
            try:
                t0 = time.time()
                res = mod.generate_signal(symbol, history_data)
                latency_ms = (time.time() - t0) * 1000
                mod.record_success(latency_ms)
                results[name] = res
                # Extract confidence/direction
                conf = getattr(res, "confidence", 0.0) if hasattr(res, "confidence") else res.get("confidence", 0.0) if isinstance(res, dict) else 0.0
                direction = getattr(res, "signal", None) if hasattr(res, "signal") else res.get("direction") or res.get("signal") if isinstance(res, dict) else None
                if direction:
                    dir_up = str(direction).upper()
                    if dir_up in ("BUY", "SELL", "NEUTRAL"):
                        directions[name] = dir_up
                        confidences[name] = float(conf)
            except Exception as exc:
                mod.record_failure(str(exc))
                log.warning("Module %s signal failed: %s", name, exc)

        # Weighted voting
        vote: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}
        with self._lock:
            weights = dict(self.module_weights)

        for mod_name, direction in directions.items():
            w = weights.get(mod_name, 0.0) * confidences.get(mod_name, 0.0)
            vote[direction] += w

        # Determine final
        final_direction = max(vote, key=lambda k: vote[k]) if sum(vote.values()) > 0 else "NEUTRAL"
        total_conf = sum(confidences.values()) / len(confidences) if confidences else 0.0
        weighted_conf = sum(vote.values())  # already weighted

        consensus = {
            "symbol": symbol,
            "final_signal": final_direction if weighted_conf > 0.3 else None,
            "confidence": total_conf,
            "weighted_confidence": weighted_conf,
            "votes": vote,
            "module_results": {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in results.items()},
            "latency_ms": (time.time() - start) * 1000,
            "timestamp": time.time(),
        }

        self.event_bus.publish("SIGNAL_GENERATED", consensus, source="Organism")
        return consensus

    # ---------- self-improvement ----------
    def _self_improvement_loop(self):
        log.info("Self-improvement loop started interval %ds", self.config.self_improvement_interval_sec)
        while self._running:
            try:
                time.sleep(self.config.self_improvement_interval_sec)
                if not self._running:
                    break
                self.run_self_improvement_cycle()
            except Exception as exc:
                log.exception("Self-improvement loop error: %s", exc)

    def run_self_improvement_cycle(self) -> Dict[str, Any]:
        """Evaluate modules, adjust weights, call self_improve hooks."""
        with self._lock:
            mods = dict(self.modules)

        improvements: Dict[str, Any] = {}
        # Compute simple performance proxy from feedback_scores
        avg_scores: Dict[str, float] = {}
        for mod_name, scores in self._feedback_scores.items():
            if scores:
                avg_scores[mod_name] = sum(scores[-20:]) / len(scores[-20:])

        # Adjust weights: boost high performers, decay low
        with self._lock:
            for name in self.module_weights:
                score = avg_scores.get(name, 0.5)
                # Score 0-1, adjust weight by +/-10%
                if score > 0.6:
                    self.module_weights[name] *= 1.05
                elif score < 0.4:
                    self.module_weights[name] *= 0.95
            self._normalize_weights()

        for name, mod in mods.items():
            try:
                hist = list(self.performance_history)[-50:]
                res = mod.self_improve(hist)
                improvements[name] = res
            except Exception as exc:
                log.warning("Self-improve failed for %s: %s", name, exc)
                improvements[name] = {"error": str(exc)}

        result = {
            "timestamp": time.time(),
            "weights": dict(self.module_weights),
            "avg_scores": avg_scores,
            "improvements": improvements,
        }

        self.performance_history.append(result)
        self.event_bus.publish("SELF_IMPROVEMENT", result, source="Organism")
        log.info("Self-improvement cycle complete: %d modules improved", sum(1 for v in improvements.values() if v.get("improved")))

        # Persist log
        try:
            log_path = Path(self.config.log_dir) / f"self_improve_{int(time.time())}.json"
            import json
            with open(log_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception:
            pass

        return result

    def feedback(self, module_name: str, reward: float):
        """External reward signal for module performance (e.g., from trade PnL)."""
        self._feedback_scores[module_name].append(max(0.0, min(1.0, reward)))
        if len(self._feedback_scores[module_name]) > 100:
            self._feedback_scores[module_name] = self._feedback_scores[module_name][-100:]

    # ---------- health ----------
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
            # If failure count exceeded, isolate
            if mod.health.error_count >= self.config.max_module_failures and mod.enabled and self.config.isolated_on_failure:
                mod.enabled = False
                mod.health.status = "isolated"
                log.warning("Module %s isolated after %d failures", name, mod.health.error_count)
                self.event_bus.publish("MODULE_HEALTH", {"module": name, "status": "isolated", "reason": mod.health.last_error}, source="Organism")
            # Stale heartbeat
            if time.time() - mod.health.last_heartbeat > self.config.health_check_interval_sec * 3:
                log.debug("Module %s heartbeat stale", name)

    # ---------- events ----------
    def _on_module_health(self, event):
        log.debug("Health event: %s", event.payload)

    def _on_order_filled(self, event):
        payload = event.payload
        mod_name = payload.get("source_module")
        pnl = payload.get("pnl")
        if mod_name and pnl is not None:
            reward = 1.0 if pnl > 0 else 0.0
            self.feedback(mod_name, reward)

    # ---------- introspection ----------
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "modules": {name: mod.to_dict() for name, mod in self.modules.items()},
                "weights": dict(self.module_weights),
                "legacy_count": len(self.legacy_modules),
                "performance_history_len": len(self.performance_history),
                "feedback_scores": {k: sum(v[-10:]) / len(v[-10:]) if v else 0.0 for k, v in self._feedback_scores.items()},
            }


# Global singleton accessor
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
