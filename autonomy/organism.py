"""Canonical self-wiring trading organism.

The organism is the integration point for the active runtime.  It provides:

* one canonical event bus shared with ``core`` and ``okx_live``;
* module discovery, health tracking and failure isolation;
* consensus signals with market-regime context;
* durable outcome/mistake memory;
* bounded auto-repair and self-coding artifacts;
* adaptive module weights that never bypass execution/risk guardrails.

Autonomy here means that the system can diagnose and prepare low-risk
improvements by itself.  It does *not* mean that generated code may overwrite
live order, risk, credential or safety code.  Those paths stay protected and
require a human promotion step.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import json
import logging
import pkgutil
import shlex
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

from core.base_module import (
    BaseTradingModule as _CoreBaseTradingModule,
    ModuleHealth,
    ModuleResult,
    get_registered_modules as _get_core_registered_modules,
    register_module as _core_register_module,
)
from core.event_bus import EventBus, Event, get_event_bus as _get_core_event_bus

from .audit import AuditTrail
from .gold_set import GoldSetStressTester
from .learning import LearningStore
from .market import MarketRegime, MarketRegimeDetector
from .monitor import AutonomousMonitor
from .self_coding import ApprovalPolicy, SelfCodingEngine
from .sentinel import MultiTimeframeSentinel, SentinelConfig, SentinelDecision
from .shadow import ShadowDeployment, ShadowManager, ShadowPolicy

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ bus API
# Keep these names for compatibility with the pre-canonical autonomy module;
# they now point at the same event bus used by core and the OKX adapters.
_LocalEvent = Event
_LocalEventBus = EventBus


def get_event_bus() -> EventBus:
    return _get_core_event_bus()


# -------------------------------------------------------------- module API
MODULE_REGISTRY: Dict[str, Type["BaseModule"]] = {}


class BaseModule(_CoreBaseTradingModule):
    """Concrete compatibility base with autonomous lifecycle hooks."""

    module_name = "base"
    category = "general"
    version = "1.0.0"
    dependencies: List[str] = []

    def initialize(self) -> bool:
        return True

    def analyze(self, market_data: Dict[str, Any]) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name, signal="NEUTRAL", confidence=0.0
        )

    def generate_signal(
        self, symbol: str, history_data: Dict[str, Any]
    ) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name, signal="NEUTRAL", confidence=0.0
        )


# Compatibility name used by older integrations.  New code may use either
# BaseModule or the canonical BaseTradingModule spelling.
BaseTradingModule = BaseModule


def register_module(cls=None, *, name: Optional[str] = None):
    """Register a module in both the autonomy and core registries."""

    def decorator(inner_cls):
        reg_name = name or getattr(inner_cls, "module_name", inner_cls.__name__)
        MODULE_REGISTRY[reg_name] = inner_cls
        try:
            _core_register_module(inner_cls, name=reg_name)
        except Exception:
            # A plugin can still participate in the autonomy registry if a
            # legacy core registry is unavailable.
            pass
        return inner_cls

    return decorator(cls) if cls is not None else decorator


def get_registered_modules() -> Dict[str, Type[BaseModule]]:
    """Return the union of decorated autonomy and core modules."""

    merged: Dict[str, Type[BaseModule]] = dict(_get_core_registered_modules())  # type: ignore[assignment]
    merged.update(MODULE_REGISTRY)
    return merged


@dataclass
class OrganismConfig:
    auto_discover: bool = True
    self_improvement_interval_sec: int = 300
    health_check_interval_sec: int = 30
    enable_self_improvement: bool = True
    isolated_on_failure: bool = True
    max_module_failures: int = 5
    log_dir: str = "data/organism_logs"
    audit_path: Optional[str] = None
    min_confidence: float = 0.60
    min_weighted_confidence: float = 0.30

    # Autonomous learning/coding controls.
    learning_path: Optional[str] = None
    self_coding_dir: Optional[str] = None
    self_coding_enabled: bool = True
    auto_approve_low_risk: bool = True
    auto_apply_low_risk: bool = True
    auto_repair: bool = True
    self_code_each_cycle: bool = True
    max_auto_changes_per_cycle: int = 3
    market_regime_window: int = 50
    module_packages: tuple[str, ...] = ("autonomy", "core", "advanced_modules")
    run_baseline_tests: bool = True
    baseline_test_command: tuple[str, ...] = ()
    shadow_min_observations: int = 100
    shadow_min_outperformance: float = 0.05
    shadow_max_drawdown_delta: float = 0.01
    auto_promote_shadows: bool = True
    sentinel_sigma_threshold: float = 3.0
    sentinel_stabilization_observations: int = 3

    @classmethod
    def from_env(cls) -> "OrganismConfig":
        """Build operational controls from environment variables.

        This keeps deployment configuration out of generated artifacts and
        makes the self-coding boundary explicit in container/CLI launches.
        """
        import os

        truthy = {"1", "true", "yes", "on"}
        return cls(
            self_improvement_interval_sec=int(
                os.getenv("ORGANISM_SELF_IMPROVE_SEC", "300")
            ),
            health_check_interval_sec=int(os.getenv("ORGANISM_HEALTH_CHECK_SEC", "30")),
            enable_self_improvement=os.getenv(
                "ORGANISM_ENABLE_SELF_IMPROVEMENT", "true"
            ).lower()
            in truthy,
            self_coding_enabled=os.getenv("ORGANISM_SELF_CODING", "true").lower()
            in truthy,
            auto_approve_low_risk=os.getenv(
                "ORGANISM_AUTO_APPROVE_LOW_RISK", "true"
            ).lower()
            in truthy,
            auto_apply_low_risk=os.getenv(
                "ORGANISM_AUTO_APPLY_LOW_RISK", "true"
            ).lower()
            in truthy,
            auto_repair=os.getenv("ORGANISM_AUTO_REPAIR", "true").lower() in truthy,
            max_auto_changes_per_cycle=int(os.getenv("ORGANISM_MAX_AUTO_CHANGES", "3")),
            learning_path=os.getenv("QTS_LEARNING_PATH") or None,
            self_coding_dir=os.getenv("QTS_SELF_CODING_DIR") or None,
            audit_path=os.getenv("QTS_AUDIT_PATH") or None,
            run_baseline_tests=os.getenv("ORGANISM_RUN_BASELINE_TESTS", "true").lower()
            in truthy,
            baseline_test_command=tuple(
                shlex.split(os.getenv("QTS_BASELINE_TEST_COMMAND", ""))
            ),
            shadow_min_observations=int(os.getenv("ORGANISM_SHADOW_TICKS", "100")),
            shadow_min_outperformance=float(
                os.getenv("ORGANISM_SHADOW_OUTPERFORMANCE", "0.05")
            ),
            shadow_max_drawdown_delta=float(
                os.getenv("ORGANISM_SHADOW_DD_DELTA", "0.01")
            ),
            auto_promote_shadows=os.getenv(
                "ORGANISM_AUTO_PROMOTE_SHADOWS", "true"
            ).lower()
            in truthy,
            sentinel_sigma_threshold=float(os.getenv("ORGANISM_SENTINEL_SIGMA", "3.0")),
            sentinel_stabilization_observations=int(
                os.getenv("ORGANISM_SENTINEL_STABLE_TICKS", "3")
            ),
        )


class ModuleAutoDiscovery:
    """Discover decorated/new-contract modules without importing archives."""

    @staticmethod
    def discover_decorated() -> Dict[str, Type[_CoreBaseTradingModule]]:
        return get_registered_modules()  # type: ignore[return-value]

    @staticmethod
    def discover_in_package(
        package_name: str,
    ) -> Dict[str, Type[_CoreBaseTradingModule]]:
        discovered: Dict[str, Type[_CoreBaseTradingModule]] = {}
        try:
            package = importlib.import_module(package_name)
        except Exception as exc:
            log.debug("Could not import package %s: %s", package_name, exc)
            return discovered
        package_path = getattr(package, "__path__", None)
        if not package_path:
            return discovered

        prefix = package.__name__ + "."
        for _, module_name, is_package in pkgutil.iter_modules(package_path, prefix):
            if is_package or module_name.rsplit(".", 1)[-1].startswith("_"):
                continue
            try:
                module = importlib.import_module(module_name)
                for _, candidate in inspect.getmembers(module, inspect.isclass):
                    if candidate in (_CoreBaseTradingModule, BaseModule):
                        continue
                    try:
                        is_module = issubclass(candidate, _CoreBaseTradingModule)
                    except TypeError:
                        is_module = False
                    if is_module and candidate.__module__ == module.__name__:
                        name = getattr(candidate, "module_name", candidate.__name__)
                        discovered[str(name)] = candidate
            except Exception as exc:
                # Optional research modules are allowed to fail import; they
                # are not part of the active runtime until they meet the
                # contract and their dependencies are installed.
                log.debug("Failed to inspect %s: %s", module_name, exc)
        return discovered


# --------------------------------------------------------------- utilities
def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in list(value)[:100]]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    # pandas/numpy scalar values can usually be represented by float/int.
    for converter in (float, int):
        try:
            converted = converter(value)
            return converted
        except (TypeError, ValueError, OverflowError):
            continue
    return str(value)


def _result_signal(result: Any) -> tuple[Optional[str], float, Dict[str, Any]]:
    if isinstance(result, ModuleResult):
        return result.signal, float(result.confidence), result.to_dict()
    if isinstance(result, Mapping):
        signal = result.get("signal", result.get("direction"))
        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return signal, confidence, dict(result)
    return None, 0.0, {"value": result}


class Organism:
    """Self-wiring, learning and bounded self-improvement coordinator."""

    def __init__(
        self,
        config: Optional[OrganismConfig] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.config = config or OrganismConfig()
        self.event_bus = event_bus or get_event_bus()
        self.audit_trail = AuditTrail(
            self.config.audit_path or "audit_logs/autonomous_events.jsonl"
        )
        self.survival_audit = AuditTrail("audit_logs/survival_mode.jsonl")
        self.shadow_audit = AuditTrail("audit_logs/shadow_promotions.jsonl")
        self.audit_trail.subscribe(self.event_bus)
        self.modules: Dict[str, _CoreBaseTradingModule] = {}
        self.module_weights: Dict[str, float] = {}
        self.performance_history: deque[Dict[str, Any]] = deque(maxlen=500)
        self._feedback_scores: Dict[str, List[float]] = defaultdict(list)
        self._last_regime = MarketRegime()
        self._running = False
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._improvement_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._module_handlers: Dict[str, Callable] = {}
        self.monitor = AutonomousMonitor(
            max_module_errors=self.config.max_module_failures,
            error_callback=self._monitor_quarantine,
            resource_callback=self._monitor_resource_pressure,
        )

        self.learning_store = LearningStore(path=self.config.learning_path)
        self.market_detector = MarketRegimeDetector(
            window=self.config.market_regime_window
        )
        self.sentinel = MultiTimeframeSentinel(
            SentinelConfig(
                sigma_threshold=self.config.sentinel_sigma_threshold,
                stabilization_observations=self.config.sentinel_stabilization_observations,
            )
        )
        self._last_sentinel_decision = SentinelDecision(False, False, False)
        self.shadow_manager = ShadowManager(
            policy=ShadowPolicy(
                min_observations=self.config.shadow_min_observations,
                min_outperformance=self.config.shadow_min_outperformance,
                max_drawdown_delta=self.config.shadow_max_drawdown_delta,
            ),
            gold_tester=GoldSetStressTester(),
            promote_callback=self._promote_shadow,
        )
        policy = ApprovalPolicy(
            auto_approve_low_risk=self.config.auto_approve_low_risk,
            auto_apply_low_risk=self.config.auto_apply_low_risk,
            run_baseline_tests=self.config.run_baseline_tests,
            baseline_command=self.config.baseline_test_command,
        )
        self.self_coder = SelfCodingEngine(
            project_root=Path(__file__).resolve().parents[1],
            artifact_dir=self.config.self_coding_dir,
            policy=policy,
            event_bus=self.event_bus,
        )
        # Semantic aliases keep integrations readable and preserve the
        # expected "self-improvement engine" vocabulary.
        self.self_improvement_engine = self.self_coder
        self.code_generator = self.self_coder
        self._coding_cycles = 0
        self._last_cycle: Dict[str, Any] = {}
        self.consensus_engine = __import__(
            "autonomy.consensus", fromlist=["ConsensusEngine"]
        ).ConsensusEngine(min_weighted_confidence=self.config.min_weighted_confidence)

        self.event_bus.subscribe("ORDER_FILLED", self._on_order_filled)
        try:
            # Learning is lane 3 work; queue it so a critical execution event
            # is never held behind JSONL persistence or model updates.
            self.event_bus.subscribe(
                "TRADE_OUTCOME", self._on_order_filled, asynchronous=True
            )
        except TypeError:
            self.event_bus.subscribe("TRADE_OUTCOME", self._on_order_filled)

    # -------------------------------------------------------------- discovery
    def discover_and_wire(self, project_root: Optional[Path] = None) -> Dict[str, Any]:
        discovered = ModuleAutoDiscovery.discover_decorated()
        if self.config.auto_discover:
            for package_name in self.config.module_packages:
                for name, candidate in ModuleAutoDiscovery.discover_in_package(
                    package_name
                ).items():
                    discovered.setdefault(name, candidate)

        active: List[str] = []
        failed: Dict[str, str] = {}
        with self._lock:
            for name, candidate in discovered.items():
                if name in self.modules:
                    continue
                try:
                    instance = self._instantiate(candidate, self.event_bus)
                    initialized = bool(instance.initialize())
                    if not initialized:
                        raise RuntimeError("initialize() returned False")
                    self.modules[name] = instance
                    self.module_weights[name] = 1.0
                    self._wire_module_events(instance)
                    active.append(name)
                    self.event_bus.publish(
                        "MODULE_REGISTERED",
                        {
                            "module": name,
                            "category": getattr(instance, "category", "general"),
                        },
                        source="AutonomyOrganism",
                    )
                except Exception as exc:
                    failed[str(name)] = str(exc)
                    log.warning("Failed to instantiate module %s: %s", name, exc)
            self._normalize_weights()

        payload = {
            "modules": active,
            "active": list(self.modules),
            "total_active": len(self.modules),
            "failed": failed,
        }
        self.event_bus.publish("ORGANISM_WIRED", payload, source="AutonomyOrganism")
        return payload

    @staticmethod
    def _instantiate(
        candidate: Type[_CoreBaseTradingModule], event_bus: Any
    ) -> _CoreBaseTradingModule:
        try:
            return candidate(config={}, event_bus=event_bus)
        except TypeError:
            try:
                return candidate(config={})
            except TypeError:
                return candidate()

    def _wire_module_events(self, module: _CoreBaseTradingModule) -> None:
        name = str(getattr(module, "module_name", module.__class__.__name__))

        def handler(event: Event, target=module):
            try:
                target.on_event(event.event_type, event.payload)
            except Exception as exc:
                target.record_failure(str(exc))
                self.monitor.log_error(str(exc), name)
                log.debug("Module %s event error: %s", name, exc)

        self._module_handlers[name] = handler
        self.event_bus.subscribe("*", handler)

    def _normalize_weights(self) -> None:
        with self._lock:
            enabled = {
                name: weight
                for name, weight in self.module_weights.items()
                if self.modules.get(name) and self.modules[name].enabled
            }
            total = sum(max(0.0, weight) for weight in enabled.values())
            if total <= 0:
                return
            for name in self.module_weights:
                self.module_weights[name] = max(0.0, enabled.get(name, 0.0)) / total

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.monitor.start()
        if self.config.enable_self_improvement:
            self._improvement_thread = threading.Thread(
                target=self._self_improvement_loop,
                name="OrganismSelfImprove",
                daemon=True,
            )
            self._improvement_thread.start()
        self._health_thread = threading.Thread(
            target=self._health_check_loop, name="OrganismHealth", daemon=True
        )
        self._health_thread.start()
        self.event_bus.publish(
            "ORGANISM_STARTED",
            {"modules": len(self.modules)},
            source="AutonomyOrganism",
        )

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        for thread in (self._improvement_thread, self._health_thread):
            if thread:
                thread.join(timeout=2)
        self._improvement_thread = None
        self._health_thread = None
        self.monitor.stop()
        self.event_bus.publish("ORGANISM_STOPPED", {}, source="AutonomyOrganism")

    def _self_improvement_loop(self) -> None:
        interval = max(1, int(self.config.self_improvement_interval_sec))
        while self._running and not self._stop_event.wait(interval):
            try:
                self.run_self_improvement_cycle()
            except Exception:
                log.exception("Self-improvement cycle failed")

    def _health_check_loop(self) -> None:
        interval = max(1, int(self.config.health_check_interval_sec))
        while self._running and not self._stop_event.wait(interval):
            try:
                self._run_health_check()
            except Exception:
                log.exception("Health check failed")

    # -------------------------------------------------------------- signals
    def generate_consensus_signal(
        self, symbol: str, history_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            sentinel = self.sentinel.evaluate(history_data)
        except Exception as exc:
            # A dead/buggy sentinel is a safety failure, never permission to
            # continue autonomous trading.
            self.monitor.log_error(str(exc), "panic_sentinel")
            sentinel = self.sentinel.force_survival(f"sentinel unavailable: {exc}")
        if not self.sentinel.heartbeat_ok():
            sentinel = self.sentinel.force_survival("sentinel heartbeat unavailable")
        previous_sentinel = self._last_sentinel_decision
        self._last_sentinel_decision = sentinel
        if sentinel.changed:
            event_type = (
                "SURVIVAL_MODE" if sentinel.survival_mode else "CLEAR_SURVIVAL_MODE"
            )
            sentinel_payload = {
                "symbol": symbol,
                "reason": sentinel.reason,
                "risk_profile": sentinel.risk_profile,
                "sentinel": sentinel.to_dict(),
            }
            self.audit_trail.record(
                event_type, sentinel_payload, source="PanicSentinel"
            )
            self.survival_audit.record(
                event_type,
                sentinel_payload,
                source="PanicSentinel",
            )
            self.event_bus.publish(
                event_type,
                sentinel_payload,
                source="PanicSentinel",
            )
        elif sentinel.survival_mode and not previous_sentinel.survival_mode:
            self.event_bus.publish(
                "SURVIVAL_MODE",
                {
                    "symbol": symbol,
                    "reason": sentinel.reason,
                    "risk_profile": sentinel.risk_profile,
                },
                source="PanicSentinel",
            )

        regime = self.market_detector.detect(history_data)
        self._last_regime = regime
        self.learning_store.record_regime(regime.label, regime.to_dict())
        self.event_bus.publish(
            "MARKET_REGIME",
            {"symbol": symbol, "regime": regime.label, "regime_data": regime.to_dict()},
            source="MarketRegimeDetector",
        )

        results: Dict[str, Any] = {}
        directions: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        prediction_ids: Dict[str, str] = {}
        started = time.time()
        with self._lock:
            snapshot = dict(self.modules)
            weights = dict(self.module_weights)
        if sentinel.survival_mode:
            # Keep collecting observations for diagnosis, but do not let
            # adaptive weights influence a safety decision.
            active_count = max(
                1, sum(getattr(module, "enabled", True) for module in snapshot.values())
            )
            weights = {
                name: (1.0 / active_count if getattr(module, "enabled", True) else 0.0)
                for name, module in snapshot.items()
            }

        for name, module in snapshot.items():
            if not getattr(module, "enabled", True):
                continue
            try:
                before = time.time()
                result = module.generate_signal(symbol, history_data)
                latency_ms = (time.time() - before) * 1000.0
                module.record_success(latency_ms)
                signal, confidence, result_dict = _result_signal(result)
                result_dict["latency_ms"] = latency_ms
                result_dict["module_name"] = name
                results[name] = _json_safe(result_dict)
                normalized_signal = str(signal).upper() if signal is not None else None
                if normalized_signal in {"BUY", "SELL", "NEUTRAL"}:
                    directions[name] = normalized_signal
                    confidences[name] = max(0.0, min(1.0, confidence))
                    prediction_ids[name] = self.learning_store.record_prediction(
                        module_name=name,
                        symbol=symbol,
                        signal=normalized_signal,
                        confidence=confidence,
                        regime=regime.label,
                        context={"features": result_dict.get("features", {})},
                    )
            except Exception as exc:
                module.record_failure(str(exc))
                self.monitor.log_error(str(exc), name)
                results[name] = {"module_name": name, "error": str(exc)}

        consensus = self.consensus_engine.compute(
            symbol=symbol,
            directions=directions,
            confidences=confidences,
            weights=weights,
            module_results=results,
            latency_ms=(time.time() - started) * 1000.0,
        )
        final_signal = consensus.final_signal
        if consensus.confidence < self.config.min_confidence or sentinel.survival_mode:
            final_signal = None
        source_module = next(
            (
                name
                for name, direction in directions.items()
                if direction == final_signal
            ),
            None,
        )
        weighted_confidence = (
            0.0 if sentinel.survival_mode else consensus.weighted_confidence
        )
        confidence = 0.0 if sentinel.survival_mode else consensus.confidence
        shadow_status = {}
        if not sentinel.survival_mode:
            shadow_status = self.shadow_manager.observe(
                symbol,
                history_data,
                promote=self.config.auto_promote_shadows,
            )
        else:
            # Candidate and active clones still see the same observations, but
            # no candidate can promote during a panic regime.
            shadow_status = self.shadow_manager.observe(
                symbol, history_data, promote=False
            )
        for shadow in shadow_status.values():
            if shadow.get("status") == "rejected":
                comparison = shadow.get("comparison", {})
                gold_report = comparison.get("gold_report") or {}
                failures = gold_report.get("failures") or []
                if failures:
                    self.learning_store.record_mistake(
                        module_name=shadow.get("module_name", "unknown"),
                        lesson="Dangerous overfitting rejected by gold set: "
                        + "; ".join(failures),
                        regime=regime.label,
                    )
        payload = {
            "symbol": consensus.symbol,
            "final_signal": final_signal,
            "source_module": source_module,
            "confidence": confidence,
            "weighted_confidence": weighted_confidence,
            "votes": consensus.votes,
            "module_results": consensus.module_results,
            "prediction_ids": prediction_ids,
            "regime": regime.label,
            "regime_data": regime.to_dict(),
            "survival_mode": sentinel.survival_mode,
            "survival_reason": sentinel.reason if sentinel.survival_mode else "",
            "risk_profile": sentinel.risk_profile,
            "sentinel": sentinel.to_dict(),
            "shadow": shadow_status,
            "latency_ms": consensus.latency_ms,
            "timestamp": consensus.timestamp,
        }
        self.event_bus.publish("CONSENSUS_REACHED", payload, source="AutonomyOrganism")
        self.event_bus.publish("SIGNAL_GENERATED", payload, source="AutonomyOrganism")
        return payload

    # --------------------------------------------------------------- learning
    def feedback(
        self,
        module_name: str,
        reward: float,
        *,
        pnl: Optional[float] = None,
        prediction_id: Optional[str] = None,
        symbol: Optional[str] = None,
        reason: str = "feedback",
    ) -> Dict[str, Any]:
        reward = max(0.0, min(1.0, float(reward)))
        with self._lock:
            scores = self._feedback_scores[module_name]
            scores.append(reward)
            if len(scores) > 100:
                del scores[:-100]
            module = self.modules.get(module_name)
        if prediction_id or pnl is not None:
            record = self.learning_store.record_outcome(
                prediction_id=prediction_id,
                module_name=module_name,
                symbol=symbol,
                pnl=pnl,
                reward=reward,
                reason=reason,
                regime=self._last_regime.label,
            )
        else:
            record = self.learning_store.record_feedback(
                module_name=module_name,
                reward=reward,
                symbol=symbol,
                regime=self._last_regime.label,
                reason=reason,
            )
        if module is not None:
            try:
                module.learn_from_outcome(
                    {"reward": reward, "pnl": pnl, "reason": reason, "record": record}
                )
            except Exception as exc:
                module.record_failure(str(exc))
        self.event_bus.publish(
            "LEARNING_FEEDBACK",
            {
                "module": module_name,
                "reward": reward,
                "pnl": pnl,
                "prediction_id": prediction_id,
                "reason": reason,
            },
            source="AutonomyOrganism",
        )
        return record

    # Backwards-compatible vocabulary for integrations that call the feedback
    # loop "learn".
    learn = feedback

    def _on_order_filled(self, event: Event) -> None:
        payload = event.payload if hasattr(event, "payload") else event
        if not isinstance(payload, Mapping):
            return
        pnl = payload.get("realized_pnl", payload.get("pnl"))
        # A fill acknowledgement is not a realized outcome.  Wait for a
        # reconciliation event carrying realized PnL so the learning store
        # does not permanently record an unknown fill as a win/loss.
        if pnl is None:
            return
        prediction_ids = payload.get("prediction_ids", {})
        source_module = payload.get("source_module") or payload.get("module_name")
        symbol = payload.get("symbol")
        try:
            pnl_value = float(pnl) if pnl is not None else None
        except (TypeError, ValueError):
            pnl_value = None
        if pnl_value is None:
            return

        if isinstance(prediction_ids, Mapping) and prediction_ids:
            # All modules that contributed to the signal receive the same
            # realized outcome; the store deduplicates repeated fill events.
            for module_name, prediction_id in prediction_ids.items():
                self.feedback(
                    str(module_name),
                    0.5 if pnl_value is None else (1.0 if pnl_value > 0 else 0.0),
                    pnl=pnl_value,
                    prediction_id=str(prediction_id),
                    symbol=symbol,
                    reason="order fill reconciliation",
                )
        elif source_module:
            self.feedback(
                str(source_module),
                0.5 if pnl_value is None else (1.0 if pnl_value > 0 else 0.0),
                pnl=pnl_value,
                symbol=symbol,
                reason="order fill without prediction id",
            )

    # ---------------------------------------------------------- improvement
    def _module_context(self, name: str) -> Dict[str, Any]:
        stats = self.learning_store.module_stats(name, regime=self._last_regime.label)
        mistakes = self.learning_store.mistakes(name, limit=10)
        return {
            "regime": self._last_regime.label,
            "regime_data": self._last_regime.to_dict(),
            "stats": stats,
            "mistakes": mistakes,
            "feedback": list(self._feedback_scores.get(name, []))[-20:],
        }

    def _update_weights(self) -> Dict[str, float]:
        with self._lock:
            for name, module in self.modules.items():
                stats = self.learning_store.module_stats(
                    name, regime=self._last_regime.label
                )
                if stats["sample_count"] == 0:
                    feedback = self._feedback_scores.get(name, [])
                    reward = (
                        sum(feedback[-20:]) / len(feedback[-20:]) if feedback else 0.5
                    )
                else:
                    reward = stats["avg_reward"]
                affinity = self.market_detector.module_affinity(
                    module, self._last_regime
                )
                # Small, damped updates prevent a few trades from taking over
                # the consensus and retain a non-zero exploration weight.
                factor = max(0.75, min(1.25, 1.0 + (reward - 0.5) * 0.20)) * affinity
                self.module_weights[name] = max(
                    0.001, self.module_weights.get(name, 1.0) * factor
                )
            self._normalize_weights()
            return dict(self.module_weights)

    def _safe_runtime_tuning(self, module: Any) -> Dict[str, Any]:
        if self._last_sentinel_decision.survival_mode:
            return {"skipped": "survival mode hard profile"}
        # Do not write a newly learned regime parameter into the active
        # module.  The parameter is applied to the candidate shadow clone and
        # reaches active only after shadow + gold-set promotion.
        return {
            "proposed": self.market_detector.adaptive_parameters(self._last_regime),
            "applied_to_active": False,
            "module": getattr(module, "module_name", module.__class__.__name__),
        }

    def run_self_improvement_cycle(self) -> Dict[str, Any]:
        if self._last_sentinel_decision.survival_mode:
            result = {
                "timestamp": time.time(),
                "cycle": self._coding_cycles + 1,
                "regime": self._last_regime.to_dict(),
                "survival_mode": True,
                "weights": dict(self.module_weights),
                "avg_scores": {},
                "improvements": {},
                "auto_applied": 0,
                "shadow_deployed": 0,
                "reason": "evolutionary lane paused during survival mode",
                "self_coder": self.self_coder.get_status(),
            }
            self._coding_cycles += 1
            self.performance_history.append(result)
            self._last_cycle = result
            self._persist_cycle(result)
            self.event_bus.publish(
                "SELF_IMPROVEMENT", result, source="AutonomyOrganism"
            )
            return result
        with self._lock:
            modules = dict(self.modules)
        weights = self._update_weights()
        regime_parameters = self.market_detector.adaptive_parameters(self._last_regime)
        improvements: Dict[str, Any] = {}
        auto_applied = 0
        shadow_deployed = 0

        for name, module in modules.items():
            context = self._module_context(name)
            runtime_tuning = self._safe_runtime_tuning(module)
            try:
                hook_result = module.self_improve(list(self.performance_history)[-50:])
            except Exception as exc:
                module.record_failure(str(exc))
                hook_result = {"module": name, "error": str(exc)}

            code_result: Optional[Dict[str, Any]] = None
            if self.config.self_coding_enabled and (
                self.config.self_code_each_cycle or context["stats"]["mistakes"] > 0
            ):
                should_apply = auto_applied < max(
                    0, int(self.config.max_auto_changes_per_cycle)
                )
                try:
                    code_result = self.self_coder.run_for_module(
                        module,
                        context=context,
                        regime_parameters=regime_parameters,
                        apply=should_apply,
                    )
                    if code_result.get("status") == "applied":
                        auto_applied += 1
                    if (
                        code_result.get("status") in {"applied", "approved"}
                        and code_result.get("risk") == "low"
                    ):
                        deployment = self.shadow_manager.deploy(
                            module_name=name,
                            proposal_id=code_result.get("proposal_id", "unknown"),
                            active_module=module,
                            candidate_parameters=code_result.get("parameters", {}),
                        )
                        code_result["shadow_id"] = deployment.shadow_id
                        code_result["deployment"] = "shadow"
                        proposal = self.self_coder.proposals.get(
                            code_result.get("proposal_id", "")
                        )
                        if proposal is not None:
                            proposal.shadow_id = deployment.shadow_id
                            proposal.deployment = "shadow"
                        shadow_deployed += 1
                except Exception as exc:
                    code_result = {"module_name": name, "error": str(exc)}
                if code_result and (
                    code_result.get("status") == "rejected" or code_result.get("error")
                ):
                    diagnosis = code_result.get("diagnosis", {})
                    lesson = (
                        diagnosis.get("validation_mistake")
                        or code_result.get("error")
                        or "candidate rejected"
                    )
                    self.learning_store.record_mistake(
                        module_name=name,
                        lesson=f"Self-coder discarded candidate: {lesson}",
                        regime=self._last_regime.label,
                    )

            improvements[name] = {
                "hook": _json_safe(hook_result),
                "runtime_tuning": _json_safe(runtime_tuning),
                "code": _json_safe(code_result),
                "stats": context["stats"],
            }

        avg_scores = {
            name: self.learning_store.module_stats(
                name, regime=self._last_regime.label
            ).get("avg_reward", 0.5)
            for name in modules
        }
        result = {
            "timestamp": time.time(),
            "cycle": self._coding_cycles + 1,
            "regime": self._last_regime.to_dict(),
            "weights": weights,
            # Kept for compatibility with the original organism API.
            "avg_scores": avg_scores,
            "improvements": improvements,
            "auto_applied": auto_applied,
            "shadow_deployed": shadow_deployed,
            "survival_mode": self._last_sentinel_decision.survival_mode,
            "self_coder": self.self_coder.get_status(),
        }
        self._coding_cycles += 1
        self.performance_history.append(result)
        self._last_cycle = result
        self._persist_cycle(result)
        self.event_bus.publish("SELF_IMPROVEMENT", result, source="AutonomyOrganism")
        return result

    def _persist_cycle(self, result: Dict[str, Any]) -> None:
        directory = Path(self.config.log_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"cycle_{int(result['timestamp'] * 1000)}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(_json_safe(result), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as exc:
            log.warning("Could not persist self-improvement cycle: %s", exc)

    def _promote_shadow(self, deployment: ShadowDeployment) -> bool:
        """Swap a candidate only after shadow and gold-set gates pass."""
        with self._lock:
            active = self.modules.get(deployment.module_name)
            if active is None or not getattr(active, "enabled", True):
                return False
            old_handler = self._module_handlers.get(deployment.module_name)
            if old_handler is not None:
                self.event_bus.unsubscribe("*", old_handler)
            candidate = deployment.candidate_module
            candidate.event_bus = self.event_bus
            candidate.enabled = True
            self.modules[deployment.module_name] = candidate
            self._wire_module_events(candidate)
        deployment.status = "promoted"
        proposal = self.self_coder.proposals.get(deployment.proposal_id)
        if proposal is not None:
            proposal.shadow_id = deployment.shadow_id
            proposal.deployment = "promoted"
        promotion_payload = {
            "shadow_id": deployment.shadow_id,
            "proposal_id": deployment.proposal_id,
            "module": deployment.module_name,
            "active_source_modified": False,
            "shadow_comparison": deployment.compare(),
        }
        self.shadow_audit.record(
            "SHADOW_PROMOTED", promotion_payload, source="ShadowManager"
        )
        self.event_bus.publish(
            "SHADOW_PROMOTED",
            promotion_payload,
            source="ShadowManager",
        )
        return True

    def _monitor_resource_pressure(self, metrics: Dict[str, Any]) -> None:
        gc.collect()
        self.audit_trail.record(
            "RESOURCE_PRESSURE", metrics, source="AutonomousMonitor"
        )
        self.event_bus.publish(
            "RISK_ALERT",
            {
                "reason": "resource pressure; evolutionary work throttled",
                "metrics": metrics,
            },
            source="AutonomousMonitor",
        )
        self.sentinel.force_survival("resource pressure")
        self.event_bus.publish(
            "SURVIVAL_MODE",
            {
                "reason": "resource pressure",
                "risk_profile": self.sentinel.RISK_PROFILE,
                "metrics": metrics,
            },
            source="AutonomousMonitor",
        )

    def _monitor_quarantine(self, module_name: str, error: str) -> None:
        module = self.modules.get(module_name)
        if module is None:
            return
        module.enabled = False
        module.health.status = "isolated"
        with self._lock:
            self.module_weights[module_name] = 0.0
            self._normalize_weights()
        self.event_bus.publish(
            "MODULE_HEALTH",
            {"module": module_name, "status": "isolated", "last_error": error},
            source="AutonomousMonitor",
        )

    # --------------------------------------------------------------- health
    def _run_health_check(self) -> None:
        if self._running and not self.sentinel.heartbeat_ok():
            decision = self.sentinel.force_survival("sentinel heartbeat stale")
            payload = {
                "reason": decision.reason,
                "risk_profile": decision.risk_profile,
                "sentinel": decision.to_dict(),
            }
            self.survival_audit.record(
                "SURVIVAL_MODE", payload, source="AutonomousMonitor"
            )
            self.event_bus.publish("SURVIVAL_MODE", payload, source="AutonomousMonitor")
        with self._lock:
            modules = dict(self.modules)
        for name, module in modules.items():
            if module.health.error_count < self.config.max_module_failures:
                continue
            if not module.enabled:
                continue
            if self.config.auto_repair:
                repair = self.self_coder.auto_fix(
                    module,
                    reason=module.health.last_error or "failure threshold reached",
                    context=self._module_context(name),
                    apply=self.config.self_coding_enabled,
                )
                if repair.get("fixed"):
                    module.enabled = True
                    module.health.status = "ok"
                    self.event_bus.publish(
                        "MODULE_HEALTH",
                        {"module": name, "status": "repaired", "repair": repair},
                        source="AutonomyOrganism",
                    )
                    continue
            if self.config.isolated_on_failure:
                module.enabled = False
                module.health.status = "isolated"
                with self._lock:
                    self.module_weights[name] = 0.0
                    self._normalize_weights()
                self.event_bus.publish(
                    "MODULE_HEALTH",
                    {
                        "module": name,
                        "status": "isolated",
                        "last_error": module.health.last_error,
                    },
                    source="AutonomyOrganism",
                )

    def self_code_module(
        self, module_name: str, *, apply: bool = True
    ) -> Dict[str, Any]:
        """Run the bounded coding workflow for one active module."""
        module = self.modules.get(module_name)
        if module is None:
            raise KeyError(f"unknown module: {module_name}")
        context = self._module_context(module_name)
        return self.self_coder.run_for_module(
            module,
            context=context,
            regime_parameters=self.market_detector.adaptive_parameters(
                self._last_regime
            ),
            apply=apply,
        )

    def repair_module(self, module_name: str, reason: str = "") -> Dict[str, Any]:
        """Attempt a safe in-memory repair and report the result."""
        module = self.modules.get(module_name)
        if module is None:
            raise KeyError(f"unknown module: {module_name}")
        result = self.self_coder.attempt_auto_repair(module, reason=reason)
        if result.get("repaired"):
            module.enabled = True
            module.health.status = "ok"
            with self._lock:
                self.module_weights[module_name] = max(
                    self.module_weights.get(module_name, 0.001), 0.001
                )
                self._normalize_weights()
        return result

    # --------------------------------------------------------------- status
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            modules = {name: module.to_dict() for name, module in self.modules.items()}
            weights = dict(self.module_weights)
        return {
            "running": self._running,
            "modules": modules,
            "weights": weights,
            "regime": self._last_regime.to_dict(),
            "sentinel": self.sentinel.get_status(),
            "shadow": self.shadow_manager.get_status(),
            "auto_promote_shadows": self.config.auto_promote_shadows,
            "monitor": self.monitor.get_status(),
            "audit": self.audit_trail.status(),
            "survival_audit": self.survival_audit.status(),
            "shadow_audit": self.shadow_audit.status(),
            "learning": self.learning_store.summary(),
            "self_coder": self.self_coder.get_status(),
            "last_cycle": _json_safe(self._last_cycle),
        }


_global_organism: Optional[Organism] = None
_global_lock = threading.Lock()


def get_organism() -> Organism:
    global _global_organism
    with _global_lock:
        if _global_organism is None:
            _global_organism = Organism()
            _global_organism.discover_and_wire()
        return _global_organism


def reset_organism() -> None:
    global _global_organism
    with _global_lock:
        if _global_organism is not None:
            _global_organism.stop()
        _global_organism = None


__all__ = [
    "BaseModule",
    "BaseTradingModule",
    "Event",
    "EventBus",
    "ModuleAutoDiscovery",
    "ModuleHealth",
    "ModuleResult",
    "MultiTimeframeSentinel",
    "Organism",
    "OrganismConfig",
    "get_event_bus",
    "get_organism",
    "get_registered_modules",
    "register_module",
    "reset_organism",
]
