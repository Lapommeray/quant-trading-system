"""
Base module for autonomous organism wiring.

All trading modules should inherit from BaseTradingModule to enable
automatic registration, event-bus communication, and self-improvement.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

log = logging.getLogger(__name__)

# Global registry populated by @register_module decorator & auto-discovery
_MODULE_REGISTRY: Dict[str, Type["BaseTradingModule"]] = {}


def register_module(
    cls: Type["BaseTradingModule"] | None = None, *, name: str | None = None
):
    """Decorator to register a module class in the global registry.

    Usage:
        @register_module
        class MyModule(BaseTradingModule): ...

        @register_module(name="custom_name")
        class MyOtherModule(BaseTradingModule): ...
    """

    def decorator(inner_cls: Type["BaseTradingModule"]):
        reg_name = name or getattr(inner_cls, "module_name", None) or inner_cls.__name__
        _MODULE_REGISTRY[reg_name] = inner_cls
        log.debug("Registered module %s -> %s", reg_name, inner_cls.__name__)
        return inner_cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)


def get_registered_modules() -> Dict[str, Type["BaseTradingModule"]]:
    return dict(_MODULE_REGISTRY)


def clear_registry():
    _MODULE_REGISTRY.clear()


@dataclass
class ModuleHealth:
    module_name: str
    status: str = "ok"  # ok, degraded, failed, isolated
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    last_error: Optional[str] = None


@dataclass
class ModuleResult:
    module_name: str
    signal: Optional[str] = None  # BUY, SELL, NEUTRAL
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "signal": self.signal,
            "confidence": self.confidence,
            "features": self.features,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class BaseTradingModule(abc.ABC):
    """Abstract base for all organism modules.

    Subclasses should implement `analyze` or `generate_signal`.
    Optional: `self_improve` hook called by organism.
    """

    module_name: str = "base"
    category: str = "general"
    version: str = "1.0.0"
    dependencies: List[str] = []  # other module names this depends on

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, event_bus: Any | None = None
    ):
        self.config = config or {}
        self.event_bus = event_bus
        self.health = ModuleHealth(module_name=self.module_name)
        self.enabled = True
        self._last_result: Optional[ModuleResult] = None

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize internal resources. Return False if cannot start."""
        raise NotImplementedError

    def analyze(self, market_data: Dict[str, Any]) -> ModuleResult:
        """Optional: override to produce analysis."""
        raise NotImplementedError("analyze() not implemented")

    def generate_signal(
        self, symbol: str, history_data: Dict[str, Any]
    ) -> ModuleResult:
        """Optional: higher-level signal generation wrapper."""
        # Default forwards to analyze if available
        data = {"symbol": symbol, "history": history_data}
        try:
            return self.analyze(data)
        except NotImplementedError:
            return ModuleResult(module_name=self.module_name)

    def on_event(self, event_type: str, payload: Dict[str, Any]):
        """Called by event bus for subscribed events. Override if needed."""
        pass

    def learn_from_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Consume a realized outcome without changing safety-critical code.

        Modules may override this hook to update a model or an allow-listed
        adaptive parameter.  The default implementation is intentionally a
        no-op so legacy modules can participate in the organism safely.
        """
        return {"module": self.module_name, "learned": False}

    def diagnose(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a small diagnostic record used by the self-coder."""
        context = context or {}
        return {
            "module": self.module_name,
            "status": self.health.status,
            "error_count": self.health.error_count,
            "last_error": self.health.last_error,
            "regime": context.get("regime", "unknown"),
        }

    def apply_adaptive_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply only non-execution tuning under the ``adaptive`` namespace.

        This prevents generated code from changing order size, leverage,
        credentials, kill-switches, or risk limits.  Subclasses can add their
        own validation but should keep this allow-list narrow.
        """
        allowed = {
            "confidence_floor",
            "weight_multiplier",
            "lookback",
            "cooldown_seconds",
            "volatility_multiplier",
            "regime_affinity_multiplier",
        }
        adaptive = self.config.setdefault("adaptive", {})
        applied = {}
        for key, value in (parameters or {}).items():
            if key not in allowed:
                continue
            if isinstance(value, (int, float)) and value == value:
                adaptive[key] = value
                applied[key] = value
        return applied

    def repair(self, reason: str = "") -> bool:
        """Reinitialize a failed module; source mutation is never attempted."""
        return bool(self.initialize())

    def self_improve(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Self-improvement hook. Return dict with improvements made."""
        return {"module": self.module_name, "improved": False}

    def self_code(
        self,
        coder: Any,
        context: Optional[Dict[str, Any]] = None,
        *,
        apply: bool = True,
    ):
        """Delegate bounded code generation to the shared coding engine."""
        if coder is None or not hasattr(coder, "run_for_module"):
            raise TypeError("coder must provide run_for_module(module, ...)")
        return coder.run_for_module(self, context=context or {}, apply=apply)

    def heartbeat(self):
        self.health.last_heartbeat = time.time()

    def record_success(self, latency_ms: float):
        self.health.success_count += 1
        self.health.status = "ok"
        # EWMA latency
        if self.health.avg_latency_ms == 0:
            self.health.avg_latency_ms = latency_ms
        else:
            self.health.avg_latency_ms = (
                0.9 * self.health.avg_latency_ms + 0.1 * latency_ms
            )
        self.heartbeat()

    def record_failure(self, error: str):
        self.health.error_count += 1
        self.health.last_error = error
        if self.health.error_count > 5 and self.health.success_count == 0:
            self.health.status = "failed"
        elif self.health.error_count > 3:
            self.health.status = "degraded"
        self.heartbeat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "category": self.category,
            "version": self.version,
            "enabled": self.enabled,
            "health": {
                "status": self.health.status,
                "error_count": self.health.error_count,
                "success_count": self.health.success_count,
                "avg_latency_ms": self.health.avg_latency_ms,
                "last_error": self.health.last_error,
                "last_heartbeat": self.health.last_heartbeat,
            },
            "dependencies": list(self.dependencies),
            "adaptive_parameters": (
                dict(self.config.get("adaptive", {}))
                if isinstance(self.config, dict)
                else {}
            ),
        }
