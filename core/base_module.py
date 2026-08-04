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


def register_module(cls: Type["BaseTradingModule"] | None = None, *, name: str | None = None):
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


class BaseTradingModule(abc.ABC):
    """Abstract base for all organism modules.

    Subclasses should implement `analyze` or `generate_signal`.
    Optional: `self_improve` hook called by organism.
    """

    module_name: str = "base"
    category: str = "general"
    version: str = "1.0.0"
    dependencies: List[str] = []  # other module names this depends on

    def __init__(self, config: Optional[Dict[str, Any]] = None, event_bus: Any | None = None):
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

    def generate_signal(self, symbol: str, history_data: Dict[str, Any]) -> ModuleResult:
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

    def self_improve(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Self-improvement hook. Return dict with improvements made."""
        return {"module": self.module_name, "improved": False}

    def heartbeat(self):
        self.health.last_heartbeat = time.time()

    def record_success(self, latency_ms: float):
        self.health.success_count += 1
        self.health.status = "ok"
        # EWMA latency
        if self.health.avg_latency_ms == 0:
            self.health.avg_latency_ms = latency_ms
        else:
            self.health.avg_latency_ms = 0.9 * self.health.avg_latency_ms + 0.1 * latency_ms
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
            },
            "dependencies": self.dependencies,
        }
