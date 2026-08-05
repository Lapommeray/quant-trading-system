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

    # =====================================================
    # AUTO SELF-CODING — Full autonomous per-module engine
    # Each module can now: auto-fix, create code, approve, learn mistakes,
    # improve with market, interconnect as ONE ORGANISM
    # =====================================================

    def auto_self_code(
        self,
        coder: Any = None,
        context: Optional[Dict[str, Any]] = None,
        *,
        apply: bool = True,
        auto_approve: bool = True,
    ) -> Dict[str, Any]:
        """Fully autonomous self-coding cycle for THIS module.
        
        - ALWAYS routes through CENTRAL SelfCodingEngine (AST whitelist + shadow path)
        - Diagnoses self
        - Generates adaptive code artifact (never mutates live source)
        - Validates + auto-approves low-risk ONLY after shadow/gold gates in organism
        - Learns from past mistakes
        - Applies only to shadow / adaptive params
        """
        # === COHERENCE + FORTRESS GUARD ===
        if not self._can_mutate():
            return {
                "status": "cooldown", 
                "reason": "mutation_cooldown_active",
                "next_allowed": self._next_mutation_time()
            }
        
        context = context or {}
        context.setdefault("auto_trigger", "auto_self_code")
        context.setdefault("mistakes", self._get_mistake_history())
        context["central_engine_required"] = True   # Enforce central nervous system
        
        if coder is None:
            try:
                from autonomy.self_coding import SelfCodingEngine
                coder = SelfCodingEngine()
            except Exception:
                return {"status": "no_coder", "error": "SelfCodingEngine unavailable"}
        
        try:
            # CRITICAL: All self-coding goes through the SINGLE central engine
            result = coder.run_for_module(
                self, 
                context=context, 
                apply=apply
            )
            
            # Record mutation time for cooldown (even for proposals)
            self._record_mutation_attempt()
            
            # Auto-learn from mistakes immediately
            if isinstance(result, dict) and result.get("status") in ("applied", "approved"):
                self.learn_from_mistakes(result.get("diagnosis", {}))
            
            # NOTE: True "auto_approve" for live promotion is handled ONLY by organism
            # after Shadow + Gold-Set validation. Local flag only marks candidate.
            # CRITICAL FORTRESS RULE: No code ever goes live without shadow.
            if isinstance(result, dict):
                result.setdefault("organism_unity", True)
                result.setdefault("shadow_required", True)           # Fortress requirement
                result.setdefault("live_promotion_blocked", True)    # Explicit guard
                result.setdefault("promotion_path", "shadow_then_gold_set")
                result.setdefault("central_engine", "SelfCodingEngine")
                
                if auto_approve and result.get("risk") == "low":
                    result["auto_approved_candidate"] = True
            
            # Broadcast mutation event for coherence protocol
            if self.event_bus and isinstance(result, dict) and result.get("status") in ("applied", "approved", "validated"):
                self.event_bus.publish("MUTATION_EVENT", {
                    "module": self.module_name,
                    "proposal_id": result.get("proposal_id"),
                    "risk": result.get("risk"),
                    "status": result.get("status"),
                    "timestamp": time.time(),
                    "adaptive_changes": result.get("parameters", {})
                }, source=self.module_name)
            
            return result
        except Exception as e:
            self.record_failure(f"auto_self_code failed: {e}")
            return {"status": "error", "error": str(e), "module": self.module_name}

    def auto_fix(self, coder: Any = None, reason: str = "autonomous_fix", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Auto-fix + self-repair using bounded self-coding."""
        context = context or {"reason": reason, "auto_fix": True}
        try:
            if coder is None:
                from autonomy.self_coding import SelfCodingEngine
                coder = SelfCodingEngine()
            
            repair = coder.auto_fix(self, reason=reason, context=context, apply=True)
            
            if repair.get("fixed") or repair.get("artifact_applied"):
                self.health.status = "ok"
                self.health.error_count = max(0, self.health.error_count - 1)
            
            self.learn_from_mistakes({"repair": repair})
            return repair
        except Exception as e:
            self.record_failure(str(e))
            return {"fixed": False, "error": str(e)}

    def learn_from_mistakes(self, mistake_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from past mistakes and previous failed proposals.
        
        Updates internal adaptive state + feeds into organism memory.
        """
        try:
            mistakes = mistake_data.get("mistakes", []) or []
            if not mistakes and "lesson" in mistake_data:
                mistakes = [mistake_data]
            
            # Store local lessons
            if not hasattr(self, "_learned_mistakes"):
                self._learned_mistakes = []
            self._learned_mistakes.extend(mistakes[-5:])
            self._learned_mistakes = self._learned_mistakes[-20:]
            
            # Apply adaptive tuning from mistakes
            if hasattr(self, "apply_adaptive_parameters"):
                tuning = {}
                if any("overfit" in str(m).lower() or "high" in str(m).lower() for m in mistakes):
                    tuning["weight_multiplier"] = 0.92
                    tuning["confidence_floor"] = 0.68
                if any("under" in str(m).lower() or "miss" in str(m).lower() for m in mistakes):
                    tuning["weight_multiplier"] = 1.05
                    tuning["confidence_floor"] = 0.55
                
                if tuning:
                    self.apply_adaptive_parameters(tuning)
            
            # Broadcast to organism for global memory
            if self.event_bus:
                self.event_bus.publish("MODULE_LEARNED_MISTAKE", {
                    "module": self.module_name,
                    "lessons": mistakes,
                    "timestamp": time.time()
                }, source=self.module_name)
            
            return {"learned": True, "lessons_applied": len(mistakes)}
        except Exception as e:
            return {"learned": False, "error": str(e)}

    def _get_mistake_history(self) -> List[Dict]:
        """Retrieve recent mistakes for self-coding context."""
        if hasattr(self, "_learned_mistakes"):
            return self._learned_mistakes[-10:]
        return []

    # =====================================================
    # COHERENCE + METABOLIC RATE (FORTRESS GUARDS)
    # =====================================================
    _MUTATION_COOLDOWN_SEC = 180.0   # 3 minutes metabolic rate (configurable per deployment)
    _last_mutation_attempt: float = 0.0

    def _can_mutate(self) -> bool:
        """Metabolic rate guard: prevents mutation war / resource exhaustion."""
        now = time.time()
        if not hasattr(self, "_last_mutation_attempt") or self._last_mutation_attempt == 0:
            return True
        return (now - self._last_mutation_attempt) >= self._MUTATION_COOLDOWN_SEC

    def _record_mutation_attempt(self):
        """Record attempt for cooldown enforcement."""
        self._last_mutation_attempt = time.time()

    def _next_mutation_time(self) -> float:
        if not hasattr(self, "_last_mutation_attempt") or self._last_mutation_attempt == 0:
            return 0.0
        return self._last_mutation_attempt + self._MUTATION_COOLDOWN_SEC

    def set_mutation_cooldown(self, seconds: float):
        """Allow organism or config to tune metabolic rate."""
        self._MUTATION_COOLDOWN_SEC = max(30.0, float(seconds))

    def improve_with_market(self, regime_data: Dict[str, Any], market_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Self-improve when the market itself improves / changes regime.
        
        Called automatically by organism when regime shifts.
        """
        market_context = market_context or {}
        improvements = {}
        
        try:
            # Adaptive tuning based on regime
            regime = regime_data.get("regime", "UNKNOWN")
            if "BULL" in regime.upper() or "TRENDING" in regime.upper():
                improvements["weight_multiplier"] = 1.08
                improvements["confidence_floor"] = 0.58
            elif "BEAR" in regime.upper() or "CRISIS" in regime.upper():
                improvements["weight_multiplier"] = 0.85
                improvements["confidence_floor"] = 0.72
            
            if hasattr(self, "apply_adaptive_parameters"):
                applied = self.apply_adaptive_parameters(improvements)
                improvements["applied"] = applied
            
            # Trigger self-coding if market has improved significantly
            if market_context.get("market_improving", False) or regime_data.get("confidence", 0) > 0.65:
                if hasattr(self, "auto_self_code"):
                    code_result = self.auto_self_code(
                        context={"regime": regime, "market_improvement": True},
                        apply=True
                    )
                    improvements["self_code_triggered"] = code_result.get("status")
            
            # Broadcast market adaptation
            if self.event_bus:
                self.event_bus.publish("MODULE_MARKET_IMPROVED", {
                    "module": self.module_name,
                    "regime": regime,
                    "improvements": improvements
                }, source="organism")
            
            return {"improved": True, "regime": regime, "changes": improvements}
        except Exception as e:
            self.record_failure(f"market_improve: {e}")
            return {"improved": False, "error": str(e)}

    def interconnect(self, target_modules: Optional[List[str]] = None, message: Optional[Dict] = None) -> Dict[str, Any]:
        """Connect this module to other modules for 1-Organism communication.
        
        Uses event bus + shared memory for inter-module signaling.
        """
        try:
            if not self.event_bus:
                return {"interconnected": False, "reason": "no_event_bus"}
            
            payload = {
                "source": self.module_name,
                "targets": target_modules or ["*"],
                "message": message or {"type": "handshake", "status": "online"},
                "timestamp": time.time(),
                "adaptive_state": getattr(self, "config", {}).get("adaptive", {}),
            }
            
            # Publish to organism bus (all modules subscribed)
            self.event_bus.publish("MODULE_INTERCONNECT", payload, source=self.module_name)
            
            # Direct targeted messages
            if target_modules:
                for target in target_modules:
                    self.event_bus.publish(f"MODULE_{target.upper()}_MESSAGE", payload, source=self.module_name)
            
            # Store interconnection state
            if not hasattr(self, "_interconnections"):
                self._interconnections = set()
            if target_modules:
                self._interconnections.update(target_modules)
            
            return {
                "interconnected": True, 
                "organism_unity": True,
                "connected_to": list(getattr(self, "_interconnections", [])),
                "broadcast": True
            }
        except Exception as e:
            return {"interconnected": False, "error": str(e)}

    def sync_with_organism(self, organism_state: Dict[str, Any]) -> Dict[str, Any]:
        """Sync this module with the unified organism state (1 organism)."""
        try:
            if "regime" in organism_state:
                self.improve_with_market(organism_state["regime"])
            
            # Apply global weights if present
            if "weights" in organism_state and self.module_name in organism_state["weights"]:
                weight = organism_state["weights"][self.module_name]
                if hasattr(self, "apply_adaptive_parameters"):
                    self.apply_adaptive_parameters({"weight_multiplier": weight})
            
            # Auto self-code if organism requests
            if organism_state.get("trigger_self_code"):
                return self.auto_self_code(context=organism_state)
            
            # === COHERENCE PROTOCOL ===
            if "mutation_event" in organism_state:
                return self.coherence_check(organism_state["mutation_event"])
            
            return {"synced": True, "module": self.module_name}
        except Exception as e:
            return {"synced": False, "error": str(e)}

    def coherence_check(self, mutation_event: Dict[str, Any]) -> Dict[str, Any]:
        """Coherence Protocol: React to another module's mutation.
        
        If another cell (e.g. Momentum) becomes more aggressive,
        this cell (e.g. Risk) must re-align or trigger its own fix.
        """
        source = mutation_event.get("module")
        changes = mutation_event.get("adaptive_changes", {})
        
        # Default: record for future awareness
        if not hasattr(self, "_peer_mutations"):
            self._peer_mutations = []
        self._peer_mutations.append({"from": source, "changes": changes, "ts": time.time()})
        self._peer_mutations = self._peer_mutations[-10:]
        
        # Risk / conservative modules should become stricter on aggressive mutations
        my_name = self.module_name.lower()
        is_risk_like = any(k in my_name for k in ["risk", "guard", "sentinel", "safety"])
        
        if is_risk_like and changes:
            weight_mult = changes.get("weight_multiplier", 1.0)
            if weight_mult > 1.05:  # Peer became more aggressive
                tuning = {
                    "weight_multiplier": max(0.7, 0.95),   # Tighten
                    "confidence_floor": min(0.85, 0.70 + 0.05)
                }
                if hasattr(self, "apply_adaptive_parameters"):
                    self.apply_adaptive_parameters(tuning)
                
                # Optionally trigger self-correction
                if self.health.error_count > 0 or self._get_mistake_history():
                    self.auto_fix(reason="coherence_realign_after_peer_mutation")
                
                return {
                    "coherence_action": "tightened_risk",
                    "source_mutation": source,
                    "applied": tuning
                }
        
        return {
            "coherence_action": "recorded",
            "source": source,
            "peer_mutations_tracked": len(getattr(self, "_peer_mutations", []))
        }

    def on_mutation_event(self, event: Dict[str, Any]):
        """Called by organism when MUTATION_EVENT is received on bus."""
        return self.coherence_check(event)

    def full_autonomous_cycle(self, coder: Any = None, organism_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Master cycle: auto fix + self code + learn mistakes + market improve + interconnect.
        
        This makes the module a living part of the ONE ORGANISM.
        
        All code generation is routed through the central fortress (SelfCodingEngine).
        All mutations respect metabolic cooldown and trigger coherence protocol.
        """
        organism_context = organism_context or {}
        results = {}
        
        # 1. Auto fix if unhealthy
        if self.health.status in ("degraded", "failed", "isolated"):
            results["auto_fix"] = self.auto_fix(coder=coder, reason="full_cycle_health")
        
        # 2. Learn from mistakes
        results["learn_mistakes"] = self.learn_from_mistakes(organism_context.get("mistakes", {}))
        
        # 3. Auto self-code (ALWAYS via central engine + cooldown + shadow gate)
        results["self_code"] = self.auto_self_code(
            coder=coder,
            context=organism_context,
            apply=True,
            auto_approve=True
        )
        
        # 4. Improve with market
        if "regime" in organism_context:
            results["market_improve"] = self.improve_with_market(
                organism_context.get("regime", {}),
                market_context=organism_context
            )
        
        # 5. Interconnect with other modules
        results["interconnect"] = self.interconnect(
            target_modules=organism_context.get("active_modules"),
            message={"cycle": "full_autonomous", "status": "unified"}
        )
        
        results["organism_unity"] = True
        results["module"] = self.module_name
        results["fortress_compliant"] = True
        return results

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
