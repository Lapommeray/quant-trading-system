"""One-result module mesh orchestrator.

This is the safety-first glue layer that makes independent modules work as one
pipeline instead of producing disconnected opinions.  Each module receives the
same shared context plus outputs from all modules that ran before it.  The mesh
then reduces all module outputs into one final, auditable trade result.

Design rules:
* one input context in, one ``UnifiedTradeResult`` out;
* module failures, missing outputs, vetoes, stale/invalid context -> HOLD;
* no synthetic fallbacks are invented to hide a broken module;
* every module output and error is preserved in the final audit trail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional


class UnifiedAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ModuleResult:
    """Normalized output from one module in the mesh."""

    name: str
    ok: bool
    output: Dict[str, Any] = field(default_factory=dict)
    action: str = "HOLD"
    confidence: float = 0.0
    score: float = 0.0
    reason: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class UnifiedTradeResult:
    """The single final result consumed by execution/risk layers."""

    action: UnifiedAction
    confidence: float
    score: float
    reason: str
    module_results: Dict[str, ModuleResult]
    context: Dict[str, Any]
    generated_at: str
    tradable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": round(self.confidence, 6),
            "score": round(self.score, 6),
            "reason": self.reason,
            "tradable": self.tradable,
            "generated_at": self.generated_at,
            "module_results": {
                name: asdict(result) for name, result in self.module_results.items()
            },
            "context": self.context,
        }


@dataclass(frozen=True)
class ModuleMeshConfig:
    """Controls fail-closed consensus behavior."""

    confidence_threshold: float = 0.62
    min_required_modules: int = 1
    fail_closed: bool = True
    require_all_ok: bool = True
    disagreement_hold_threshold: float = 0.20
    max_context_age_seconds: float = 300.0


class ModuleMeshOrchestrator:
    """Run modules in order, pass shared context, and emit one result.

    Modules can be plain callables or objects exposing one of these methods:
    ``run``, ``process``, ``analyze``, ``predict``, ``decide``, ``evaluate``.
    A module receives the mutable context dictionary, which includes:

    * original market/state fields provided by caller;
    * ``module_outputs`` from earlier modules;
    * ``event_log`` communication messages.

    Returned module outputs may be dicts, dataclasses, booleans, or numeric
    scalar signals.  Dict outputs should prefer keys like ``action``,
    ``confidence``, ``score``, ``signal``, ``reason``.
    """

    _METHODS = ("run", "process", "analyze", "predict", "decide", "evaluate")

    def __init__(
        self,
        modules: Mapping[str, Any] | Iterable[tuple[str, Any]],
        *,
        config: Optional[ModuleMeshConfig] = None,
        weights: Optional[Mapping[str, float]] = None,
    ):
        self.modules = dict(modules)
        self.config = config or ModuleMeshConfig()
        self.weights = dict(weights or {})

    def run(self, market_context: Mapping[str, Any]) -> UnifiedTradeResult:
        context: Dict[str, Any] = dict(market_context)
        context.setdefault("module_outputs", {})
        context.setdefault("event_log", [])
        context["mesh_started_at"] = _utc_now_iso()

        if not self._context_is_fresh(context):
            return self._hold(
                "stale_or_future_context",
                context=context,
                results={},
            )

        results: Dict[str, ModuleResult] = {}
        for name, module in self.modules.items():
            result = self._run_one(name, module, context)
            results[name] = result
            context["module_outputs"][name] = result.output
            context["event_log"].append(
                {
                    "module": name,
                    "ok": result.ok,
                    "action": result.action,
                    "confidence": result.confidence,
                    "score": result.score,
                    "reason": result.reason,
                }
            )

        return self._aggregate(results, context)

    def _run_one(
        self, name: str, module: Any, context: MutableMapping[str, Any]
    ) -> ModuleResult:
        try:
            callable_obj = self._resolve_callable(module)
            raw = callable_obj(context)
            normalized = self._coerce_output(raw)
            action, confidence, score = self._extract_signal(normalized)
            return ModuleResult(
                name=name,
                ok=True,
                output=normalized,
                action=action,
                confidence=confidence,
                score=score,
                reason=str(normalized.get("reason", "ok")),
            )
        except Exception as exc:  # noqa: BLE001 - fail closed and preserve audit detail
            return ModuleResult(
                name=name,
                ok=False,
                output={},
                action="HOLD",
                confidence=0.0,
                score=0.0,
                reason="module_error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _resolve_callable(
        self, module: Any
    ) -> Callable[[MutableMapping[str, Any]], Any]:
        if callable(module):
            return module
        for method in self._METHODS:
            candidate = getattr(module, method, None)
            if callable(candidate):
                return candidate
        raise TypeError("module has no callable interface")

    def _coerce_output(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            return {"action": "HOLD", "confidence": 0.0, "reason": "empty_output"}
        if isinstance(raw, Mapping):
            return dict(raw)
        if is_dataclass(raw):
            return asdict(raw)
        if isinstance(raw, bool):
            return {
                "action": "BUY" if raw else "HOLD",
                "confidence": 1.0 if raw else 0.0,
            }
        if isinstance(raw, (int, float)):
            value = _clamp(float(raw), -1.0, 1.0)
            return {"score": value, "confidence": abs(value)}
        return {
            "value": raw,
            "action": "HOLD",
            "confidence": 0.0,
            "reason": "unrecognized_output",
        }

    def _extract_signal(self, output: Mapping[str, Any]) -> tuple[str, float, float]:
        action_raw = output.get(
            "action", output.get("side", output.get("direction", "HOLD"))
        )
        action = str(action_raw).upper()
        if action in {"LONG", "BULL", "BULLISH", "1"}:
            action = "BUY"
        elif action in {"SHORT", "BEAR", "BEARISH", "-1"}:
            action = "SELL"
        elif action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"

        confidence = _clamp(
            _as_float(output.get("confidence", output.get("probability", 0.0))),
            0.0,
            1.0,
        )

        if "score" in output:
            score = _clamp(_as_float(output["score"]), -1.0, 1.0)
        elif "signal" in output:
            signal = _as_float(output["signal"])
            score = _clamp(signal, -1.0, 1.0)
            if action == "HOLD" and score > 0:
                action = "BUY"
            elif action == "HOLD" and score < 0:
                action = "SELL"
        elif action == "BUY":
            score = confidence
        elif action == "SELL":
            score = -confidence
        else:
            score = 0.0

        if confidence == 0.0 and score != 0.0:
            confidence = abs(score)
        return action, confidence, score

    def _aggregate(
        self, results: Mapping[str, ModuleResult], context: Dict[str, Any]
    ) -> UnifiedTradeResult:
        if len(results) < self.config.min_required_modules:
            return self._hold("not_enough_modules", context=context, results=results)

        failures = [r for r in results.values() if not r.ok]
        if failures and self.config.fail_closed and self.config.require_all_ok:
            return self._hold(
                "module_failure_fail_closed: " + ", ".join(r.name for r in failures),
                context=context,
                results=results,
            )

        vetoes = [
            r.name
            for r in results.values()
            if _truthy(r.output.get("veto"))
            or _truthy(r.output.get("block_trade"))
            or str(r.output.get("risk", "")).lower() in {"high", "critical"}
            or str(r.output.get("status", "")).lower() == "error"
        ]
        if vetoes:
            return self._hold(
                "module_veto: " + ", ".join(vetoes), context=context, results=results
            )

        weighted_score = 0.0
        weight_sum = 0.0
        buy_weight = 0.0
        sell_weight = 0.0
        for name, result in results.items():
            if not result.ok:
                continue
            weight = max(0.0, float(self.weights.get(name, 1.0)))
            weighted_score += result.score * weight
            weight_sum += weight
            if result.score > 0:
                buy_weight += weight
            elif result.score < 0:
                sell_weight += weight

        if weight_sum <= 0:
            return self._hold(
                "no_positive_module_weight", context=context, results=results
            )

        score = _clamp(weighted_score / weight_sum, -1.0, 1.0)
        confidence = abs(score)
        disagreement = min(buy_weight, sell_weight) / weight_sum if weight_sum else 0.0
        if disagreement > self.config.disagreement_hold_threshold:
            return self._hold(
                f"module_disagreement_{disagreement:.2f}",
                context=context,
                results=results,
                score=score,
            )

        if confidence < self.config.confidence_threshold:
            return self._hold(
                "confidence_below_threshold",
                context=context,
                results=results,
                score=score,
            )

        action = UnifiedAction.BUY if score > 0 else UnifiedAction.SELL
        return UnifiedTradeResult(
            action=action,
            confidence=confidence,
            score=score,
            reason="module_consensus",
            module_results=dict(results),
            context=self._audit_context(context),
            generated_at=_utc_now_iso(),
            tradable=True,
        )

    def _hold(
        self,
        reason: str,
        *,
        context: Dict[str, Any],
        results: Mapping[str, ModuleResult],
        score: float = 0.0,
    ) -> UnifiedTradeResult:
        return UnifiedTradeResult(
            action=UnifiedAction.HOLD,
            confidence=abs(_clamp(score, -1.0, 1.0)),
            score=_clamp(score, -1.0, 1.0),
            reason=reason,
            module_results=dict(results),
            context=self._audit_context(context),
            generated_at=_utc_now_iso(),
            tradable=False,
        )

    def _context_is_fresh(self, context: Mapping[str, Any]) -> bool:
        ts = context.get("timestamp", context.get("ts"))
        if ts is None:
            return True
        try:
            if isinstance(ts, datetime):
                stamp = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts, str):
                stamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
            else:
                stamp = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return False
        age = (
            datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)
        ).total_seconds()
        return (
            -self.config.max_context_age_seconds
            <= age
            <= self.config.max_context_age_seconds
        )

    def _audit_context(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        # Preserve communication trail, but avoid duplicating bulky raw frames.
        return {
            key: value
            for key, value in context.items()
            if key not in {"raw_data", "dataframe", "df"}
        }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ModuleMeshConfig",
    "ModuleMeshOrchestrator",
    "ModuleResult",
    "UnifiedAction",
    "UnifiedTradeResult",
]
