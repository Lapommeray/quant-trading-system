"""Shadow deployment and automatic promotion for module candidates."""

from __future__ import annotations

import copy
import math
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from .gold_set import GoldStressReport, GoldSetStressTester
from .market import extract_closes


@dataclass(frozen=True)
class ShadowPolicy:
    min_observations: int = 100
    min_outperformance: float = 0.05
    max_drawdown_delta: float = 0.01
    min_sharpe_delta: float = 0.0
    require_gold_set: bool = True


@dataclass
class ShadowMetrics:
    equity: float = 1.0
    peak_equity: float = 1.0
    max_drawdown: float = 0.0
    observations: int = 0
    trades: int = 0
    pnl: float = 0.0
    returns: list[float] = field(default_factory=list)

    def record(self, pnl: float, traded: bool) -> None:
        self.observations += 1
        if traded:
            self.trades += 1
        self.equity *= max(0.0, 1.0 + pnl)
        self.pnl = self.equity - 1.0
        self.peak_equity = max(self.peak_equity, self.equity)
        self.max_drawdown = max(
            self.max_drawdown,
            (self.peak_equity - self.equity) / max(self.peak_equity, 1e-12),
        )
        self.returns.append(pnl)
        if len(self.returns) > 2_000:
            del self.returns[:-2_000]

    @property
    def sharpe(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        mean = statistics.fmean(self.returns)
        deviation = statistics.pstdev(self.returns)
        return mean / max(deviation, 1e-12) * math.sqrt(len(self.returns))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "observations": self.observations,
            "trades": self.trades,
            "pnl": self.pnl,
            "sharpe": self.sharpe,
        }


class ShadowDeployment:
    """Run active and candidate clones on identical observations."""

    def __init__(
        self,
        *,
        module_name: str,
        proposal_id: str,
        active_module: Any,
        candidate_parameters: Optional[Mapping[str, Any]] = None,
        policy: Optional[ShadowPolicy] = None,
        gold_tester: Optional[GoldSetStressTester] = None,
        clock=time.time,
    ) -> None:
        self.shadow_id = uuid.uuid4().hex
        self.module_name = module_name
        self.proposal_id = proposal_id
        self.policy = policy or ShadowPolicy()
        self.gold_tester = gold_tester or GoldSetStressTester()
        self.clock = clock
        self.active_module = self._clone(active_module)
        self.candidate_module = self._clone(active_module)
        if candidate_parameters:
            apply_tuning = getattr(
                self.candidate_module, "apply_adaptive_parameters", None
            )
            if callable(apply_tuning):
                apply_tuning(dict(candidate_parameters))
        self.active_metrics = ShadowMetrics()
        self.candidate_metrics = ShadowMetrics()
        self._last_price: Dict[str, float] = {}
        self._last_signals: Dict[str, tuple[str, float]] = {}
        self._observations = 0
        self.status = "shadow"
        self.last_comparison: Dict[str, Any] = {}
        self.gold_report: Optional[GoldStressReport] = None

    def observe(self, symbol: str, history_data: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        active = self._signal(self.active_module, symbol, history_data)
        active_latency_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        candidate = self._signal(self.candidate_module, symbol, history_data)
        candidate_latency_ms = (time.perf_counter() - started) * 1000.0
        closes = extract_closes(history_data)
        price = closes[-1] if closes else None
        previous = self._last_price.get(symbol)
        market_return = (
            0.0
            if previous is None or previous <= 0 or price is None
            else price / previous - 1.0
        )
        self._last_price[symbol] = price or previous or 0.0

        active_pnl = self._pnl_for(active, market_return)
        candidate_pnl = self._pnl_for(candidate, market_return)
        self.active_metrics.record(active_pnl, active[0] in {"BUY", "SELL"})
        self.candidate_metrics.record(candidate_pnl, candidate[0] in {"BUY", "SELL"})
        self._last_signals[symbol] = candidate
        self._observations += 1
        comparison = self.compare()
        comparison.update(
            {
                "shadow_id": self.shadow_id,
                "symbol": symbol,
                "active_signal": active[0],
                "candidate_signal": candidate[0],
                "active_latency_ms": active_latency_ms,
                "candidate_latency_ms": candidate_latency_ms,
                "market_return": market_return,
                "status": self.status,
            }
        )
        self.last_comparison = comparison
        return comparison

    def compare(self) -> Dict[str, Any]:
        active = self.active_metrics
        candidate = self.candidate_metrics
        outperformance = candidate.pnl - active.pnl
        drawdown_ok = (
            candidate.max_drawdown
            <= active.max_drawdown + self.policy.max_drawdown_delta
        )
        sharpe_ok = candidate.sharpe >= active.sharpe + self.policy.min_sharpe_delta
        enough = self._observations >= self.policy.min_observations
        if enough and self.gold_report is None:
            self.gold_report = self.gold_tester.validate_candidate(
                self.active_module, self.candidate_module
            )
        gold_ok = self.gold_report is None or self.gold_report.passed
        eligible = (
            enough
            and outperformance >= self.policy.min_outperformance
            and drawdown_ok
            and sharpe_ok
            and gold_ok
        )
        if enough and self.gold_report is not None and not gold_ok:
            self.status = "rejected"
        elif eligible:
            self.status = "eligible_for_promotion"
        return {
            "observations": self._observations,
            "required_observations": self.policy.min_observations,
            "outperformance": outperformance,
            "drawdown_delta": candidate.max_drawdown - active.max_drawdown,
            "active": active.to_dict(),
            "candidate": candidate.to_dict(),
            "enough_observations": enough,
            "drawdown_ok": drawdown_ok,
            "sharpe_ok": sharpe_ok,
            "gold_ok": gold_ok,
            "eligible": eligible,
            "gold_report": self.gold_report.to_dict() if self.gold_report else None,
        }

    @staticmethod
    def _clone(module: Any) -> Any:
        """Clone module state without copying locks/event-bus internals."""
        try:
            cloned = copy.deepcopy(module)
            cloned.event_bus = None
            return cloned
        except Exception:
            module_type = type(module)
            config = copy.deepcopy(getattr(module, "config", {}))
            try:
                cloned = module_type(config=config, event_bus=None)
            except Exception:
                cloned = copy.copy(module)
                cloned.config = config
                cloned.event_bus = None
            return cloned

    @staticmethod
    def _signal(module: Any, symbol: str, history_data: Any) -> tuple[str, float]:
        try:
            result = module.generate_signal(symbol, history_data)
            if isinstance(result, Mapping):
                signal = result.get("signal", result.get("direction", "NEUTRAL"))
                confidence = result.get("confidence", 0.0)
            else:
                signal = getattr(result, "signal", "NEUTRAL")
                confidence = getattr(result, "confidence", 0.0)
            return str(signal).upper(), max(0.0, min(1.0, float(confidence or 0.0)))
        except Exception:
            return "NEUTRAL", 0.0

    @staticmethod
    def _pnl_for(signal: tuple[str, float], market_return: float) -> float:
        direction = 1.0 if signal[0] == "BUY" else -1.0 if signal[0] == "SELL" else 0.0
        return direction * market_return * signal[1] * 0.02

    def get_status(self) -> Dict[str, Any]:
        return {
            "shadow_id": self.shadow_id,
            "module_name": self.module_name,
            "proposal_id": self.proposal_id,
            "status": self.status,
            "comparison": self.last_comparison or self.compare(),
        }


class ShadowManager:
    """Own shadow candidates and invoke an explicit promotion callback."""

    def __init__(
        self,
        *,
        policy: Optional[ShadowPolicy] = None,
        gold_tester: Optional[GoldSetStressTester] = None,
        promote_callback: Optional[Callable[[ShadowDeployment], bool]] = None,
    ) -> None:
        self.policy = policy or ShadowPolicy()
        self.gold_tester = gold_tester or GoldSetStressTester()
        self.promote_callback = promote_callback
        self.deployments: Dict[str, ShadowDeployment] = {}

    def deploy(
        self,
        *,
        module_name: str,
        proposal_id: str,
        active_module: Any,
        candidate_parameters: Optional[Mapping[str, Any]] = None,
    ) -> ShadowDeployment:
        deployment = ShadowDeployment(
            module_name=module_name,
            proposal_id=proposal_id,
            active_module=active_module,
            candidate_parameters=candidate_parameters,
            policy=self.policy,
            gold_tester=self.gold_tester,
        )
        self.deployments[deployment.shadow_id] = deployment
        return deployment

    def observe(
        self, symbol: str, history_data: Any, *, promote: bool = True
    ) -> Dict[str, Any]:
        statuses: Dict[str, Any] = {}
        for shadow_id, deployment in list(self.deployments.items()):
            if deployment.status in {"promoted", "rejected"}:
                continue
            status = deployment.observe(symbol, history_data)
            if promote and status.get("eligible") and self.promote_callback:
                promoted = bool(self.promote_callback(deployment))
                if promoted:
                    deployment.status = "promoted"
                else:
                    deployment.status = "rejected"
            statuses[shadow_id] = deployment.get_status()
        return statuses

    def reject(self, shadow_id: str, reason: str = "") -> None:
        deployment = self.deployments.get(shadow_id)
        if deployment:
            deployment.status = "rejected"
            deployment.last_comparison["rejection_reason"] = reason

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_deployments": sum(
                deployment.status not in {"promoted", "rejected"}
                for deployment in self.deployments.values()
            ),
            "deployments": {
                shadow_id: deployment.get_status()
                for shadow_id, deployment in self.deployments.items()
            },
        }


__all__ = ["ShadowDeployment", "ShadowManager", "ShadowMetrics", "ShadowPolicy"]
