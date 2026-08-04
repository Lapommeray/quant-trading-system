"""Gold-set stress validation for lessons and shadow candidates.

The gold set is a small, versioned JSONL collection of crash and volatility
traces.  It is a safety regression set, not a performance claim.  Deployments
should extend it with audited OHLCV data for their exact assets and venue.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    symbol: str
    date: str
    kind: str
    returns: tuple[float, ...]
    source: str = "curated stress fixture"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldCase":
        values = tuple(float(value) for value in data.get("returns", []))
        return cls(
            case_id=str(data.get("case_id", data.get("date", "unknown"))),
            symbol=str(data.get("symbol", "UNKNOWN")),
            date=str(data.get("date", "unknown")),
            kind=str(data.get("kind", "volatile")),
            returns=values,
            source=str(data.get("source", "curated stress fixture")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "symbol": self.symbol,
            "date": self.date,
            "kind": self.kind,
            "returns": list(self.returns),
            "source": self.source,
        }


class GoldSet:
    def __init__(self, cases: Iterable[GoldCase], path: Optional[Path] = None) -> None:
        self.cases = list(cases)
        self.path = path

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "GoldSet":
        target = (
            Path(path)
            if path
            else Path(__file__).resolve().parents[1] / "data" / "gold_set.jsonl"
        )
        cases: List[GoldCase] = []
        try:
            with target.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(item, dict):
                        case = GoldCase.from_dict(item)
                        if case.returns:
                            cases.append(case)
        except OSError:
            pass
        return cls(cases, target)

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "cases": len(self.cases),
            "crashes": sum(case.kind == "crash" for case in self.cases),
            "volatile": sum(case.kind == "volatile" for case in self.cases),
            "symbols": sorted({case.symbol for case in self.cases}),
        }


@dataclass(frozen=True)
class GoldStressPolicy:
    max_drawdown: float = 0.25
    minimum_equity: float = 0.50
    position_fraction: float = 0.02
    minimum_samples: int = 5


@dataclass
class GoldStressReport:
    passed: bool
    dangerous_overfitting: bool
    active: Dict[str, Any]
    candidate: Dict[str, Any]
    failures: List[str] = field(default_factory=list)
    cases: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "dangerous_overfitting": self.dangerous_overfitting,
            "active": dict(self.active),
            "candidate": dict(self.candidate),
            "failures": list(self.failures),
            "cases": self.cases,
            "timestamp": self.timestamp,
        }


class GoldSetStressTester:
    """Run a conservative, deterministic paper stress simulation."""

    def __init__(
        self,
        gold_set: Optional[GoldSet] = None,
        policy: Optional[GoldStressPolicy] = None,
    ) -> None:
        self.gold_set = gold_set or GoldSet.load()
        self.policy = policy or GoldStressPolicy()

    def simulate(self, module: Any) -> Dict[str, Any]:
        equity = 1.0
        peak = equity
        max_drawdown = 0.0
        pnl_series: List[float] = []
        trades = 0
        failures: List[str] = []

        for case in self.gold_set.cases:
            prices = [100.0]
            case_equity = equity
            case_peak = case_equity
            for value in case.returns:
                prices.append(prices[-1] * math.exp(value))
            for index in range(len(case.returns)):
                history = {"close": prices[: index + 1]}
                try:
                    result = module.generate_signal(case.symbol, history)
                    signal = getattr(result, "signal", None)
                    confidence = getattr(result, "confidence", 0.0)
                    if isinstance(result, dict):
                        signal = result.get("signal", result.get("direction"))
                        confidence = result.get("confidence", 0.0)
                    signal = str(signal).upper() if signal is not None else "NEUTRAL"
                    confidence = float(confidence or 0.0)
                except Exception as exc:
                    failures.append(f"{case.case_id}: {exc}")
                    signal, confidence = "NEUTRAL", 0.0

                if signal not in {"BUY", "SELL"} or confidence <= 0:
                    pnl = 0.0
                else:
                    direction = 1.0 if signal == "BUY" else -1.0
                    pnl = (
                        direction
                        * case.returns[index]
                        * self.policy.position_fraction
                        * min(confidence, 1.0)
                    )
                    trades += 1
                case_equity *= max(0.0, 1.0 + pnl)
                pnl_series.append(pnl)
                case_peak = max(case_peak, case_equity)
                peak = max(peak, case_equity)
                max_drawdown = max(
                    max_drawdown, (case_peak - case_equity) / max(case_peak, 1e-12)
                )
            equity = case_equity

        average = statistics.fmean(pnl_series) if pnl_series else 0.0
        deviation = statistics.pstdev(pnl_series) if len(pnl_series) > 1 else 0.0
        sharpe = (
            average / max(deviation, 1e-12) * math.sqrt(len(pnl_series))
            if pnl_series
            else 0.0
        )
        return {
            "equity": equity,
            "total_return": equity - 1.0,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "trades": trades,
            "failures": failures,
            "blown_up": equity < self.policy.minimum_equity or equity <= 0,
        }

    @staticmethod
    def _clone(module: Any) -> Any:
        try:
            cloned = copy.deepcopy(module)
            cloned.event_bus = None
            return cloned
        except Exception:
            config = copy.deepcopy(getattr(module, "config", {}))
            try:
                return type(module)(config=config, event_bus=None)
            except Exception:
                cloned = copy.copy(module)
                cloned.config = config
                cloned.event_bus = None
                return cloned

    def validate_candidate(self, active: Any, candidate: Any) -> GoldStressReport:
        active_metrics = self.simulate(self._clone(active))
        candidate_metrics = self.simulate(self._clone(candidate))
        failures: List[str] = []
        if candidate_metrics["blown_up"]:
            failures.append("candidate equity breached gold-set minimum")
        if candidate_metrics["max_drawdown"] > self.policy.max_drawdown:
            failures.append("candidate drawdown exceeded gold-set limit")
        failures.extend(f"candidate: {item}" for item in candidate_metrics["failures"])
        # A candidate must not degrade stress behavior materially, even if it
        # is profitable in the current market.
        if candidate_metrics["max_drawdown"] > active_metrics["max_drawdown"] + 0.02:
            failures.append("candidate drawdown degraded versus active")
        dangerous = bool(failures)
        return GoldStressReport(
            passed=not dangerous,
            dangerous_overfitting=dangerous,
            active=active_metrics,
            candidate=candidate_metrics,
            failures=failures,
            cases=len(self.gold_set.cases),
        )


__all__ = [
    "GoldCase",
    "GoldSet",
    "GoldSetStressTester",
    "GoldStressPolicy",
    "GoldStressReport",
]
