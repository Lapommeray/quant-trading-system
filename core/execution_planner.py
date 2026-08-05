"""
ExecutionPlanner - maker-first execution routing that moves WITH the market
maker, not with the late retail crowd.

Institutional logic implemented here:

1. NEVER take the spread on a plain technical signal.  Retail enters on
   confirmation (late).  We post a limit *inside* the spread at
   ``mid + 0.1 * spread`` when order flow is favorable, capturing queue
   position instead of paying the maker-taker spread.

2. ESCALATE to immediate taker ONLY when the whale/flow stack confirms:
   whale confidence >= 0.85 AND OFI z-score >= 2.0 AND CVD agrees.  That is
   the moment the market maker is being run over - waiting 200ms more would
   mean buying after the move, i.e. with retail.

3. WAIT/ABORT on wide spreads, stale feed, or post-recheck OFI decay.
   A maker order that is no longer backed by flow gets re-priced or
   cancelled - never left to be picked off.

The planner is PURE (no I/O, no network).  It only proposes; the executor
and the safety guardrails decide.  Every branch is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Routing(str, Enum):
    MAKER = "maker"      # post limit inside spread, queue-aware
    TAKER = "taker"      # immediate market/aggressive limit (whale/OFI stack)
    WAIT = "wait"        # hold 200ms and recheck flow before deciding
    ABORT = "abort"      # do not trade


@dataclass(frozen=True)
class ExecutionPlannerConfig:
    min_confidence: float = 0.65
    # Limit price offset as a fraction of the current spread (0.1 = mid + 0.1*spread).
    inside_spread_frac: float = 0.1
    # Escalation thresholds (whale stack).
    taker_whale_conf: float = 0.85
    taker_ofi_z: float = 2.0
    # Market quality guard: abort when spread exceeds this many bps.
    max_spread_bps: float = 30.0
    # Queue guard: post only when our side has >= this fraction of top-of-book size.
    min_queue_share: float = 0.25
    # After recheck: keep the maker order only while OFI stays above this.
    maker_keep_ofi_z: float = 0.5
    # After recheck: escalate to taker if OFI z is still >= this and urgency high.
    recheck_escalate_ofi_z: float = 1.5
    # If the maker order did not fill within max_wait_ms and flow died, abort.
    max_wait_ms: float = 250.0


@dataclass
class ExecutionPlan:
    routing: Routing
    side: str
    symbol: str
    reason: str
    limit_price: Optional[float] = None
    confidence: float = 0.0
    urgency: float = 0.0
    recheck_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing": self.routing.value,
            "side": self.side,
            "symbol": self.symbol,
            "reason": self.reason,
            "limit_price": self.limit_price,
            "confidence": round(self.confidence, 4),
            "urgency": round(self.urgency, 4),
            "recheck_ms": self.recheck_ms,
            "details": self.details,
        }


class ExecutionPlanner:
    def __init__(self, config: Optional[ExecutionPlannerConfig] = None):
        self.config = config or ExecutionPlannerConfig()

    # ------------------------------------------------------------------
    def plan(
        self,
        symbol: str,
        side: str,
        confidence: float,
        market: Dict[str, Any],
        signal: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Decide maker / taker / wait / abort from flow state.

        ``market`` keys used:
            bid, ask, ofi_z, cvd_align (bool), whale_conf (0..1),
            bid_size, ask_size, feed_fresh (bool), urgency (0..1, optional)
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            return ExecutionPlan(Routing.ABORT, side, symbol, "invalid_side", confidence=confidence)

        if not market.get("feed_fresh", True):
            return ExecutionPlan(Routing.ABORT, side, symbol, "stale_feed", confidence=confidence)

        bid = float(market.get("bid") or 0.0)
        ask = float(market.get("ask") or 0.0)
        if bid <= 0 or ask <= 0 or ask <= bid:
            return ExecutionPlan(Routing.ABORT, side, symbol, "no_quote", confidence=confidence)

        spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 1e4
        if spread_bps > self.config.max_spread_bps:
            return ExecutionPlan(
                Routing.ABORT, side, symbol, f"spread_too_wide_{spread_bps:.1f}bps",
                confidence=confidence, details={"spread_bps": spread_bps},
            )

        if confidence < self.config.min_confidence:
            return ExecutionPlan(
                Routing.ABORT, side, symbol, "confidence_below_floor",
                confidence=confidence, details={"spread_bps": spread_bps},
            )

        ofi_z = float(market.get("ofi_z") or 0.0)
        whale_conf = float(market.get("whale_conf") or 0.0)
        cvd_align = bool(market.get("cvd_align", True))
        urgency = float(market.get("urgency") or 0.0)

        # --- whale/flow stack -> immediate taker (the MM is being run over) ---
        if whale_conf >= self.config.taker_whale_conf and ofi_z >= self.config.taker_ofi_z and cvd_align:
            mid = (bid + ask) / 2.0
            return ExecutionPlan(
                Routing.TAKER, side, symbol, "whale_stack_escalation",
                limit_price=mid, confidence=confidence, urgency=1.0,
                details={"spread_bps": spread_bps, "ofi_z": ofi_z, "whale_conf": whale_conf},
            )

        # --- queue-aware maker post inside the spread ---
        bid_size = float(market.get("bid_size") or 0.0)
        ask_size = float(market.get("ask_size") or 0.0)
        top_total = bid_size + ask_size
        queue_share = (bid_size / top_total) if top_total > 0 else 0.0
        # For a buy we join the bid queue; queue share must be on our side.
        our_share = queue_share if side == "buy" else (1.0 - queue_share)

        mid = (bid + ask) / 2.0
        offset = self.config.inside_spread_frac * (ask - bid)
        limit_price = mid + offset if side == "buy" else mid - offset

        if our_share >= self.config.min_queue_share and ofi_z >= self.config.maker_keep_ofi_z:
            return ExecutionPlan(
                Routing.MAKER, side, symbol, "queue_aware_maker",
                limit_price=limit_price, confidence=confidence,
                urgency=0.3 + 0.4 * min(1.0, max(0.0, ofi_z) / 2.0),
                recheck_ms=200.0,
                details={
                    "spread_bps": spread_bps, "ofi_z": ofi_z,
                    "whale_conf": whale_conf, "queue_share": round(our_share, 3),
                },
            )

        # --- flow not confirming yet -> wait one recheck cycle, don't chase ---
        return ExecutionPlan(
            Routing.WAIT, side, symbol, "flow_not_confirming",
            limit_price=limit_price, confidence=confidence,
            urgency=urgency, recheck_ms=200.0,
            details={"spread_bps": spread_bps, "ofi_z": ofi_z, "queue_share": round(our_share, 3)},
        )

    # ------------------------------------------------------------------
    def finalize_after_recheck(
        self, plan: ExecutionPlan, recheck_market: Dict[str, Any], elapsed_ms: float
    ) -> ExecutionPlan:
        """Called ~200ms after a MAKER/WAIT plan.

        * OFI still strong + urgency -> escalate to taker (don't be late).
        * OFI alive -> keep posting maker.
        * OFI dead / timeout -> abort (never get picked off).
        """
        if plan.routing not in (Routing.MAKER, Routing.WAIT):
            return plan

        ofi_z = float(recheck_market.get("ofi_z") or 0.0)
        whale_conf = float(recheck_market.get("whale_conf") or 0.0)
        urgency = float(recheck_market.get("urgency") or plan.urgency)

        if (
            ofi_z >= self.config.recheck_escalate_ofi_z
            and (whale_conf >= 0.5 or urgency >= 0.6)
        ):
            return ExecutionPlan(
                Routing.TAKER, plan.side, plan.symbol,
                "recheck_escalation", limit_price=plan.limit_price,
                confidence=plan.confidence, urgency=1.0,
                details={**plan.details, "recheck_ofi_z": ofi_z},
            )

        if ofi_z >= self.config.maker_keep_ofi_z:
            return ExecutionPlan(
                Routing.MAKER, plan.side, plan.symbol,
                "maker_hold_after_recheck", limit_price=plan.limit_price,
                confidence=plan.confidence, urgency=urgency, recheck_ms=200.0,
                details={**plan.details, "recheck_ofi_z": ofi_z},
            )

        if elapsed_ms > self.config.max_wait_ms:
            return ExecutionPlan(
                Routing.ABORT, plan.side, plan.symbol,
                "flow_died_timeout", confidence=plan.confidence,
                details={**plan.details, "recheck_ofi_z": ofi_z},
            )

        return ExecutionPlan(
            Routing.WAIT, plan.side, plan.symbol,
            "flow_died_recheck_again", limit_price=plan.limit_price,
            confidence=plan.confidence, urgency=urgency, recheck_ms=200.0,
            details={**plan.details, "recheck_ofi_z": ofi_z},
        )


__all__ = ["ExecutionPlanner", "ExecutionPlannerConfig", "ExecutionPlan", "Routing"]
