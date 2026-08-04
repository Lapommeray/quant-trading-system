"""Mandatory guardrail wrapper for direct trade calls."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .audit import AuditTrail
from .guardrails import AutonomousGuardrails


class ProtectedTradeExecutor:
    """Ensure direct adapters pass the same immutable guardrails as the bus."""

    def __init__(
        self,
        okx_client: Any,
        guardrails: Optional[AutonomousGuardrails] = None,
        audit_path: Optional[str] = None,
    ) -> None:
        self.okx = okx_client
        self.guardrails = guardrails or AutonomousGuardrails()
        self.audit_trail = AuditTrail(audit_path or "audit_logs/trade_audit.jsonl")
        self.audit_log: list[Dict[str, Any]] = []

    def execute_trade(
        self, signal: Dict[str, Any], source: str = "autonomous"
    ) -> Dict[str, Any]:
        params = {
            "position_size_pct": signal.get(
                "position_size_pct", signal.get("position_size", 0.0)
            ),
            "leverage": signal.get("leverage", 1.0),
            "side": signal.get("side", signal.get("final_signal")),
            "symbol": signal.get("symbol", signal.get("asset")),
        }
        allowed, reason = self.guardrails.validate_trade(params)
        self._audit(params, allowed, reason, source)
        if not allowed:
            return {"executed": False, "reason": f"BLOCKED: {reason}", "order_id": None}
        try:
            if hasattr(self.okx, "place_order_from_signal"):
                result = self.okx.place_order_from_signal(signal)
            else:
                result = self.okx.place_order(**params)
            success = (
                bool(result.get("success"))
                if isinstance(result, dict)
                else bool(getattr(result, "success", False))
            )
            if not success:
                return {
                    "executed": False,
                    "reason": "exchange rejected order",
                    "order_id": None,
                }
            result_dict = (
                result.to_dict()
                if hasattr(result, "to_dict")
                else result if isinstance(result, dict) else {}
            )
            self.guardrails.record_trade_result(float(result_dict.get("pnl") or 0.0))
            return {
                "executed": True,
                "reason": "Success",
                "order_id": result_dict.get("order_id"),
            }
        except Exception as exc:
            self.guardrails.trigger_emergency_stop(f"execution error: {exc}")
            return {
                "executed": False,
                "reason": f"Execution error: {exc}",
                "order_id": None,
            }

    def _audit(
        self, params: Dict[str, Any], allowed: bool, reason: str, source: str
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "params": dict(params),
            "allowed": allowed,
            "reason": reason,
        }
        self.audit_log.append(entry)
        self.audit_trail.record("TRADE_ATTEMPT", entry, source=source)
        if len(self.audit_log) > 1_000:
            del self.audit_log[:-1_000]


__all__ = ["ProtectedTradeExecutor"]
