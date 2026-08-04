"""Composed autonomous safety core for direct integrations."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from autonomy.guardrails import AutonomousGuardrails
from autonomy.monitor import AutonomousMonitor
from autonomy.protected_executor import ProtectedTradeExecutor
from autonomy.self_coding import SafeCodeValidator


class AutonomousCore:
    """Compose validation, protected execution and monitoring layers.

    The canonical `autonomy.Organism` remains responsible for shadow module
    promotion. This facade is for integrations that previously called a
    monolithic autonomous core directly.
    """

    def __init__(
        self, okx_client: Any, *, artifact_dir: Optional[str | Path] = None
    ) -> None:
        self.guardrails = AutonomousGuardrails()
        self.validator = SafeCodeValidator()
        self.trade_executor = ProtectedTradeExecutor(okx_client, self.guardrails)
        self.monitor = AutonomousMonitor(error_callback=self._on_module_error)
        self.artifact_dir = Path(artifact_dir or "strategies/evolved/pending")
        self._disabled_modules: set[str] = set()

    def process_generated_code(self, code: str, module_name: str) -> Dict[str, Any]:
        report = self.validator.validate(code, filename=f"{module_name}.py")
        if not report.passed:
            self.monitor.log_error(
                "code validation failed: " + "; ".join(report.errors), module_name
            )
            return {"approved": False, "reason": report.errors, "penalty": True}
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", module_name)
        destination = self.artifact_dir / f"{safe_name}_{int(time.time())}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(code, encoding="utf-8")
        return {
            "approved": True,
            "risk": "low",
            "path": str(destination),
            "deployment": "shadow_required",
            "live_source_modified": False,
        }

    def execute_signal(
        self, signal: Dict[str, Any], source: str = "autonomous"
    ) -> Dict[str, Any]:
        return self.trade_executor.execute_trade(signal, source)

    def start(self) -> None:
        self.monitor.start()

    def shutdown(self) -> None:
        self.monitor.stop()

    def _on_module_error(self, module: str, error: str) -> None:
        self._disabled_modules.add(module)


__all__ = ["AutonomousCore"]
