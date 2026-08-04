"""
Wrapper re-exporting execution.okx_engine for package API.

This module provides a package-local import path:
    from quant_trading_system.execution.okx_executor import OKXExecutor
and integrates with quant_trading_system.config.Settings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from quant_trading_system.config import settings

# Re-use core implementation - import lazily to avoid circular issues
try:
    from execution.okx_engine import OKXEngine, OKXOrderRequest, OKXOrderResult, OrderSide, OrderType
except ImportError:
    # fallback when executed as package without top-level execution folder in PYTHONPATH
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from execution.okx_engine import OKXEngine, OKXOrderRequest, OKXOrderResult, OrderSide, OrderType


@dataclass
class OKXExecutorConfig:
    max_leverage: float = settings.okx_max_leverage
    max_position_pct: float = settings.okx_max_position_pct
    max_daily_loss_pct: float = settings.okx_max_daily_loss_pct
    paper_mode: bool = settings.okx_paper_mode


class OKXExecutor(OKXEngine):
    """Package-level executor inheriting full OKXEngine logic."""

    def __init__(self, config: Optional[OKXExecutorConfig] = None, event_bus: Optional[Any] = None):
        cfg = config or OKXExecutorConfig()
        super().__init__(
            paper_mode=cfg.paper_mode,
            event_bus=event_bus,
            max_leverage=cfg.max_leverage,
            max_position_pct=cfg.max_position_pct,
            max_daily_loss_pct=cfg.max_daily_loss_pct,
        )

    def execute_from_organism_signal(self, consensus: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convenience for organism consensus dict."""
        result = self.place_order_from_signal(consensus)
        return result.to_dict() if result else None


__all__ = ["OKXExecutor", "OKXOrderRequest", "OKXOrderResult", "OrderSide", "OrderType"]
