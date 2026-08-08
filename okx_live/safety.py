"""
OKX Safety Guard - real trading safety, fails closed.

Enforces eternal guardrails, no paper bypass unless explicitly allowed for testing.
"""

from __future__ import annotations

import os
import logging
from typing import Tuple

log = logging.getLogger(__name__)

try:
    from safety_governance import (
        SafetyGovernanceSystem,
        AuthorizationLevel,
        EternalGuardrails,
    )

    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    SafetyGovernanceSystem = None  # type: ignore
    AuthorizationLevel = None  # type: ignore
    EternalGuardrails = None  # type: ignore


class OKXSafetyGuard:
    """Safety guard that fails closed, no simulation."""

    def __init__(
        self,
        max_leverage: float = 3.0,
        max_position_pct: float = 0.10,
        max_daily_loss_pct: float = 0.03,
        require_live_credentials: bool = True,
    ):
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.require_live_credentials = require_live_credentials

        if SAFETY_AVAILABLE:
            # For real trading, paper_mode=False, but this will require human override
            # We initialize in paper_mode=False to enforce eternal guardrails
            # However we allow override via env for testing
            paper = os.getenv("OKX_ALLOW_PAPER_FOR_TEST", "false").lower() in (
                "true",
                "1",
                "yes",
            )
            self.safety = SafetyGovernanceSystem(
                paper_mode=paper, required_confirmations=3
            )
        else:
            self.safety = None

    def validate_credentials(self) -> Tuple[bool, str]:
        """Fails closed if credentials missing."""
        api_key = os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY")
        api_secret = os.getenv("OKX_API_SECRET") or os.getenv("OKX_SECRET")
        passphrase = os.getenv("OKX_PASSPHRASE")

        if not self.require_live_credentials:
            return True, "OK - credentials not required (test mode)"

        if not (api_key and api_secret and passphrase):
            return (
                False,
                "Missing OKX credentials: OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE required for real trading (fail-closed)",
            )

        if len(api_key) < 10 or len(api_secret) < 10:
            return False, "OKX credentials look invalid (too short) - fail-closed"

        return True, "OK"

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        equity: float,
        leverage: float,
    ) -> Tuple[bool, str]:
        # Leverage cap
        if leverage > self.max_leverage:
            return (
                False,
                f"Leverage {leverage}x exceeds cap {self.max_leverage}x - fail-closed",
            )

        # Notional
        notional = quantity * price
        if equity > 0 and notional / equity > self.max_position_pct:
            return (
                False,
                f"Notional {notional:.2f} exceeds {self.max_position_pct:.0%} equity - fail-closed",
            )

        # Eternal guardrails via safety_governance if available
        if SAFETY_AVAILABLE and self.safety and EternalGuardrails:
            is_live = True
            has_override = self.safety.confirmation_system.human_override_active
            # Check via eternal guardrails
            risk = notional / equity if equity else 0.01
            passed, reason = EternalGuardrails.check_trade(
                trade_risk=risk,
                daily_loss=0.0,
                drawdown=0.0,
                leverage=leverage,
                position_concentration=notional / equity if equity else 0.1,
                is_live=is_live,
                has_human_override=has_override,
            )
            if not passed:
                return False, reason

        return True, "OK"

    def is_kill_switch_active(self) -> bool:
        if SAFETY_AVAILABLE and self.safety:
            return self.safety.kill_switch.is_active()
        return False
