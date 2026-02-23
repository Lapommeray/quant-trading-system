"""
Risk Guard - Adaptive Risk & Exit Manager

Enforces position-level and session-level risk controls for intraday trading.

Controls:
- Per-trade risk cap (max % of equity at risk)
- Max drawdown limit (pause trading if exceeded)
- Volatility throttle (pause when vol exceeds threshold)
- Trade cooldown (hold after consecutive losses)
- Session loss limit (stop after N% session loss)
- Trailing stop logic

Pure Python/numpy. No external APIs.
"""

import time
import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger("RiskGuard")


class RiskGuard:
    """
    Adaptive risk management layer.

    Plugs into UnifiedAIIndicator to gate entries and suggest exits.
    Tracks equity, drawdown, and session performance.
    """

    def __init__(
        self,
        max_risk_pct: float = 1.0,
        max_dd_pct: float = 5.0,
        vol_throttle: float = 0.02,
        max_consecutive_losses: int = 3,
        session_loss_limit_pct: float = 2.0,
        cooldown_seconds: int = 300,
    ):
        self.max_risk_pct = max_risk_pct
        self.max_dd_pct = max_dd_pct
        self.vol_throttle = vol_throttle
        self.max_consecutive_losses = max_consecutive_losses
        self.session_loss_limit_pct = session_loss_limit_pct
        self.cooldown_seconds = cooldown_seconds

        self.equity = 100000.0
        self.peak_equity = self.equity
        self.current_dd = 0.0

        self.session_start_equity = self.equity
        self.consecutive_losses = 0
        self.last_loss_time = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        self._trade_log: List[Dict[str, Any]] = []

        logger.info(
            f"RiskGuard initialized: max_risk={max_risk_pct}%, "
            f"max_dd={max_dd_pct}%, vol_throttle={vol_throttle}"
        )

    def check_entry(
        self,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss: float,
        volatility: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Check if a new entry is allowed.

        Returns:
            dict with 'allowed' (bool), 'reason' (str), 'risk_pct' (float)
        """
        risk_amount = abs(quantity * (price - stop_loss))
        risk_pct = (risk_amount / self.equity) * 100 if self.equity > 0 else 100

        if risk_pct > self.max_risk_pct:
            return {
                "allowed": False,
                "reason": f"risk {risk_pct:.2f}% > max {self.max_risk_pct}%",
                "risk_pct": risk_pct,
            }

        dd_pct = (self.current_dd / self.peak_equity * 100) if self.peak_equity > 0 else 0
        if dd_pct > self.max_dd_pct:
            return {
                "allowed": False,
                "reason": f"drawdown {dd_pct:.2f}% > max {self.max_dd_pct}%",
                "risk_pct": risk_pct,
            }

        if volatility > self.vol_throttle > 0:
            return {
                "allowed": False,
                "reason": f"volatility {volatility:.4f} > throttle {self.vol_throttle}",
                "risk_pct": risk_pct,
            }

        if self.consecutive_losses >= self.max_consecutive_losses:
            elapsed = time.time() - self.last_loss_time
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return {
                    "allowed": False,
                    "reason": f"cooldown: {self.consecutive_losses} consecutive losses, {remaining:.0f}s remaining",
                    "risk_pct": risk_pct,
                }

        session_pnl_pct = 0.0
        if self.session_start_equity > 0:
            session_pnl_pct = ((self.equity - self.session_start_equity) / self.session_start_equity) * 100
        if session_pnl_pct < -self.session_loss_limit_pct:
            return {
                "allowed": False,
                "reason": f"session loss {session_pnl_pct:.2f}% > limit {self.session_loss_limit_pct}%",
                "risk_pct": risk_pct,
            }

        return {"allowed": True, "reason": "ok", "risk_pct": risk_pct}

    def check_signal(
        self,
        signal_direction: str,
        confidence: float = 0.0,
        current_volatility: float = 0.0,
    ) -> tuple:
        """
        Signal-level risk check (no position details needed).
        Used by UnifiedAIIndicator before emitting a directional signal.

        Returns:
            (allowed: bool, reason: str)
        """
        dd_pct = (self.current_dd / self.peak_equity * 100) if self.peak_equity > 0 else 0
        if dd_pct > self.max_dd_pct:
            return False, f"drawdown {dd_pct:.2f}% > max {self.max_dd_pct}%"

        if current_volatility > self.vol_throttle > 0:
            return False, f"volatility {current_volatility:.4f} > throttle {self.vol_throttle}"

        if self.consecutive_losses >= self.max_consecutive_losses:
            elapsed = time.time() - self.last_loss_time
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return False, f"cooldown: {self.consecutive_losses} losses, {remaining:.0f}s left"

        session_pnl_pct = 0.0
        if self.session_start_equity > 0:
            session_pnl_pct = ((self.equity - self.session_start_equity) / self.session_start_equity) * 100
        if session_pnl_pct < -self.session_loss_limit_pct:
            return False, f"session loss {session_pnl_pct:.2f}% > limit {self.session_loss_limit_pct}%"

        return True, "ok"

    def suggest_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        confidence: float,
        direction: str = "BUY",
    ) -> Dict[str, Any]:
        """
        Suggest whether to exit a position.

        Returns:
            dict with 'exit' (bool), 'reason' (str)
        """
        if direction == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        if pnl_pct < -1.0:
            return {"exit": True, "reason": f"hard stop: {pnl_pct:.2f}% loss", "pnl_pct": pnl_pct}

        if confidence < 0.4:
            return {"exit": True, "reason": f"confidence dropped to {confidence:.3f}", "pnl_pct": pnl_pct}

        if pnl_pct > 2.0 and confidence < 0.7:
            return {"exit": True, "reason": f"take profit: {pnl_pct:.2f}% gain, confidence={confidence:.3f}", "pnl_pct": pnl_pct}

        return {"exit": False, "reason": "hold", "pnl_pct": pnl_pct}

    def record_trade(self, pnl: float, symbol: str = ""):
        """Record a completed trade for tracking."""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl >= 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.last_loss_time = time.time()

        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.current_dd = max(0, self.peak_equity - self.equity)

        self._trade_log.append({
            "symbol": symbol,
            "pnl": pnl,
            "equity": self.equity,
            "dd_pct": (self.current_dd / self.peak_equity * 100) if self.peak_equity > 0 else 0,
            "time": time.time(),
        })
        if len(self._trade_log) > 10000:
            self._trade_log = self._trade_log[-5000:]

    def update_equity(self, new_equity: float):
        self.equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
        self.current_dd = max(0, self.peak_equity - new_equity)

    def reset_session(self):
        self.session_start_equity = self.equity
        self.consecutive_losses = 0
        logger.info(f"Session reset. Equity: {self.equity:.2f}")

    def get_status(self) -> Dict[str, Any]:
        dd_pct = (self.current_dd / self.peak_equity * 100) if self.peak_equity > 0 else 0
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        session_pnl = self.equity - self.session_start_equity

        return {
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_dd_pct": round(dd_pct, 2),
            "session_pnl": round(session_pnl, 2),
            "total_trades": self.total_trades,
            "win_rate": round(win_rate, 2),
            "consecutive_losses": self.consecutive_losses,
            "total_pnl": round(self.total_pnl, 2),
            "risk_allowed": round(self.max_risk_pct * self.equity / 100, 2),
        }
