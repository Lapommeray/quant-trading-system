"""
OKX Live Config - real trading configuration, fail-closed.

No simulation fallback. Requires real credentials and ccxt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class OKXLiveConfig:
    api_key: str = field(
        default_factory=lambda: os.getenv("OKX_API_KEY") or os.getenv("OKX_KEY") or ""
    )
    api_secret: str = field(
        default_factory=lambda: os.getenv("OKX_API_SECRET")
        or os.getenv("OKX_SECRET")
        or ""
    )
    passphrase: str = field(default_factory=lambda: os.getenv("OKX_PASSPHRASE") or "")
    live_trading: bool = field(
        default_factory=lambda: os.getenv("OKX_LIVE_TRADING", "false").lower()
        in ("true", "1", "yes")
    )
    allow_paper_for_test: bool = field(
        default_factory=lambda: os.getenv("OKX_ALLOW_PAPER_FOR_TEST", "false").lower()
        in ("true", "1", "yes")
    )

    max_leverage: float = field(
        default_factory=lambda: float(os.getenv("OKX_MAX_LEVERAGE", "3.0"))
    )
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("OKX_MAX_POSITION_PCT", "0.10"))
    )
    max_daily_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("OKX_MAX_DAILY_LOSS_PCT", "0.03"))
    )
    max_orders_per_minute: int = field(
        default_factory=lambda: int(os.getenv("OKX_MAX_ORDERS_PER_MIN", "20"))
    )

    default_symbols: List[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("QT_DEFAULT_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
            if s.strip()
        ]
        or ["BTC/USDT"]
    )
    log_dir: Path = field(default_factory=lambda: Path("audit_logs"))

    def validate_for_real_trading(self) -> None:
        """Fails closed if real trading prereqs missing."""
        missing = []

        # ccxt required
        try:
            import ccxt  # noqa: F401
        except ImportError:
            missing.append("ccxt package (pip install ccxt)")

        if self.live_trading:
            if not self.api_key:
                missing.append("OKX_API_KEY env var")
            if not self.api_secret:
                missing.append("OKX_API_SECRET env var")
            if not self.passphrase:
                missing.append("OKX_PASSPHRASE env var")
            if self.api_key and len(self.api_key) < 10:
                missing.append("OKX_API_KEY looks invalid")
            if self.api_secret and len(self.api_secret) < 10:
                missing.append("OKX_API_SECRET looks invalid")

        if missing:
            raise RuntimeError(
                f"Real trading prerequisites missing (fail-closed): {', '.join(missing)} - "
                f"Set OKX_LIVE_TRADING=false for paper or set credentials for live. No synthetic fallback."
            )

    @property
    def is_paper(self) -> bool:
        return not self.live_trading

    @classmethod
    def from_env(cls) -> "OKXLiveConfig":
        cfg = cls()
        # Validate immediately for real trading mode
        if cfg.live_trading:
            cfg.validate_for_real_trading()
        return cfg


# Global singleton for convenience
_config: Optional[OKXLiveConfig] = None


def get_okx_config() -> OKXLiveConfig:
    global _config
    if _config is None:
        _config = OKXLiveConfig.from_env()
    return _config


def reset_config():
    global _config
    _config = None
