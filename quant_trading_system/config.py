"""Centralized runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List


@dataclass(frozen=True)
class Settings:
    data_cache_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "cache"
    )
    max_price_age_days: int = int(os.getenv("QT_MAX_PRICE_AGE_DAYS", "7"))
    log_level: str = os.getenv("QT_LOG_LEVEL", "INFO").upper()
    default_symbols: List[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("QT_DEFAULT_SYMBOLS", "AAPL,MSFT,TSLA").split(",")
            if s.strip()
        ]
    )
    default_start: str = os.getenv("QT_DEFAULT_START", "2022-01-01")
    default_end: str = os.getenv("QT_DEFAULT_END", "2022-12-31")

    # OKX Live Trading
    okx_api_key: str = os.getenv("OKX_API_KEY", "")
    okx_api_secret: str = os.getenv("OKX_API_SECRET", "")
    okx_passphrase: str = os.getenv("OKX_PASSPHRASE", "")
    okx_live_trading: bool = os.getenv("OKX_LIVE_TRADING", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    okx_max_leverage: float = float(os.getenv("OKX_MAX_LEVERAGE", "3.0"))
    okx_max_position_pct: float = float(os.getenv("OKX_MAX_POSITION_PCT", "0.10"))
    okx_max_daily_loss_pct: float = float(os.getenv("OKX_MAX_DAILY_LOSS_PCT", "0.03"))
    okx_paper_mode: bool = field(
        default_factory=lambda: not os.getenv("OKX_LIVE_TRADING", "false").lower()
        in ("true", "1", "yes")
    )

    # Organism
    organism_self_improve_interval: int = int(
        os.getenv("ORGANISM_SELF_IMPROVE_SEC", "300")
    )
    organism_auto_discover: bool = os.getenv(
        "ORGANISM_AUTO_DISCOVER", "true"
    ).lower() in ("true", "1", "yes")

    # Safety
    safety_required_confirmations: int = int(
        os.getenv("SAFETY_REQUIRED_CONFIRMATIONS", "3")
    )
    safety_kill_switch_cooldown_min: int = int(
        os.getenv("SAFETY_KILL_COOLDOWN_MIN", "30")
    )


settings = Settings()


def ensure_cache_dir() -> Path:
    """Create the cache directory if it does not exist and return its path."""
    settings.data_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_cache_dir
