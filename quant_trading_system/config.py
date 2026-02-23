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
        default_factory=lambda: [s.strip() for s in os.getenv("QT_DEFAULT_SYMBOLS", "AAPL,MSFT,TSLA").split(",") if s.strip()]
    )
    default_start: str = os.getenv("QT_DEFAULT_START", "2022-01-01")
    default_end: str = os.getenv("QT_DEFAULT_END", "2022-12-31")


settings = Settings()


def ensure_cache_dir() -> Path:
    """Create the cache directory if it does not exist and return its path."""
    settings.data_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_cache_dir
