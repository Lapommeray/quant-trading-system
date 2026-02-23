"""Pydantic schemas for the API layer."""

from pydantic import BaseModel
from typing import List


class CandleResponse(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    candles: List[CandleResponse]
