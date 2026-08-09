import math

import pandas as pd
import pytest

from quant_trading_system.data_quality import (
    DataQualityError,
    normalize_ohlcv_frame,
    validate_market_tick,
)


def _valid_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1200],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    )


def test_normalize_ohlcv_accepts_valid_data_and_keeps_utc():
    result = normalize_ohlcv_frame(_valid_ohlcv(), symbol="SPY")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(result.index.tz) == "UTC"
    assert result["Close"].iloc[-1] == 102.0


def test_normalize_ohlcv_rejects_duplicate_timestamps():
    df = _valid_ohlcv()
    df.index = [df.index[0], df.index[0]]

    with pytest.raises(DataQualityError, match="duplicate timestamps"):
        normalize_ohlcv_frame(df, symbol="SPY")


def test_normalize_ohlcv_rejects_impossible_high_low():
    df = _valid_ohlcv()
    df.loc[df.index[1], "High"] = 101.0

    with pytest.raises(DataQualityError, match="high is below"):
        normalize_ohlcv_frame(df, symbol="SPY")


def test_normalize_ohlcv_rejects_nan_or_infinite_values():
    df = _valid_ohlcv()
    df.loc[df.index[0], "Close"] = math.inf

    with pytest.raises(DataQualityError, match="non-finite"):
        normalize_ohlcv_frame(df, symbol="SPY")


def test_validate_market_tick_accepts_valid_tick():
    tick = validate_market_tick(
        {
            "ts": 1_700_000_000,
            "price": 4500,
            "open": 4490,
            "high": 4510,
            "low": 4480,
            "volume": 10,
        },
        symbol="SPX",
    )

    assert tick["price"] == 4500.0
    assert tick["volume"] == 10.0


def test_validate_market_tick_rejects_negative_or_bad_tick():
    with pytest.raises(DataQualityError, match="price must be positive"):
        validate_market_tick({"price": -1}, symbol="SPX")

    with pytest.raises(DataQualityError, match="low is above"):
        validate_market_tick(
            {"price": 100, "open": 100, "high": 101, "low": 100.5}, symbol="SPX"
        )
