import pandas as pd

from quant_trading_system.data_feeds import yfinance_feed


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [100, 200, 300],
        },
        index=idx,
    )


def test_get_price_history_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(yfinance_feed, "CACHE_ROOT", tmp_path)
    cached = _sample_df()
    path = yfinance_feed._cache_path("AAPL", "2023-01-01", "2023-01-04")
    cached.to_csv(path)

    result = yfinance_feed.get_price_history("AAPL", "2023-01-01", "2023-01-04", max_age_days=999)

    pd.testing.assert_frame_equal(result, cached, check_freq=False)


def test_get_price_history_downloads_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(yfinance_feed, "CACHE_ROOT", tmp_path)
    sample = _sample_df()

    def _fake_download(symbol, start, end):
        assert symbol == "MSFT"
        return sample

    monkeypatch.setattr(yfinance_feed, "_download", _fake_download)

    result = yfinance_feed.get_price_history("MSFT", "2023-01-01", "2023-01-04", force_download=True)

    assert len(result) == 3
    assert yfinance_feed._cache_path("MSFT", "2023-01-01", "2023-01-04").exists()
