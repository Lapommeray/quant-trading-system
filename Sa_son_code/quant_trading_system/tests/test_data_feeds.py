from Sa_son_code.quant_trading_system.data_feeds.alpha_vantage_adapter import YFinanceAdapter


class _FakeHistory:
    def iterrows(self):
        from datetime import datetime

        yield datetime(2024, 1, 1), {
            "Open": 100,
            "High": 110,
            "Low": 90,
            "Close": 105,
            "Volume": 1000,
        }


class _FakeTicker:
    def history(self, period="1mo", interval="1d"):
        return _FakeHistory()


class _FakeYF:
    def Ticker(self, symbol):
        return _FakeTicker()


def test_yfinance_adapter_parses_ohlcv(monkeypatch):
    adapter = YFinanceAdapter.__new__(YFinanceAdapter)
    adapter._yf = _FakeYF()

    rows = adapter.get_ohlcv("AAPL")
    assert len(rows) == 1
    assert rows[0]["close"] == 105.0
