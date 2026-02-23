"""API entrypoint using yfinance-backed market data."""

from fastapi import FastAPI, HTTPException

from Sa_son_code.quant_trading_system.api.schemas import MarketDataResponse
from Sa_son_code.quant_trading_system.data_feeds.alpha_vantage_adapter import YFinanceAdapter

app = FastAPI(title="Quant Trading System API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "data_provider": "yfinance"}


@app.get("/market/{symbol}", response_model=MarketDataResponse)
def market_data(symbol: str, period: str = "1mo", interval: str = "1d") -> MarketDataResponse:
    try:
        adapter = YFinanceAdapter()
        candles = adapter.get_ohlcv(symbol=symbol, period=period, interval=interval)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch market data: {exc}") from exc

    return MarketDataResponse(symbol=symbol, period=period, interval=interval, candles=candles)
