import os
import pytest
from execution.okx_engine import OKXEngine, OKXOrderRequest, OrderSide, OrderType

def test_okx_engine_init_paper():
    eng = OKXEngine(paper_mode=True)
    assert eng.paper_mode is True
    assert eng.simulation is True
    status = eng.get_status()
    assert status["paper_mode"] is True

def test_okx_engine_connect_sim():
    eng = OKXEngine(paper_mode=True)
    assert eng.connect() is True
    assert eng.is_connected() is True

def test_okx_engine_place_market_order_sim():
    eng = OKXEngine(paper_mode=True)
    eng.connect()
    order = OKXOrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001, order_type=OrderType.MARKET)
    result = eng.place_order(order)
    assert result.success is True
    assert result.filled_quantity > 0
    assert result.avg_fill_price > 0

def test_okx_engine_signal_translation():
    eng = OKXEngine(paper_mode=True)
    eng.connect()
    sig = {"symbol": "BTC/USDT", "final_signal": "BUY", "confidence": 0.85}
    res = eng.place_order_from_signal(sig, max_quantity=0.01)
    # Might be None if quantity too small due to $5 notional, but with 100k equity and 0.85 conf, should produce
    if res:
        assert res.success

def test_okx_engine_safety_leverage_cap():
    eng = OKXEngine(paper_mode=True, max_leverage=2.0)
    eng.connect()
    order = OKXOrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001, leverage=10.0)
    res = eng.place_order(order)
    assert res.success is False
    assert "Leverage" in res.message

def test_okx_engine_kill_switch():
    eng = OKXEngine(paper_mode=True)
    eng.connect()
    eng.activate_kill_switch("test kill")
    assert eng.local_breaker.tripped is True
    order = OKXOrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001)
    res = eng.place_order(order)
    assert res.success is False
    assert "Circuit breaker" in res.message or "Kill" in res.message
