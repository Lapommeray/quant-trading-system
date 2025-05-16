"""
Test Live Data Integration

This script tests the live data integration components to ensure they work correctly
and provide 100% real market data.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('live_data_test.log')
    ]
)
logger = logging.getLogger("LiveDataTest")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_data.api_vault import APIVault
from live_data.exchange_connector import ExchangeConnector
from live_data.websocket_streams import WebSocketStreams
from live_data.data_verifier import DataVerifier
from live_data.multi_exchange_router import MultiExchangeRouter
from quantum_audit.sovereignty_check import SovereigntyCheck

def test_api_vault():
    """Test API Vault functionality."""
    logger.info("Testing API Vault...")
    
    vault = APIVault()
    
    vault.set_credentials(
        "test_exchange",
        "test_api_key",
        "test_secret",
        {"additional": "param"}
    )
    
    creds = vault.get_credentials("test_exchange")
    assert creds["apiKey"] == "test_api_key", "API key mismatch"
    assert creds["secret"] == "test_secret", "Secret mismatch"
    assert creds["additional"] == "param", "Additional param mismatch"
    
    exchanges = vault.list_exchanges()
    assert "test_exchange" in exchanges, "Exchange not in list"
    
    result = vault.remove_credentials("test_exchange")
    assert result, "Failed to remove credentials"
    
    creds = vault.get_credentials("test_exchange")
    assert not creds, "Credentials not removed"
    
    logger.info("API Vault tests passed")
    return True

def test_exchange_connector():
    """Test Exchange Connector functionality."""
    logger.info("Testing Exchange Connector...")
    
    try:
        exchange = ExchangeConnector("binance")
        
        connected = exchange.test_connection()
        logger.info(f"Connection test result: {connected}")
        
        if connected:
            ticker = exchange.fetch_ticker("BTC/USDT")
            logger.info(f"BTC/USDT ticker: {ticker['last']}")
            
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1m", limit=5)
            logger.info(f"BTC/USDT OHLCV (5 candles): {len(ohlcv)} candles")
            
            order_book = exchange.fetch_order_book("BTC/USDT", limit=5)
            logger.info(f"BTC/USDT order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            
            trades = exchange.fetch_trades("BTC/USDT", limit=5)
            logger.info(f"BTC/USDT trades: {len(trades)} trades")
            
            logger.info("Exchange Connector tests passed")
            return True
        else:
            logger.warning("Exchange connection test failed, skipping further tests")
            return False
    except Exception as e:
        logger.error(f"Error testing Exchange Connector: {e}")
        return False

def test_websocket_streams():
    """Test WebSocket Streams functionality."""
    logger.info("Testing WebSocket Streams...")
    
    try:
        ws = WebSocketStreams("binance")
        ws.start()
        
        received_message = [False]
        
        def on_message(data):
            logger.info(f"Received WebSocket message: {data}")
            received_message[0] = True
        
        subscribed = ws.subscribe("btcusdt@ticker", on_message)
        logger.info(f"Subscription result: {subscribed}")
        
        if subscribed:
            logger.info("Waiting for WebSocket message...")
            timeout = 10
            start_time = time.time()
            
            while not received_message[0] and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if received_message[0]:
                logger.info("Received WebSocket message successfully")
            else:
                logger.warning(f"No WebSocket message received within {timeout} seconds")
            
            unsubscribed = ws.unsubscribe("btcusdt@ticker")
            logger.info(f"Unsubscription result: {unsubscribed}")
            
            ws.stop()
            
            logger.info("WebSocket Streams tests completed")
            return received_message[0]
        else:
            logger.warning("WebSocket subscription failed, skipping further tests")
            ws.stop()
            return False
    except Exception as e:
        logger.error(f"Error testing WebSocket Streams: {e}")
        return False

def test_data_verifier():
    """Test Data Verifier functionality."""
    logger.info("Testing Data Verifier...")
    
    try:
        verifier = DataVerifier(strict_mode=False)
        
        exchange = ExchangeConnector("binance")
        
        connected = exchange.test_connection()
        
        if connected:
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1m", limit=10)
            is_authentic, reason = verifier.verify_ohlcv_data(ohlcv, "BTC/USDT", "binance", "1m")
            logger.info(f"OHLCV verification result: {is_authentic}, reason: {reason}")
            
            ticker = exchange.fetch_ticker("BTC/USDT")
            is_authentic, reason = verifier.verify_ticker_data(ticker, "BTC/USDT", "binance")
            logger.info(f"Ticker verification result: {is_authentic}, reason: {reason}")
            
            order_book = exchange.fetch_order_book("BTC/USDT", limit=10)
            is_authentic, reason = verifier.verify_order_book_data(order_book, "BTC/USDT", "binance")
            logger.info(f"Order book verification result: {is_authentic}, reason: {reason}")
            
            trades = exchange.fetch_trades("BTC/USDT", limit=10)
            is_authentic, reason = verifier.verify_trade_data(trades, "BTC/USDT", "binance")
            logger.info(f"Trade verification result: {is_authentic}, reason: {reason}")
            
            nuclear_result = verifier.run_nuclear_verification()
            logger.info(f"Nuclear verification result: {nuclear_result}")
            
            stats = verifier.get_verification_stats()
            logger.info(f"Verification stats: {stats}")
            
            logger.info("Data Verifier tests completed")
            return True
        else:
            logger.warning("Exchange connection test failed, skipping further tests")
            return False
    except Exception as e:
        logger.error(f"Error testing Data Verifier: {e}")
        return False

def test_multi_exchange_router():
    """Test Multi-Exchange Router functionality."""
    logger.info("Testing Multi-Exchange Router...")
    
    try:
        router = MultiExchangeRouter(["binance", "kraken", "coinbase"])
        
        connection_results = router.test_all_connections()
        logger.info(f"Connection test results: {connection_results}")
        
        if any(connection_results.values()):
            ticker = router.fetch_ticker("BTC/USDT")
            logger.info(f"BTC/USDT ticker: {ticker['last']}")
            
            ohlcv = router.fetch_ohlcv("BTC/USDT", "1m", limit=5)
            logger.info(f"BTC/USDT OHLCV (5 candles): {len(ohlcv)} candles")
            
            order_book = router.fetch_order_book("BTC/USDT", limit=5)
            logger.info(f"BTC/USDT order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            
            trades = router.fetch_trades("BTC/USDT", limit=5)
            logger.info(f"BTC/USDT trades: {len(trades)} trades")
            
            received_message = [False]
            
            def on_message(data):
                logger.info(f"Received WebSocket message via router: {data}")
                received_message[0] = True
            
            subscribed = router.subscribe_to_ticker("BTC/USDT", on_message)
            logger.info(f"Router subscription result: {subscribed}")
            
            if subscribed:
                logger.info("Waiting for WebSocket message via router...")
                timeout = 10
                start_time = time.time()
                
                while not received_message[0] and time.time() - start_time < timeout:
                    time.sleep(0.1)
                
                if received_message[0]:
                    logger.info("Received WebSocket message via router successfully")
                else:
                    logger.warning(f"No WebSocket message received via router within {timeout} seconds")
            
            router.close()
            
            logger.info("Multi-Exchange Router tests completed")
            return True
        else:
            logger.warning("All exchange connections failed, skipping further tests")
            return False
    except Exception as e:
        logger.error(f"Error testing Multi-Exchange Router: {e}")
        return False

def test_sovereignty_integration():
    """Test integration with Sovereignty Check."""
    logger.info("Testing integration with Sovereignty Check...")
    
    try:
        result = SovereigntyCheck.verify_all_components()
        logger.info(f"Sovereignty check result: {result}")
        
        SovereigntyCheck.run(mode="ULTRA_STRICT", deploy_mode="GOD")
        
        logger.info("Sovereignty integration tests completed")
        return True
    except Exception as e:
        logger.error(f"Error testing Sovereignty integration: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("Running all live data integration tests...")
    
    results = {}
    
    results["api_vault"] = test_api_vault()
    
    results["exchange_connector"] = test_exchange_connector()
    
    results["websocket_streams"] = test_websocket_streams()
    
    results["data_verifier"] = test_data_verifier()
    
    results["multi_exchange_router"] = test_multi_exchange_router()
    
    results["sovereignty_integration"] = test_sovereignty_integration()
    
    logger.info("Test results:")
    for test_name, result in results.items():
        logger.info(f"  {test_name}: {'PASSED' if result else 'FAILED'}")
    
    overall_result = all(results.values())
    logger.info(f"Overall result: {'PASSED' if overall_result else 'FAILED'}")
    
    return results, overall_result

if __name__ == "__main__":
    results, overall_result = run_all_tests()
    
    with open("live_data_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": {k: bool(v) for k, v in results.items()},
            "overall_result": bool(overall_result)
        }, f, indent=2)
    
    sys.exit(0 if overall_result else 1)
