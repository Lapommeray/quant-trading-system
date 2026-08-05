"""
Polygon Pre-Broker Data Hub - S&P500 SPY/ES BEFORE broker
Reads Polygon.io WS stocks trades/quotes BEFORE Alpaca/IBKR REST.

Asset: S&P500 SPY, ES futures via proxy

Polygon WS: wss://socket.polygon.io/stocks
- T.SPY trades, Q.SPY quotes, FMV.SPY fair market value

ES futures lead: Use IBKR market depth for ES if available, but Polygon gives SPY direct with 20ms latency vs Alpaca bars 500ms.

Pushes to DataRingSPY zero-copy before broker.

Move with market maker: SPY OFI at NBBO, ES lead 5-15ms, liquidation via TRF dark prints.

Move instantly: Pre-broker data hub reads exchange direct, not broker aggregated.
"""
import time
import threading
import logging
import os

logger = logging.getLogger("PolygonPreBroker")

try:
    import websocket
    import orjson
    HAS_ORJSON = True
except ImportError:
    import json as orjson
    HAS_ORJSON = False
    import websocket

from core.data_ring import get_data_ring

class PolygonPreBrokerConnector:
    """
    Pre-broker hub for SPY / S&P500
    """
    def __init__(self, api_key=None, symbols=None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY") or os.getenv("POLYGON_KEY")
        self.symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT"]  # S&P500 components + SPY
        self.ws_url = "wss://socket.polygon.io/stocks"
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.last_message_ts = time.time()
        self.latency_history = []
        self.thread = None
        # Track last bid/ask per symbol
        self.last_quote = {}  # symbol -> {bid, ask, bid_size, ask_size}

    def _on_message(self, ws, message):
        start = time.time()
        try:
            if HAS_ORJSON:
                try:
                    data = orjson.loads(message) if isinstance(message, (str, bytes, bytearray)) else message
                except:
                    import json
                    data = json.loads(message)
            else:
                import json
                data = json.loads(message)

            # Polygon stocks message is list of events
            if isinstance(data, dict):
                data = [data]

            for ev in data:
                ev_type = ev.get("ev")
                if ev_type == "T":  # Trade
                    self._handle_trade(ev)
                elif ev_type == "Q":  # Quote
                    self._handle_quote(ev)
                elif ev_type == "FMV":  # Fair Market Value
                    self._handle_fmv(ev)
                elif ev_type == "status":
                    if ev.get("status") == "auth_success":
                        logger.info("Polygon auth success")
                        self._subscribe()
                    elif "auth_failed" in str(ev.get("status")):
                        logger.error(f"Polygon auth failed: {ev}")
                elif ev.get("action") == "subscribed":
                    logger.info(f"Polygon subscribed: {ev}")

            self.last_message_ts = time.time()
            self.reconnect_attempts = 0
            latency = (time.time() - start)*1000
            self.latency_history.append(latency)
            if len(self.latency_history) > 100:
                self.latency_history = self.latency_history[-100:]

        except Exception as e:
            logger.error(f"Polygon parse error: {e} msg={str(message)[:200]}")

    def _handle_trade(self, ev):
        try:
            symbol = ev.get("sym")
            price = float(ev.get("p", 0))
            size = float(ev.get("s", 0))
            # Polygon trade conditions: condition codes, exchange
            # For classification, use tick vs last quote
            q = self.last_quote.get(symbol, {})
            bid = q.get("bid", price*0.999)
            ask = q.get("ask", price*1.001)
            # Side classification via quote
            side = 0
            if price >= ask:
                side = 1
            elif price <= bid:
                side = -1
            else:
                # tick test vs previous price needed, use 0 neutral
                side = 0

            ring = get_data_ring(symbol)
            ring.push(
                ts=time.time(),
                bid=bid,
                ask=ask,
                bid_size=float(q.get("bid_size", 100)),
                ask_size=float(q.get("ask_size", 100)),
                price=price,
                qty=size,
                side=side
            )

        except Exception as e:
            logger.debug(f"trade handle {e}")

    def _handle_quote(self, ev):
        try:
            symbol = ev.get("sym")
            bid_price = float(ev.get("bp", ev.get("b", 0)))
            ask_price = float(ev.get("ap", ev.get("a", 0)))
            bid_size = float(ev.get("bs", ev.get("bs", 100)))
            ask_size = float(ev.get("as", 100))

            self.last_quote[symbol] = {
                "bid": bid_price,
                "ask": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "ts": time.time()
            }

            ring = get_data_ring(symbol)
            ring.push(
                ts=time.time(),
                bid=bid_price,
                ask=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                price=(bid_price+ask_price)/2,
                qty=0,
                side=0
            )

            # Check for spoof: large quote size appearing then disappearing
            # This detection here pre-broker before MM intent detector
            # If size >5x avg, could be spoof - publish event
            try:
                # Simple spoof check: if size > 5x recent avg
                # Need history, skip for now, MM detector will handle
                pass
            except:
                pass

        except Exception as e:
            logger.debug(f"quote handle {e}")

    def _handle_fmv(self, ev):
        try:
            symbol = ev.get("sym")
            fmv = float(ev.get("f", ev.get("fmv", 0)))
            # Fair market value - can be used for arbitrage vs NBBO
            # If FMV deviates > 0.05% from mid, arbitrage signal
            q = self.last_quote.get(symbol, {})
            if q:
                mid = (q.get("bid",0)+q.get("ask",0))/2
                if mid>0:
                    dev = abs(fmv-mid)/mid
                    if dev > 0.0005:  # 5 bps
                        try:
                            from core.event_bus import get_event_bus, EventPriority
                            get_event_bus().publish(
                                "FMV_ARBITRAGE",
                                {"symbol": symbol, "fmv": fmv, "mid": mid, "dev": dev},
                                source="polygon_pre_broker",
                                priority=EventPriority.OPERATIONAL
                            )
                        except:
                            pass
        except Exception as e:
            logger.debug(f"fmv handle {e}")

    def _on_error(self, ws, error):
        logger.error(f"Polygon WS error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"Polygon WS closed {code} {msg}")
        if self.running:
            self._reconnect()

    def _on_open(self, ws):
        logger.info(f"Polygon Pre-Broker connected: {self.ws_url}")
        self.reconnect_attempts = 0
        # Auth
        if self.api_key:
            auth_msg = f'{{"action":"auth","params":"{self.api_key}"}}'
            ws.send(auth_msg)
            time.sleep(0.5)
            self._subscribe()
        else:
            logger.warning("No POLYGON_API_KEY set, cannot auth Polygon WS")

    def _subscribe(self):
        try:
            # Subscribe to trades and quotes for symbols
            # Format: T.SPY,Q.SPY,FMV.SPY etc
            subs = []
            for s in self.symbols:
                subs.append(f"T.{s}")
                subs.append(f"Q.{s}")
                subs.append(f"FMV.{s}")
            sub_str = ",".join(subs)
            msg = f'{{"action":"subscribe","params":"{sub_str}"}}'
            if self.ws:
                self.ws.send(msg)
                logger.info(f"Polygon subscribed to {sub_str}")
        except Exception as e:
            logger.error(f"Polygon subscribe failed {e}")

    def _reconnect(self):
        self.reconnect_attempts += 1
        backoff = min(0.1 * (2 ** self.reconnect_attempts), 5.0)
        logger.info(f"Polygon reconnect in {backoff}s attempt {self.reconnect_attempts}")
        time.sleep(backoff)
        if self.running:
            self.start()

    def start(self):
        if self.running:
            return
        if not self.api_key:
            logger.error("POLYGON_API_KEY not set, cannot start Polygon pre-broker")
            return
        self.running = True
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.thread = threading.Thread(target=self.ws.run_forever, kwargs={"ping_interval":20, "ping_timeout":10})
        self.thread.daemon = True
        self.thread.start()
        logger.info("Polygon Pre-Broker started")

    def stop(self):
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        logger.info("Polygon Pre-Broker stopped")

    def get_latency_stats(self):
        if not self.latency_history:
            return {"p50":0,"p95":0}
        import numpy as np
        arr = np.array(self.latency_history)
        return {
            "p50": float(np.percentile(arr,50)),
            "p95": float(np.percentile(arr,95)),
            "avg": float(np.mean(arr)),
            "last_msg_age": time.time()-self.last_message_ts
        }
