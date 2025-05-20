"""
WebSocket Streams - Real-Time Market Data

This module provides real-time market data streams using WebSocket connections
to cryptocurrency exchanges.
"""

import os
import json
import time
import logging
import threading
import websocket
from typing import Dict, List, Any, Optional, Callable, Union
from queue import Queue

logger = logging.getLogger(__name__)

class WebSocketStreams:
    """
    WebSocket streams for real-time market data from cryptocurrency exchanges.
    
    This class provides methods for connecting to WebSocket streams from various
    exchanges and processing real-time market data.
    """
    
    def __init__(self, exchange: str):
        """
        Initialize WebSocket streams.
        
        Parameters:
        - exchange: Exchange name (e.g., 'binance', 'coinbase')
        """
        self.exchange = exchange.lower()
        self.connections = {}
        self.callbacks = {}
        self.running = False
        self.message_queue = Queue()
        self.processor_thread = None
        
        self.ws_endpoints = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'coinbase': 'wss://ws-feed.pro.coinbase.com',
            'kraken': 'wss://ws.kraken.com',
            'kucoin': 'wss://push1.kucoin.com/endpoint',
            'bitfinex': 'wss://api-pub.bitfinex.com/ws/2',
            'huobi': 'wss://api.huobi.pro/ws',
            'okex': 'wss://ws.okex.com:8443/ws/v5/public',
            'ftx': 'wss://ftx.com/ws/',
            'bybit': 'wss://stream.bybit.com/realtime',
        }
        
        if self.exchange not in self.ws_endpoints:
            raise ValueError(f"Exchange {self.exchange} not supported for WebSocket streams")
        
        self.ws_endpoint = self.ws_endpoints[self.exchange]
        logger.info(f"Initialized WebSocket streams for {self.exchange}")
    
    def _on_message(self, ws, message):
        """
        Handle WebSocket message.
        
        Parameters:
        - ws: WebSocket connection
        - message: Message received
        """
        try:
            data = json.loads(message)
            stream = None
            
            if self.exchange == 'binance':
                if 'stream' in data:
                    stream = data['stream']
            elif self.exchange == 'coinbase':
                if 'type' in data and 'product_id' in data:
                    stream = f"{data['type']}:{data['product_id']}"
            elif self.exchange == 'kraken':
                if 'channelName' in data and 'pair' in data:
                    stream = f"{data['channelName']}:{data['pair']}"
            
            self.message_queue.put((stream, data))
        except json.JSONDecodeError:
            logger.error(f"Failed to decode WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """
        Handle WebSocket error.
        
        Parameters:
        - ws: WebSocket connection
        - error: Error received
        """
        logger.error(f"WebSocket error for {self.exchange}: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket connection close.
        
        Parameters:
        - ws: WebSocket connection
        - close_status_code: Status code for close
        - close_msg: Close message
        """
        logger.info(f"WebSocket connection closed for {self.exchange}: {close_msg} ({close_status_code})")
        
        if self.running:
            logger.info(f"Attempting to reconnect WebSocket for {self.exchange}...")
            time.sleep(5)  # Wait before reconnecting
            for stream_name, ws_conn in list(self.connections.items()):
                if ws_conn == ws:
                    self.subscribe(stream_name, self.callbacks.get(stream_name))
    
    def _on_open(self, ws):
        """
        Handle WebSocket connection open.
        
        Parameters:
        - ws: WebSocket connection
        """
        logger.info(f"WebSocket connection opened for {self.exchange}")
    
    def _message_processor(self):
        """Process messages from the queue and dispatch to callbacks."""
        while self.running:
            try:
                stream, data = self.message_queue.get(timeout=1)
                if stream in self.callbacks and self.callbacks[stream]:
                    self.callbacks[stream](data)
                self.message_queue.task_done()
            except Exception as e:
                if not isinstance(e, TimeoutError):
                    logger.error(f"Error processing message: {e}")
    
    def start(self):
        """Start WebSocket streams."""
        if not self.running:
            self.running = True
            self.processor_thread = threading.Thread(target=self._message_processor)
            self.processor_thread.daemon = True
            self.processor_thread.start()
            logger.info(f"Started WebSocket streams for {self.exchange}")
    
    def stop(self):
        """Stop WebSocket streams."""
        self.running = False
        
        for stream_name, ws in list(self.connections.items()):
            try:
                ws.close()
                logger.info(f"Closed WebSocket connection for {stream_name}")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection for {stream_name}: {e}")
        
        self.connections = {}
        self.callbacks = {}
        
        if self.processor_thread:
            self.processor_thread.join(timeout=5)
            self.processor_thread = None
        
        logger.info(f"Stopped WebSocket streams for {self.exchange}")
    
    def subscribe(self, stream_name: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Subscribe to a WebSocket stream.
        
        Parameters:
        - stream_name: Name of the stream to subscribe to
        - callback: Callback function to handle messages
        
        Returns:
        - True if subscription was successful, False otherwise
        """
        if not self.running:
            self.start()
        
        if stream_name in self.connections:
            try:
                self.connections[stream_name].close()
            except Exception:
                pass
        
        try:
            if self.exchange == 'binance':
                ws_url = f"{self.ws_endpoint}/{stream_name}"
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()
                
                self.connections[stream_name] = ws
                self.callbacks[stream_name] = callback
                
                logger.info(f"Subscribed to {stream_name} on {self.exchange}")
                return True
            
            elif self.exchange == 'coinbase':
                ws = websocket.WebSocketApp(
                    self.ws_endpoint,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                parts = stream_name.split(":")
                channel = parts[0]
                product_id = parts[1] if len(parts) > 1 else None
                
                def on_open_cb(ws):
                    self._on_open(ws)
                    subscribe_msg = {
                        "type": "subscribe",
                        "channels": [{"name": channel}]
                    }
                    if product_id:
                        subscribe_msg["channels"][0]["product_ids"] = [product_id]
                    ws.send(json.dumps(subscribe_msg))
                
                ws.on_open = on_open_cb
                
                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()
                
                self.connections[stream_name] = ws
                self.callbacks[stream_name] = callback
                
                logger.info(f"Subscribed to {stream_name} on {self.exchange}")
                return True
            
            elif self.exchange == 'kraken':
                ws = websocket.WebSocketApp(
                    self.ws_endpoint,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                parts = stream_name.split(":")
                channel = parts[0]
                pair = parts[1] if len(parts) > 1 else None
                
                def on_open_cb(ws):
                    self._on_open(ws)
                    subscribe_msg = {
                        "name": "subscribe",
                        "subscription": {"name": channel}
                    }
                    if pair:
                        subscribe_msg["pair"] = [pair]
                    ws.send(json.dumps(subscribe_msg))
                
                ws.on_open = on_open_cb
                
                ws_thread = threading.Thread(target=ws.run_forever)
                ws_thread.daemon = True
                ws_thread.start()
                
                self.connections[stream_name] = ws
                self.callbacks[stream_name] = callback
                
                logger.info(f"Subscribed to {stream_name} on {self.exchange}")
                return True
            
            else:
                logger.error(f"WebSocket subscription not implemented for {self.exchange}")
                return False
        
        except Exception as e:
            logger.error(f"Error subscribing to {stream_name} on {self.exchange}: {e}")
            return False
    
    def unsubscribe(self, stream_name: str) -> bool:
        """
        Unsubscribe from a WebSocket stream.
        
        Parameters:
        - stream_name: Name of the stream to unsubscribe from
        
        Returns:
        - True if unsubscription was successful, False otherwise
        """
        if stream_name in self.connections:
            try:
                if self.exchange == 'binance':
                    pass
                elif self.exchange == 'coinbase':
                    parts = stream_name.split(":")
                    channel = parts[0]
                    product_id = parts[1] if len(parts) > 1 else None
                    
                    unsubscribe_msg = {
                        "type": "unsubscribe",
                        "channels": [{"name": channel}]
                    }
                    if product_id:
                        unsubscribe_msg["channels"][0]["product_ids"] = [product_id]
                    
                    self.connections[stream_name].send(json.dumps(unsubscribe_msg))
                elif self.exchange == 'kraken':
                    parts = stream_name.split(":")
                    channel = parts[0]
                    pair = parts[1] if len(parts) > 1 else None
                    
                    unsubscribe_msg = {
                        "name": "unsubscribe",
                        "subscription": {"name": channel}
                    }
                    if pair:
                        unsubscribe_msg["pair"] = [pair]
                    
                    self.connections[stream_name].send(json.dumps(unsubscribe_msg))
                
                self.connections[stream_name].close()
                del self.connections[stream_name]
                
                if stream_name in self.callbacks:
                    del self.callbacks[stream_name]
                
                logger.info(f"Unsubscribed from {stream_name} on {self.exchange}")
                return True
            except Exception as e:
                logger.error(f"Error unsubscribing from {stream_name} on {self.exchange}: {e}")
                return False
        else:
            logger.warning(f"Not subscribed to {stream_name} on {self.exchange}")
            return False
    
    def is_subscribed(self, stream_name: str) -> bool:
        """
        Check if subscribed to a stream.
        
        Parameters:
        - stream_name: Name of the stream to check
        
        Returns:
        - True if subscribed, False otherwise
        """
        return stream_name in self.connections and self.connections[stream_name].sock and self.connections[stream_name].sock.connected
