"""
Live Data Integration Module

This module provides real-time market data integration with multiple exchanges
using CCXT and WebSocket connections. It ensures 100% real data with no synthetic
or fake data sources.
"""

from live_data.exchange_connector import ExchangeConnector
from live_data.websocket_streams import WebSocketStreams
from live_data.api_vault import APIVault
from live_data.data_verifier import DataVerifier
from live_data.multi_exchange_router import MultiExchangeRouter

__all__ = [
    'ExchangeConnector',
    'WebSocketStreams',
    'APIVault',
    'DataVerifier',
    'MultiExchangeRouter'
]
