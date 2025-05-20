"""
Data Feeds Package

This package provides data feeds for the QMP Overrider system.
It includes adapters for Alpha Vantage, cryptocurrency exchanges, and other data sources.
"""

from .alpha_vantage_adapter import AlphaVantageAdapter
from .crypto_feed import CryptoFeed

__all__ = ['AlphaVantageAdapter', 'CryptoFeed']
