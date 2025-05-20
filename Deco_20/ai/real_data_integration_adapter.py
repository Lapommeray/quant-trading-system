"""
Real Data Integration Adapter for Conscious Intelligence Layer

This module adapts real data sources to be used by the Conscious Intelligence Layer,
transforming raw data into conscious perceptions of market intention.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from real_data_integration.real_data_connector import RealDataConnector, ComplianceCheck

class RealDataIntegrationAdapter:
    """
    Adapts real data sources to be used by the Conscious Intelligence Layer,
    transforming raw data into conscious perceptions of market intention.
    """
    
    def __init__(self, algorithm, api_keys=None):
        """
        Initialize the real data integration adapter.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        - api_keys: Dictionary of API keys for various data sources
        """
        self.algorithm = algorithm
        self.connector = RealDataConnector(algorithm, api_keys)
        self.compliance = ComplianceCheck()
        self.perception_cache = {}
        self.last_perception = {}
        
        self.logger = logging.getLogger("RealDataIntegrationAdapter")
        self.logger.setLevel(logging.INFO)
        
        algorithm.Debug("Real Data Integration Adapter initialized")
    
    def perceive_market_intention(self, symbol, history_data=None):
        """
        Transform raw data into conscious perceptions of market intention.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - Dictionary containing market intention perceptions
        """
        symbol_str = str(symbol)
        
        onchain_data = None
        dark_pool_data = None
        sentiment_data = None
        
        if symbol_str in ["BTCUSD", "ETHUSD"]:
            crypto_symbol = symbol_str[:3]  # Extract BTC or ETH
            onchain_data = self.connector.get_onchain_crypto_data(crypto_symbol)
        
        if symbol_str in ["DIA", "QQQ"]:
            dark_pool_data = self.connector.get_dark_pool_data(symbol_str)
            sentiment_data = self.connector.get_earnings_sentiment(symbol_str)
        
        if symbol_str == "XAUUSD":
            satellite_data = self.connector.get_satellite_data("port", "global_shipping")
        else:
            satellite_data = None
        
        compliance_result = self.compliance.pre_trade_check(symbol_str)
        
        intention = self._transform_to_intention(
            symbol_str,
            onchain_data,
            dark_pool_data,
            sentiment_data,
            satellite_data,
            history_data
        )
        
        intention["compliance"] = {
            "allowed": compliance_result["allowed"],
            "warnings": compliance_result["warnings"],
            "limit_order_size": compliance_result["limit_order_size"]
        }
        
        self.perception_cache[symbol_str] = intention
        self.last_perception[symbol_str] = self.algorithm.Time
        
        return intention
    
    def _transform_to_intention(self, symbol, onchain_data, dark_pool_data, 
                               sentiment_data, satellite_data, history_data):
        """
        Transform raw data into conscious perceptions of market intention.
        
        Parameters:
        - symbol: Trading symbol
        - onchain_data: On-chain crypto data
        - dark_pool_data: Dark pool and institutional order flow data
        - sentiment_data: Earnings call sentiment analysis
        - satellite_data: Satellite imagery and analytics data
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - Dictionary containing market intention perceptions
        """
        intention = {
            "symbol": symbol,
            "timestamp": self.algorithm.Time,
            "quantum_field": {
                "probability_shift": 0.5,
                "timeline_convergence": 0.5,
                "quantum_entanglement": 0.5
            },
            "emotional_field": {
                "fear_greed_balance": 0.5,
                "institutional_sentiment": 0.5,
                "retail_sentiment": 0.5
            },
            "fractal_field": {
                "compression_state": 0.5,
                "expansion_potential": 0.5,
                "harmonic_alignment": 0.5
            },
            "intention_field": {
                "directional_intent": "neutral",
                "strength": 0.5,
                "clarity": 0.5
            }
        }
        
        if onchain_data and not onchain_data.get("is_simulated", False):
            transfers = onchain_data.get("value", 0)
            historical_avg = onchain_data.get("historical_avg", transfers)
            
            if historical_avg > 0:
                z_score = (transfers - historical_avg) / max(1, onchain_data.get("historical_std", 1))
                probability_shift = 0.5 + (z_score / 10.0)  # Scale to reasonable range
                
                intention["quantum_field"]["probability_shift"] = max(0.0, min(1.0, probability_shift))
                
                if z_score > 1.5:
                    intention["intention_field"]["directional_intent"] = "bullish"
                    intention["intention_field"]["strength"] = min(1.0, 0.5 + (z_score / 10.0))
                elif z_score < -1.5:
                    intention["intention_field"]["directional_intent"] = "bearish"
                    intention["intention_field"]["strength"] = min(1.0, 0.5 + (abs(z_score) / 10.0))
        
        if dark_pool_data and not dark_pool_data.get("is_simulated", False):
            spoofing_score = dark_pool_data.get("spoofing_score", 0.5)
            
            intention["emotional_field"]["institutional_sentiment"] = 1.0 - spoofing_score
            
            intention["intention_field"]["clarity"] = 1.0 - spoofing_score
        
        if sentiment_data and not sentiment_data.get("is_simulated", False):
            sentiment_score = sentiment_data.get("sentiment_score", 0.5)
            
            intention["emotional_field"]["fear_greed_balance"] = sentiment_score
            
            if sentiment_score > 0.7 and intention["intention_field"]["directional_intent"] == "neutral":
                intention["intention_field"]["directional_intent"] = "bullish"
                intention["intention_field"]["strength"] = sentiment_score
            elif sentiment_score < 0.3 and intention["intention_field"]["directional_intent"] == "neutral":
                intention["intention_field"]["directional_intent"] = "bearish"
                intention["intention_field"]["strength"] = 1.0 - sentiment_score
        
        if satellite_data and not satellite_data.get("is_simulated", False):
            activity_score = satellite_data.get("activity_score", 0.5)
            
            intention["fractal_field"]["expansion_potential"] = activity_score
            
            intention["quantum_field"]["quantum_entanglement"] = activity_score
        
        if history_data and "1m" in history_data and not history_data["1m"].empty:
            df_1m = history_data["1m"]
            if len(df_1m) > 20:
                recent_volatility = df_1m["Close"].pct_change().tail(20).std()
                historical_volatility = df_1m["Close"].pct_change().std()
                
                if historical_volatility > 0:
                    compression_ratio = recent_volatility / historical_volatility
                    
                    compression_state = 1.0 - min(1.0, compression_ratio)
                    intention["fractal_field"]["compression_state"] = compression_state
                    
                    if compression_state > 0.7:
                        intention["fractal_field"]["expansion_potential"] = min(1.0, compression_state + 0.2)
            
            if len(df_1m) > 50:
                price_range = df_1m["High"].max() - df_1m["Low"].min()
                if price_range > 0:
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    fib_prices = [df_1m["Low"].min() + (level * price_range) for level in fib_levels]
                    
                    current_price = df_1m["Close"].iloc[-1]
                    min_distance = min([abs(current_price - fib_price) / price_range for fib_price in fib_prices])
                    
                    harmonic_alignment = 1.0 - min(1.0, min_distance * 10)
                    intention["fractal_field"]["harmonic_alignment"] = harmonic_alignment
                    
                    intention["quantum_field"]["timeline_convergence"] = harmonic_alignment
        
        self._synthesize_intention_field(intention)
        
        return intention
    
    def _synthesize_intention_field(self, intention):
        """
        Synthesize the final intention field from all perceptions.
        
        Parameters:
        - intention: Dictionary containing market intention perceptions
        """
        quantum_score = (
            intention["quantum_field"]["probability_shift"] * 0.4 +
            intention["quantum_field"]["timeline_convergence"] * 0.3 +
            intention["quantum_field"]["quantum_entanglement"] * 0.3
        )
        
        emotional_score = (
            intention["emotional_field"]["fear_greed_balance"] * 0.4 +
            intention["emotional_field"]["institutional_sentiment"] * 0.4 +
            intention["emotional_field"]["retail_sentiment"] * 0.2
        )
        
        fractal_score = (
            intention["fractal_field"]["compression_state"] * 0.3 +
            intention["fractal_field"]["expansion_potential"] * 0.4 +
            intention["fractal_field"]["harmonic_alignment"] * 0.3
        )
        
        if intention["intention_field"]["directional_intent"] == "neutral":
            combined_score = (quantum_score + emotional_score + fractal_score) / 3
            
            if combined_score > 0.6:
                intention["intention_field"]["directional_intent"] = "bullish"
                intention["intention_field"]["strength"] = combined_score
            elif combined_score < 0.4:
                intention["intention_field"]["directional_intent"] = "bearish"
                intention["intention_field"]["strength"] = 1.0 - combined_score
        
        scores = [quantum_score, emotional_score, fractal_score]
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        
        clarity = 1.0 - min(1.0, variance * 10)
        intention["intention_field"]["clarity"] = clarity
    
    def get_last_perception(self, symbol):
        """
        Get the last market intention perception for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary containing market intention perceptions
        """
        symbol_str = str(symbol)
        
        if symbol_str in self.perception_cache:
            return self.perception_cache[symbol_str]
            
        return None
