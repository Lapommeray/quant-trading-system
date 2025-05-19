"""
Dark Pool DNA Sequencing
Analyzes institutional order flow patterns for hidden liquidity signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import random
from collections import deque

class DarkPoolDNA:
    def __init__(self):
        """Initialize Dark Pool DNA Sequencer"""
        self.dna_sequences = {}
        self.last_update = {}
        self.update_frequency = timedelta(hours=2)
        self.sequence_memory = 50  # Number of sequences to remember
        self.trade_memory = deque(maxlen=self.sequence_memory)
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        self.prime_clusters = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
    def fetch_dark_pool_dna(self, symbol):
        """
        Fetch dark pool DNA for a symbol
        
        In production, this would connect to institutional order flow data.
        For now, we use synthetic data generation.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with dark pool DNA data
        """
        if (symbol in self.dna_sequences and symbol in self.last_update and 
            datetime.now() - self.last_update[symbol] < self.update_frequency):
            return self.dna_sequences[symbol]
        
        now = datetime.now()
        
        dna_structure = {
            "symbol": symbol,
            "timestamp": now.isoformat(),
            "order_size_dna": self._generate_order_size_dna(symbol),
            "timing_dna": self._generate_timing_dna(),
            "price_level_dna": self._generate_price_level_dna(symbol),
            "trade_clusters": self._generate_trade_clusters(symbol),
            "institutional_fingerprints": self._generate_institutional_fingerprints()
        }
        
        self.dna_sequences[symbol] = dna_structure
        self.last_update[symbol] = now
        
        return dna_structure
    
    def _generate_order_size_dna(self, symbol):
        """Generate synthetic order size DNA"""
        if 'BTC' in symbol:
            base_size = 0.5  # BTC
            modulo_preference = 5  # Preference for sizes that are multiples of 0.5 BTC
        elif 'ETH' in symbol:
            base_size = 5  # ETH
            modulo_preference = 5  # Preference for sizes that are multiples of 5 ETH
        elif 'XAU' in symbol:
            base_size = 100  # Gold oz
            modulo_preference = 100  # Preference for sizes that are multiples of 100 oz
        elif symbol in ['SPY', 'QQQ', 'DIA']:
            base_size = 1000  # ETF shares
            modulo_preference = 500  # Preference for sizes that are multiples of 500 shares
        else:
            base_size = 100  # Default
            modulo_preference = 100
        
        order_sizes = []
        
        fib_sizes = [base_size * level for level in self.fibonacci_levels]
        
        prime_sizes = [base_size * prime for prime in self.prime_clusters]
        
        round_sizes = [base_size * i * 10 for i in range(1, 6)]
        
        all_sizes = fib_sizes + prime_sizes + round_sizes
        
        for _ in range(10):
            size = random.choice(all_sizes) * random.uniform(0.9, 1.1)
            size = round(size / modulo_preference) * modulo_preference
            order_sizes.append(size)
        
        size_distribution = {}
        for size in order_sizes:
            size_key = str(int(size))
            if size_key in size_distribution:
                size_distribution[size_key] += 1
            else:
                size_distribution[size_key] = 1
        
        modulo_distribution = {}
        for size in order_sizes:
            modulo = size % modulo_preference
            modulo_key = str(int(modulo))
            if modulo_key in modulo_distribution:
                modulo_distribution[modulo_key] += 1
            else:
                modulo_distribution[modulo_key] = 1
        
        return {
            "typical_sizes": sorted(order_sizes),
            "size_distribution": size_distribution,
            "modulo_distribution": modulo_distribution,
            "fibonacci_preference": random.uniform(0.3, 0.8),
            "prime_preference": random.uniform(0.2, 0.6),
            "round_number_preference": random.uniform(0.4, 0.9)
        }
    
    def _generate_timing_dna(self):
        """Generate synthetic timing DNA"""
        hours = list(range(24))
        hour_weights = []
        
        for hour in hours:
            if 9 <= hour <= 16:  # US market hours (EST)
                weight = random.uniform(0.7, 1.0)
            elif 3 <= hour <= 6:  # Asian market hours
                weight = random.uniform(0.4, 0.7)
            elif 17 <= hour <= 20:  # European market hours overlap
                weight = random.uniform(0.5, 0.8)
            else:
                weight = random.uniform(0.1, 0.4)
            hour_weights.append(weight)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_weights = []
        
        for day in days:
            if day in ["Tuesday", "Wednesday"]:
                weight = random.uniform(0.7, 1.0)  # Mid-week preference
            elif day == "Friday":
                weight = random.uniform(0.4, 0.7)  # Lower Friday preference
            else:
                weight = random.uniform(0.5, 0.9)
            day_weights.append(weight)
        
        timestamp_patterns = []
        
        interval_minutes = random.choice([5, 10, 15, 30, 60])
        timestamp_patterns.append({
            "type": "regular_interval",
            "interval_minutes": interval_minutes,
            "confidence": random.uniform(0.6, 0.9)
        })
        
        timestamp_patterns.append({
            "type": "end_of_hour",
            "minutes_before": random.randint(1, 5),
            "confidence": random.uniform(0.7, 0.95)
        })
        
        timestamp_patterns.append({
            "type": "market_open_close",
            "minutes_after_open": random.randint(5, 30),
            "minutes_before_close": random.randint(5, 30),
            "confidence": random.uniform(0.8, 0.98)
        })
        
        return {
            "hour_preferences": dict(zip([str(h) for h in hours], hour_weights)),
            "day_preferences": dict(zip(days, day_weights)),
            "timestamp_patterns": timestamp_patterns,
            "interval_preference": interval_minutes,
            "end_of_period_preference": random.uniform(0.5, 0.9)
        }
    
    def _generate_price_level_dna(self, symbol):
        """Generate synthetic price level DNA"""
        price_levels = []
        
        base_price = self._get_base_price(symbol)
        range_size = base_price * 0.1  # 10% range
        
        fib_levels = []
        for level in self.fibonacci_levels:
            price = base_price - (range_size * level)
            fib_levels.append({
                "price": price,
                "type": "fibonacci",
                "level": level,
                "strength": random.uniform(0.7, 0.95)
            })
        
        psych_levels = []
        for i in range(-5, 6):
            if i == 0:
                continue
            
            magnitude = 10 ** (len(str(int(base_price))) - 1)
            price = round(base_price / magnitude) * magnitude + (i * magnitude / 10)
            
            psych_levels.append({
                "price": price,
                "type": "psychological",
                "level": i,
                "strength": random.uniform(0.6, 0.9)
            })
        
        vwap_levels = []
        vwap = base_price * random.uniform(0.98, 1.02)
        vwap_levels.append({
            "price": vwap,
            "type": "vwap",
            "period": "daily",
            "strength": random.uniform(0.75, 0.95)
        })
        
        price_levels = fib_levels + psych_levels + vwap_levels
        
        return {
            "price_levels": sorted(price_levels, key=lambda x: x["price"]),
            "fibonacci_preference": random.uniform(0.6, 0.9),
            "psychological_preference": random.uniform(0.5, 0.8),
            "vwap_preference": random.uniform(0.7, 0.95)
        }
    
    def _generate_trade_clusters(self, symbol):
        """Generate synthetic trade clusters"""
        clusters = []
        
        for _ in range(3):
            hour = random.randint(9, 16)
            minute = random.choice([0, 15, 30, 45])
            
            clusters.append({
                "type": "time",
                "hour": hour,
                "minute": minute,
                "duration_minutes": random.randint(5, 15),
                "volume_multiplier": random.uniform(1.5, 3.0),
                "confidence": random.uniform(0.7, 0.9)
            })
        
        base_price = self._get_base_price(symbol)
        
        for _ in range(3):
            price = base_price * random.uniform(0.95, 1.05)
            
            clusters.append({
                "type": "price",
                "price": price,
                "price_range": price * random.uniform(0.005, 0.02),
                "volume_multiplier": random.uniform(1.5, 3.0),
                "confidence": random.uniform(0.7, 0.9)
            })
        
        clusters.append({
            "type": "news",
            "minutes_after_news": random.randint(5, 30),
            "duration_minutes": random.randint(15, 60),
            "volume_multiplier": random.uniform(2.0, 5.0),
            "confidence": random.uniform(0.8, 0.95)
        })
        
        return clusters
    
    def _generate_institutional_fingerprints(self):
        """Generate synthetic institutional fingerprints"""
        fingerprints = []
        
        institutions = [
            "BlackRock", "Vanguard", "Fidelity", "State Street",
            "JPMorgan", "Goldman Sachs", "Citadel", "Renaissance",
            "Two Sigma", "AQR", "Bridgewater", "PIMCO"
        ]
        
        selected_institutions = random.sample(institutions, random.randint(3, 6))
        
        for institution in selected_institutions:
            fingerprint = {
                "institution": institution,
                "order_size_preference": random.choice(self.fibonacci_levels) * 1000,
                "time_of_day_preference": random.randint(9, 16),
                "day_of_week_preference": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
                "execution_style": random.choice(["aggressive", "passive", "mixed"]),
                "confidence": random.uniform(0.6, 0.9)
            }
            
            fingerprints.append(fingerprint)
        
        return fingerprints
    
    def _get_base_price(self, symbol):
        """Get base price for a symbol"""
        if 'BTC' in symbol:
            return 40000
        elif 'ETH' in symbol:
            return 2500
        elif 'XAU' in symbol:
            return 1800
        elif symbol == 'SPY':
            return 420
        elif symbol == 'QQQ':
            return 360
        elif symbol == 'DIA':
            return 350
        else:
            return 100
    
    def analyze_dna_sequence(self, symbol, current_price=None):
        """
        Analyze dark pool DNA for trading signals
        
        Parameters:
        - symbol: Trading symbol
        - current_price: Current market price (optional)
        
        Returns:
        - Dictionary with signal information
        """
        dna = self.fetch_dark_pool_dna(symbol)
        
        signal = {
            "direction": None,
            "confidence": 0,
            "message": "No clear DNA signal",
            "dna_markers": {}
        }
        
        if current_price and "price_levels" in dna.get("price_level_dna", {}):
            price_levels = dna["price_level_dna"]["price_levels"]
            
            closest_levels = []
            for level in price_levels:
                distance = abs(level["price"] - current_price)
                level_info = {
                    "price": level["price"],
                    "type": level["type"],
                    "distance": distance,
                    "distance_percent": distance / current_price,
                    "strength": level["strength"]
                }
                closest_levels.append(level_info)
            
            closest_levels = sorted(closest_levels, key=lambda x: x["distance"])[:3]
            
            strong_levels = [level for level in closest_levels 
                            if level["distance_percent"] < 0.01 and level["strength"] > 0.7]
            
            if strong_levels:
                supports = [level for level in strong_levels if level["price"] < current_price]
                resistances = [level for level in strong_levels if level["price"] > current_price]
                
                if supports and not resistances:
                    signal["direction"] = "BUY"
                    signal["confidence"] = max(level["strength"] for level in supports)
                    signal["message"] = "Price near strong DNA support levels"
                elif resistances and not supports:
                    signal["direction"] = "SELL"
                    signal["confidence"] = max(level["strength"] for level in resistances)
                    signal["message"] = "Price near strong DNA resistance levels"
            
            signal["dna_markers"]["price_levels"] = closest_levels
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        day = now.strftime("%A")
        
        if "timing_dna" in dna:
            timing_dna = dna["timing_dna"]
            
            for pattern in timing_dna.get("timestamp_patterns", []):
                if pattern["type"] == "end_of_hour" and minute >= 60 - pattern["minutes_before"]:
                    if signal["direction"] is None:
                        signal["direction"] = "BUY" if random.random() < 0.5 else "SELL"
                        signal["confidence"] = pattern["confidence"] * 0.7
                        signal["message"] = "End of hour DNA pattern detected"
                elif pattern["type"] == "market_open_close":
                    if hour == 9 and minute >= 30 and minute < 30 + pattern["minutes_after_open"]:
                        if signal["direction"] is None:
                            signal["direction"] = "BUY"  # Usually bullish after open
                            signal["confidence"] = pattern["confidence"] * 0.8
                            signal["message"] = "Market open DNA pattern detected"
                    
                    elif hour == 15 and minute >= 60 - pattern["minutes_before_close"]:
                        if signal["direction"] is None:
                            signal["direction"] = "SELL"  # Usually bearish before close
                            signal["confidence"] = pattern["confidence"] * 0.8
                            signal["message"] = "Market close DNA pattern detected"
        
        return signal
    
    def get_dna_report(self, symbol):
        """
        Generate detailed DNA report for a symbol
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dictionary with detailed DNA metrics
        """
        dna = self.fetch_dark_pool_dna(symbol)
        
        report = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "dna_sequence": dna,
            "trade_memory_length": len(self.trade_memory),
            "analysis": self.analyze_dna_sequence(symbol)
        }
        
        return report
