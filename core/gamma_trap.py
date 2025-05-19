"""
Gamma Trap Module
Exploits dealer gamma hedging patterns for market edge
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

class GammaTrap:
    def __init__(self):
        """Initialize Gamma Trap module"""
        self.options_data = {}
        self.gamma_exposure = {}
        self.hedging_thresholds = {
            'low': -5e9,    # -$5 billion in gamma
            'extreme': -10e9 # -$10 billion in gamma
        }
        self.last_update = {}
        self.update_frequency = timedelta(hours=4)
    
    def fetch_options_data(self, symbol):
        """
        Fetch options data for a symbol
        
        In production, this would connect to options data provider.
        For now, we use synthetic data generation.
        
        Parameters:
        - symbol: Trading symbol (like SPX for S&P 500)
        
        Returns:
        - Dictionary with options data
        """
        if (symbol in self.options_data and symbol in self.last_update and 
            datetime.now() - self.last_update[symbol] < self.update_frequency):
            return self.options_data[symbol]
        
        expiries = []
        now = datetime.now()
        
        for i in range(5):  # Next 5 weeks
            expiries.append(now + timedelta(days=(4 - now.weekday()) + 7*i))
        
        for i in range(3):  # Next 3 months
            month_end = datetime(now.year, now.month + i if now.month + i <= 12 else now.month + i - 12, 
                               28 if now.month + i != 2 else 28)
            expiries.append(month_end)
        
        for i in range(4):  # Next 4 quarters
            quarter = (now.month - 1) // 3 + i
            year = now.year + quarter // 4
            quarter = quarter % 4
            quarter_end = datetime(year, quarter * 3 + 3, 28)
            expiries.append(quarter_end)
        
        expiries = sorted(list(set([exp.strftime("%Y-%m-%d") for exp in expiries])))
        
        base_price = self._get_base_price(symbol)
        
        strike_range = np.arange(base_price * 0.8, base_price * 1.2, base_price * 0.01)
        
        options_chains = {}
        total_gamma = 0
        
        for expiry in expiries:
            options_chains[expiry] = []
            days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - now).days
            
            for strike in strike_range:
                iv = self._calculate_synthetic_iv(base_price, strike, days_to_expiry)
                
                call_delta = self._calculate_synthetic_delta(base_price, strike, days_to_expiry, iv, is_call=True)
                call_gamma = self._calculate_synthetic_gamma(base_price, strike, days_to_expiry, iv)
                call_open_interest = int(np.random.exponential(1000) * math.exp(-0.01 * abs(strike - base_price)))
                
                put_delta = self._calculate_synthetic_delta(base_price, strike, days_to_expiry, iv, is_call=False)
                put_gamma = call_gamma  # Gamma is the same for calls and puts at same strike/expiry
                put_open_interest = int(np.random.exponential(1000) * math.exp(-0.01 * abs(strike - base_price)))
                
                dealer_call_position = -call_open_interest * np.random.normal(0.7, 0.2)
                dealer_put_position = -put_open_interest * np.random.normal(0.7, 0.2)
                
                call_gamma_exposure = dealer_call_position * call_gamma * 100 * base_price
                put_gamma_exposure = dealer_put_position * put_gamma * 100 * base_price
                
                total_gamma += call_gamma_exposure + put_gamma_exposure
                
                options_chains[expiry].append({
                    "strike": strike,
                    "call": {
                        "delta": call_delta,
                        "gamma": call_gamma,
                        "open_interest": call_open_interest,
                        "dealer_position": dealer_call_position,
                        "gamma_exposure": call_gamma_exposure
                    },
                    "put": {
                        "delta": put_delta,
                        "gamma": put_gamma,
                        "open_interest": put_open_interest,
                        "dealer_position": dealer_put_position,
                        "gamma_exposure": put_gamma_exposure
                    }
                })
        
        self.options_data[symbol] = {
            "price": base_price,
            "timestamp": now.isoformat(),
            "expiries": expiries,
            "options_chains": options_chains
        }
        
        self.gamma_exposure[symbol] = total_gamma
        
        self.last_update[symbol] = now
        
        return self.options_data[symbol]
    
    def _get_base_price(self, symbol):
        """Get base price for a symbol"""
        if symbol == 'SPX':
            return 4200
        elif symbol == 'NDX':
            return 14500
        elif symbol == 'RUT':
            return 2000
        elif 'BTC' in symbol:
            return 40000
        elif 'ETH' in symbol:
            return 2500
        elif 'XAU' in symbol:
            return 1800
        else:
            return 100
    
    def _calculate_synthetic_iv(self, price, strike, days_to_expiry):
        """Calculate synthetic implied volatility"""
        moneyness = abs(1 - strike/price)
        time_factor = math.sqrt(max(1, days_to_expiry) / 30)
        
        iv = 0.2 + moneyness * 0.5 + 0.05 * time_factor
        
        iv += np.random.normal(0, 0.02)
        
        return max(0.05, iv)
    
    def _calculate_synthetic_delta(self, price, strike, days_to_expiry, iv, is_call=True):
        """Calculate synthetic delta"""
        moneyness = price / strike
        time_factor = math.sqrt(max(1, days_to_expiry) / 365)
        
        if is_call:
            delta = 0.5 + 0.5 * math.tanh((moneyness - 1) / (iv * time_factor))
        else:
            delta = -0.5 - 0.5 * math.tanh((moneyness - 1) / (iv * time_factor))
        
        return max(-1, min(1, delta))
    
    def _calculate_synthetic_gamma(self, price, strike, days_to_expiry, iv):
        """Calculate synthetic gamma"""
        moneyness = price / strike
        time_factor = math.sqrt(max(1, days_to_expiry) / 365)
        
        gamma = math.exp(-((math.log(moneyness)) ** 2) / (2 * (iv * time_factor) ** 2)) / (price * iv * time_factor * math.sqrt(2 * math.pi))
        
        return gamma
    
    def calculate_dealer_gamma_exposure(self, symbol="SPX"):
        """
        Calculate dealer gamma exposure
        
        Parameters:
        - symbol: Trading symbol (default: SPX)
        
        Returns:
        - Dealer gamma exposure in dollars
        """
        self.fetch_options_data(symbol)
        
        return self.gamma_exposure.get(symbol, 0)
    
    def analyze_gamma_hedging(self, symbol="SPX", current_price=None):
        """
        Analyze gamma hedging patterns for trading signals
        
        Parameters:
        - symbol: Trading symbol (default: SPX)
        - current_price: Current market price (optional)
        
        Returns:
        - Dictionary with signal information
        """
        options_data = self.fetch_options_data(symbol)
        
        current_price = current_price or options_data["price"]
        
        gamma_exposure = self.gamma_exposure.get(symbol, 0)
        
        now = datetime.now()
        closest_expiry = min(options_data["expiries"], 
                            key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - now).days))
        
        options_chain = options_data["options_chains"][closest_expiry]
        
        call_oi_by_strike = {opt["strike"]: opt["call"]["open_interest"] for opt in options_chain}
        put_oi_by_strike = {opt["strike"]: opt["put"]["open_interest"] for opt in options_chain}
        
        top_call_strikes = sorted(call_oi_by_strike.keys(), key=lambda k: call_oi_by_strike[k], reverse=True)[:3]
        top_put_strikes = sorted(put_oi_by_strike.keys(), key=lambda k: put_oi_by_strike[k], reverse=True)[:3]
        
        if gamma_exposure < self.hedging_thresholds["extreme"]:
            direction = "BUY" if current_price < options_data["price"] * 0.99 else "SELL"
            confidence = min(0.9, abs(gamma_exposure / self.hedging_thresholds["extreme"]))
            message = "Extreme negative gamma exposure: dealers will amplify market moves"
        elif gamma_exposure < self.hedging_thresholds["low"]:
            direction = "BUY" if current_price < options_data["price"] * 0.995 else "SELL"
            confidence = min(0.7, abs(gamma_exposure / self.hedging_thresholds["low"]))
            message = "Significant negative gamma exposure: dealers likely to amplify market moves"
        elif gamma_exposure > abs(self.hedging_thresholds["low"]):
            direction = "SELL" if current_price < options_data["price"] * 0.99 else "BUY"
            confidence = min(0.6, gamma_exposure / abs(self.hedging_thresholds["low"]))
            message = "Positive gamma exposure: dealers will dampen market volatility"
        else:
            direction = None
            confidence = 0
            message = "Neutral gamma exposure: limited dealer hedging impact"
        
        near_major_call = any(abs(current_price - strike) / current_price < 0.01 for strike in top_call_strikes)
        near_major_put = any(abs(current_price - strike) / current_price < 0.01 for strike in top_put_strikes)
        
        if near_major_call or near_major_put:
            message += ". Price near major OI strike: potential gamma flip level"
            confidence *= 0.8
        
        return {
            "direction": direction,
            "confidence": confidence,
            "message": message,
            "gamma_exposure": gamma_exposure,
            "gamma_exposure_normalized": gamma_exposure / abs(self.hedging_thresholds["extreme"]),
            "top_call_strikes": top_call_strikes,
            "top_put_strikes": top_put_strikes,
            "near_gamma_flip": near_major_call or near_major_put
        }
    
    def get_gamma_report(self, symbol="SPX"):
        """
        Generate detailed gamma report for a symbol
        
        Parameters:
        - symbol: Trading symbol (default: SPX)
        
        Returns:
        - Dictionary with detailed gamma metrics
        """
        options_data = self.fetch_options_data(symbol)
        
        gamma_by_expiry = {}
        for expiry, chain in options_data["options_chains"].items():
            expiry_gamma = sum(opt["call"]["gamma_exposure"] + opt["put"]["gamma_exposure"] for opt in chain)
            gamma_by_expiry[expiry] = expiry_gamma
        
        nearest_expiry = min(options_data["expiries"], 
                           key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - datetime.now()).days))
        
        gamma_by_strike = {}
        for opt in options_data["options_chains"][nearest_expiry]:
            strike = opt["strike"]
            gamma_by_strike[strike] = opt["call"]["gamma_exposure"] + opt["put"]["gamma_exposure"]
        
        report = {
            "symbol": symbol,
            "price": options_data["price"],
            "timestamp": options_data["timestamp"],
            "total_gamma_exposure": self.gamma_exposure.get(symbol, 0),
            "gamma_exposure_normalized": self.gamma_exposure.get(symbol, 0) / abs(self.hedging_thresholds["extreme"]),
            "gamma_by_expiry": gamma_by_expiry,
            "gamma_by_strike": gamma_by_strike,
            "nearest_expiry": nearest_expiry,
            "hedging_thresholds": self.hedging_thresholds,
            "analysis": self.analyze_gamma_hedging(symbol)
        }
        
        return report
