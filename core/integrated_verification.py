"""
Integrated Verification System
Combines all verification modules into a unified system
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

from .dark_pool_mapper import DarkPoolMapper
from .gamma_trap import GammaTrap
from .retail_sentiment import RetailSentimentAnalyzer
from .alpha_equation import AlphaEquation
from .order_book_reconstruction import OrderBookReconstructor
from .fill_engine import FillEngine

class IntegratedVerification:
    def __init__(self):
        """Initialize Integrated Verification System"""
        self.dark_pool = DarkPoolMapper()
        self.gamma_trap = GammaTrap()
        self.sentiment = RetailSentimentAnalyzer()
        self.alpha = AlphaEquation()
        self.order_book = OrderBookReconstructor()
        self.fill_engine = FillEngine(slippage_enabled=True, order_book_simulation=True)
        
        self.modules_enabled = {
            "dark_pool": True,
            "gamma_trap": True,
            "sentiment": True,
            "alpha": True,
            "order_book": True
        }
    
    def analyze_symbol(self, symbol, current_price):
        """
        Perform comprehensive analysis of a symbol
        
        Parameters:
        - symbol: Trading symbol
        - current_price: Current market price
        
        Returns:
        - Dictionary with analysis results
        """
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "signals": {}
        }
        
        if self.modules_enabled["dark_pool"]:
            dark_pool_signal = self.dark_pool.analyze_dark_pool_signal(symbol, current_price)
            results["signals"]["dark_pool"] = dark_pool_signal
        
        if self.modules_enabled["gamma_trap"]:
            gamma_signal = self.gamma_trap.analyze_gamma_hedging(symbol, current_price)
            results["signals"]["gamma_trap"] = gamma_signal
        
        if self.modules_enabled["sentiment"]:
            sentiment_signal = self.sentiment.analyze_sentiment(symbol)
            results["signals"]["sentiment"] = sentiment_signal
        
        if self.modules_enabled["order_book"]:
            self.order_book.update(symbol, current_price)
            liquidity = self.order_book.get_liquidity_metrics()
            results["signals"]["order_book"] = liquidity
        
        combined_signal = self._combine_signals(results["signals"])
        results["combined_signal"] = combined_signal
        
        return results
    
    def _combine_signals(self, signals):
        """
        Combine signals from different modules
        
        Parameters:
        - signals: Dictionary with signals from different modules
        
        Returns:
        - Dictionary with combined signal
        """
        buy_votes = 0
        sell_votes = 0
        buy_confidence = 0
        sell_confidence = 0
        
        if "dark_pool" in signals:
            dark_pool = signals["dark_pool"]
            if dark_pool["direction"] == "BUY":
                buy_votes += 1
                buy_confidence += dark_pool["confidence"]
            elif dark_pool["direction"] == "SELL":
                sell_votes += 1
                sell_confidence += dark_pool["confidence"]
        
        if "gamma_trap" in signals:
            gamma = signals["gamma_trap"]
            if gamma["direction"] == "BUY":
                buy_votes += 1
                buy_confidence += gamma["confidence"]
            elif gamma["direction"] == "SELL":
                sell_votes += 1
                sell_confidence += gamma["confidence"]
        
        if "sentiment" in signals:
            sentiment = signals["sentiment"]
            if sentiment["direction"] == "BUY":
                buy_votes += 1
                buy_confidence += sentiment["confidence"]
            elif sentiment["direction"] == "SELL":
                sell_votes += 1
                sell_confidence += sentiment["confidence"]
        
        if "order_book" in signals:
            order_book = signals["order_book"]
            if order_book["imbalance"] < -0.2:
                buy_votes += 1
                buy_confidence += min(0.7, abs(order_book["imbalance"]))
            elif order_book["imbalance"] > 0.2:
                sell_votes += 1
                sell_confidence += min(0.7, abs(order_book["imbalance"]))
        
        total_votes = buy_votes + sell_votes
        
        if total_votes == 0:
            direction = None
            confidence = 0
            message = "No clear signal"
        elif buy_votes > sell_votes:
            direction = "BUY"
            confidence = buy_confidence / buy_votes
            message = f"Buy signal with {buy_votes}/{total_votes} votes"
        elif sell_votes > buy_votes:
            direction = "SELL"
            confidence = sell_confidence / sell_votes
            message = f"Sell signal with {sell_votes}/{total_votes} votes"
        else:
            if buy_confidence > sell_confidence:
                direction = "BUY"
                confidence = buy_confidence / buy_votes
                message = f"Buy signal (tiebreaker) with {buy_votes}/{total_votes} votes"
            elif sell_confidence > buy_confidence:
                direction = "SELL"
                confidence = sell_confidence / sell_votes
                message = f"Sell signal (tiebreaker) with {sell_votes}/{total_votes} votes"
            else:
                direction = None
                confidence = 0
                message = "Perfectly balanced signals - no clear direction"
        
        return {
            "direction": direction,
            "confidence": confidence,
            "message": message,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "buy_confidence": buy_confidence,
            "sell_confidence": sell_confidence
        }
    
    def execute_trade(self, symbol, direction, price, volume):
        """
        Execute a trade with realistic fill simulation
        
        Parameters:
        - symbol: Trading symbol
        - direction: Trade direction ("BUY" or "SELL")
        - price: Current market price
        - volume: Trading volume
        
        Returns:
        - Dictionary with trade execution details
        """
        self.order_book.update(symbol, price)
        
        impact = self.order_book.calculate_market_impact(direction, volume)
        
        fill = self.fill_engine.execute_order(symbol, direction, price, volume)
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "requested_price": price,
            "fill_price": fill["fill_price"],
            "volume": volume,
            "slippage_bps": fill["slippage_bps"],
            "market_impact_bps": impact["market_impact_bps"],
            "total_cost_bps": fill["slippage_bps"] + impact["market_impact_bps"],
            "latency_ms": fill["latency_ms"]
        }
        
        return result
    
    def generate_verification_report(self, trades_df, symbol, output_dir="./reports"):
        """
        Generate comprehensive verification report
        
        Parameters:
        - trades_df: DataFrame with trade data
        - symbol: Trading symbol
        - output_dir: Output directory for reports
        
        Returns:
        - Dictionary with report paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        reports = {}
        
        if self.modules_enabled["alpha"]:
            alpha_path = os.path.join(output_dir, f"{symbol}_alpha_report.json")
            alpha_report = self.alpha.generate_alpha_report(trades_df, alpha_path)
            reports["alpha"] = alpha_path
        
        if self.modules_enabled["dark_pool"]:
            dark_pool_path = os.path.join(output_dir, f"{symbol}_dark_pool_report.json")
            dark_pool_report = self.dark_pool.get_dark_pool_report(symbol)
            with open(dark_pool_path, 'w') as f:
                json.dump(dark_pool_report, f, indent=4)
            reports["dark_pool"] = dark_pool_path
        
        if self.modules_enabled["gamma_trap"]:
            gamma_path = os.path.join(output_dir, f"{symbol}_gamma_report.json")
            gamma_report = self.gamma_trap.get_gamma_report(symbol)
            with open(gamma_path, 'w') as f:
                json.dump(gamma_report, f, indent=4)
            reports["gamma"] = gamma_path
        
        if self.modules_enabled["sentiment"]:
            sentiment_path = os.path.join(output_dir, f"{symbol}_sentiment_report.json")
            sentiment_report = self.sentiment.get_sentiment_report(symbol)
            with open(sentiment_path, 'w') as f:
                json.dump(sentiment_report, f, indent=4)
            reports["sentiment"] = sentiment_path
        
        costs_path = os.path.join(output_dir, f"{symbol}_costs.log")
        self.fill_engine.generate_costs_log(costs_path)
        reports["costs"] = costs_path
        
        trades_path = os.path.join(output_dir, f"{symbol}_trades.csv")
        self.fill_engine.save_trades_csv(trades_path)
        reports["trades"] = trades_path
        
        summary = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "reports": reports,
            "modules_enabled": self.modules_enabled,
            "trade_count": len(trades_df),
            "alpha_summary": alpha_report["overall_alpha"] if self.modules_enabled["alpha"] else None
        }
        
        summary_path = os.path.join(output_dir, f"{symbol}_verification_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Verification reports generated in {output_dir}")
        
        return {
            "summary": summary_path,
            "reports": reports
        }
