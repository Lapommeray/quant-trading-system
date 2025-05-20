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
from .neural_pattern_recognition import NeuralPatternRecognition
from .dark_pool_dna import DarkPoolDNA
from .market_regime_detection import MarketRegimeDetection

class IntegratedVerification:
    def __init__(self):
        """Initialize Integrated Verification System"""
        self.dark_pool = DarkPoolMapper()
        self.gamma_trap = GammaTrap()
        self.sentiment = RetailSentimentAnalyzer()
        self.alpha = AlphaEquation()
        self.order_book = OrderBookReconstructor()
        self.fill_engine = FillEngine(slippage_enabled=True, order_book_simulation=True)
        self.neural_pattern = NeuralPatternRecognition()
        self.dark_pool_dna = DarkPoolDNA()
        self.market_regime = MarketRegimeDetection()
        
        self.modules_enabled = {
            "dark_pool": True,
            "gamma_trap": True,
            "sentiment": True,
            "alpha": True,
            "order_book": True,
            "neural_pattern": True,
            "dark_pool_dna": True,
            "market_regime": True
        }
    
    def analyze_symbol(self, symbol, current_price, high_price=None, low_price=None):
        """
        Perform comprehensive analysis of a symbol
        
        Parameters:
        - symbol: Trading symbol
        - current_price: Current market price
        - high_price: High price (optional, for market regime detection)
        - low_price: Low price (optional, for market regime detection)
        
        Returns:
        - Dictionary with analysis results
        """
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "signals": {}
        }
        
        if self.modules_enabled["market_regime"]:
            self.market_regime.update_price_memory(symbol, current_price, high_price, low_price)
            regime = self.market_regime.get_current_regime(symbol)
            results["market_regime"] = regime
            
            if regime["regime"] == "crisis":
                results["trading_allowed"] = False
                results["trading_message"] = "Trading halted - crisis regime detected"
                return results
        
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
        
        if self.modules_enabled["neural_pattern"]:
            self.neural_pattern.update_price_memory(symbol, current_price)
            neural_signal = self.neural_pattern.analyze_neural_patterns(symbol)
            results["signals"]["neural_pattern"] = neural_signal
        
        if self.modules_enabled["dark_pool_dna"]:
            dna_signal = self.dark_pool_dna.analyze_dna_sequence(symbol, current_price)
            results["signals"]["dark_pool_dna"] = dna_signal
        
        combined_signal = self._combine_signals(results["signals"])
        results["combined_signal"] = combined_signal
        
        # Final trading decision based on market regime and combined signal
        if self.modules_enabled["market_regime"]:
            regime = results["market_regime"]
            if regime["regime"] in ["crisis", "pre_crisis"]:
                if combined_signal["confidence"] > 0.85:
                    results["trading_allowed"] = True
                    results["trading_message"] = f"Trading allowed despite {regime['regime']} regime - very strong signal"
                else:
                    results["trading_allowed"] = False
                    results["trading_message"] = f"Trading halted - {regime['regime']} regime with insufficient signal strength"
            else:
                results["trading_allowed"] = True
                results["trading_message"] = f"Trading allowed - {regime['regime']} regime"
        else:
            results["trading_allowed"] = True
            results["trading_message"] = "Trading allowed - market regime detection disabled"
        
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
        
        if "neural_pattern" in signals:
            neural = signals["neural_pattern"]
            if neural["direction"] == "BUY":
                buy_votes += 1.5
                buy_confidence += neural["confidence"] * 1.5
            elif neural["direction"] == "SELL":
                sell_votes += 1.5
                sell_confidence += neural["confidence"] * 1.5
        
        if "dark_pool_dna" in signals:
            dna = signals["dark_pool_dna"]
            if dna["direction"] == "BUY":
                buy_votes += 1.2
                buy_confidence += dna["confidence"] * 1.2
            elif dna["direction"] == "SELL":
                sell_votes += 1.2
                sell_confidence += dna["confidence"] * 1.2
        
        total_votes = buy_votes + sell_votes
        
        if total_votes == 0:
            direction = None
            confidence = 0
            message = "No clear signal"
        elif buy_votes > sell_votes:
            direction = "BUY"
            confidence = buy_confidence / buy_votes
            message = f"Buy signal with {buy_votes:.1f}/{total_votes:.1f} votes"
        elif sell_votes > buy_votes:
            direction = "SELL"
            confidence = sell_confidence / sell_votes
            message = f"Sell signal with {sell_votes:.1f}/{total_votes:.1f} votes"
        else:
            if buy_confidence > sell_confidence:
                direction = "BUY"
                confidence = buy_confidence / buy_votes
                message = f"Buy signal (tiebreaker) with {buy_votes:.1f}/{total_votes:.1f} votes"
            elif sell_confidence > buy_confidence:
                direction = "SELL"
                confidence = sell_confidence / sell_votes
                message = f"Sell signal (tiebreaker) with {sell_votes:.1f}/{total_votes:.1f} votes"
            else:
                direction = None
                confidence = 0
                message = "Perfectly balanced signals - no clear direction"
        
        signal_count = len([s for s in signals.values() if s.get("direction") is not None])
        signal_agreement = max(buy_votes, sell_votes) / total_votes if total_votes > 0 else 0
        
        return {
            "direction": direction,
            "confidence": confidence,
            "message": message,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "buy_confidence": buy_confidence,
            "sell_confidence": sell_confidence,
            "signal_count": signal_count,
            "signal_agreement": signal_agreement
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
        
        if self.modules_enabled["neural_pattern"]:
            neural_path = os.path.join(output_dir, f"{symbol}_neural_pattern_report.json")
            neural_report = self.neural_pattern.get_neural_report(symbol)
            with open(neural_path, 'w') as f:
                json.dump(neural_report, f, indent=4)
            reports["neural_pattern"] = neural_path
        
        if self.modules_enabled["dark_pool_dna"]:
            dna_path = os.path.join(output_dir, f"{symbol}_dark_pool_dna_report.json")
            dna_report = self.dark_pool_dna.get_dna_report(symbol)
            with open(dna_path, 'w') as f:
                json.dump(dna_report, f, indent=4)
            reports["dark_pool_dna"] = dna_path
        
        if self.modules_enabled["market_regime"]:
            regime_path = os.path.join(output_dir, f"{symbol}_market_regime_report.json")
            regime_report = self.market_regime.get_regime_report(symbol)
            with open(regime_path, 'w') as f:
                json.dump(regime_report, f, indent=4)
            reports["market_regime"] = regime_path
        
        costs_path = os.path.join(output_dir, f"{symbol}_costs.log")
        self.fill_engine.generate_costs_log(costs_path)
        reports["costs"] = costs_path
        
        trades_path = os.path.join(output_dir, f"{symbol}_trades.csv")
        self.fill_engine.save_trades_csv(trades_path)
        reports["trades"] = trades_path
        
        win_rate = None
        profit_factor = None
        max_drawdown = None
        
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades_df)
            
            total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            if 'cumulative_pnl' in trades_df.columns:
                cumulative = trades_df['cumulative_pnl'].values
                peak = np.maximum.accumulate(cumulative)
                drawdown = (peak - cumulative) / peak
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        summary = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "reports": reports,
            "modules_enabled": self.modules_enabled,
            "trade_count": len(trades_df),
            "performance_metrics": {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown
            },
            "alpha_summary": alpha_report["overall_alpha"] if self.modules_enabled["alpha"] else None
        }
        
        summary_path = os.path.join(output_dir, f"{symbol}_verification_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Verification reports generated in {output_dir}")
        print(f"Performance metrics: Win Rate={win_rate:.2%}, Profit Factor={profit_factor:.2f}, Max Drawdown={max_drawdown:.2%}")
        
        return {
            "summary": summary_path,
            "reports": reports,
            "performance_metrics": {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown
            }
        }
