"""
Metrics - Performance Analytics Module

Computes trading performance metrics for live monitoring and backtest evaluation.

Metrics:
- Sharpe ratio (annualized)
- Sortino ratio (downside risk only)
- Win rate and profit factor
- Precision/recall for directional signals
- Max drawdown
- Average trade duration
- Calmar ratio

Pure Python/numpy. No external dependencies.
"""

import logging
import math
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger("Metrics")


class PerformanceMetrics:
    """
    Streaming performance calculator.

    Can be fed trades one-by-one (live) or given a batch (backtest).
    All metrics update incrementally.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self._returns: List[float] = []
        self._equity_curve: List[float] = []
        self._trades: List[Dict[str, Any]] = []
        self._predictions: List[Dict[str, str]] = []

    def record_return(self, pct_return: float):
        self._returns.append(pct_return)

    def record_equity(self, equity: float):
        self._equity_curve.append(equity)

    def record_trade(self, trade: Dict[str, Any]):
        self._trades.append(trade)

    def record_prediction(self, predicted: str, actual: str):
        self._predictions.append({"predicted": predicted, "actual": actual})

    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        if len(self._returns) < 5:
            return 0.0
        arr = np.array(self._returns)
        mean_r = np.mean(arr)
        std_r = np.std(arr)
        if std_r == 0:
            return 0.0
        daily_rf = self.risk_free_rate / periods_per_year
        return float((mean_r - daily_rf) / std_r * math.sqrt(periods_per_year))

    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        if len(self._returns) < 5:
            return 0.0
        arr = np.array(self._returns)
        mean_r = np.mean(arr)
        downside = arr[arr < 0]
        if len(downside) == 0:
            return float("inf") if mean_r > 0 else 0.0
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0.0
        daily_rf = self.risk_free_rate / periods_per_year
        return float((mean_r - daily_rf) / downside_std * math.sqrt(periods_per_year))

    def max_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        arr = np.array(self._equity_curve)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / peak
        return float(np.max(dd))

    def calmar_ratio(self, periods_per_year: int = 252) -> float:
        mdd = self.max_drawdown()
        if mdd == 0 or len(self._returns) < 5:
            return 0.0
        annual_return = np.mean(self._returns) * periods_per_year
        return float(annual_return / mdd)

    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        return wins / len(self._trades)

    def profit_factor(self) -> float:
        gross_profit = sum(t["pnl"] for t in self._trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self._trades if t.get("pnl", 0) < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def precision(self, direction: str = "BUY") -> float:
        if not self._predictions:
            return 0.0
        tp = sum(
            1 for p in self._predictions
            if p["predicted"] == direction and p["actual"] == direction
        )
        fp = sum(
            1 for p in self._predictions
            if p["predicted"] == direction and p["actual"] != direction
        )
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, direction: str = "BUY") -> float:
        if not self._predictions:
            return 0.0
        tp = sum(
            1 for p in self._predictions
            if p["predicted"] == direction and p["actual"] == direction
        )
        fn = sum(
            1 for p in self._predictions
            if p["predicted"] != direction and p["actual"] == direction
        )
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1_score(self, direction: str = "BUY") -> float:
        p = self.precision(direction)
        r = self.recall(direction)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def expectancy(self) -> float:
        if not self._trades:
            return 0.0
        wr = self.win_rate()
        wins = [t["pnl"] for t in self._trades if t.get("pnl", 0) > 0]
        losses = [abs(t["pnl"]) for t in self._trades if t.get("pnl", 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        return float(wr * avg_win - (1 - wr) * avg_loss)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_trades": len(self._trades),
            "total_returns": len(self._returns),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "sortino_ratio": round(self.sortino_ratio(), 4),
            "calmar_ratio": round(self.calmar_ratio(), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "win_rate": round(self.win_rate(), 4),
            "profit_factor": round(self.profit_factor(), 4),
            "expectancy": round(self.expectancy(), 4),
            "buy_precision": round(self.precision("BUY"), 4),
            "buy_recall": round(self.recall("BUY"), 4),
            "buy_f1": round(self.f1_score("BUY"), 4),
            "sell_precision": round(self.precision("SELL"), 4),
            "sell_recall": round(self.recall("SELL"), 4),
            "sell_f1": round(self.f1_score("SELL"), 4),
        }

    def reset(self):
        self._returns = []
        self._equity_curve = []
        self._trades = []
        self._predictions = []
