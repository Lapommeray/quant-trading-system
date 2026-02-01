"""
Institutional Risk Management System

Advanced risk management with:
- Dynamic Kelly Criterion position sizing
- Monte Carlo drawdown simulation
- Circuit breakers for intraday loss, correlation spikes, liquidity drops
- Volatility targeting
- Risk parity allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
from datetime import datetime, timedelta
import threading

logger = logging.getLogger("InstitutionalRiskManager")


class RiskLevel(Enum):
    """Risk level classifications"""
    MINIMAL = "minimal"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: datetime
    portfolio_var: float
    portfolio_cvar: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    overall_risk_level: RiskLevel
    
    
@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    recommended_size: float
    kelly_fraction: float
    volatility_adjusted_size: float
    risk_parity_weight: float
    max_allowed_size: float
    reasoning: str


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status"""
    triggered: bool
    trigger_reason: str
    trigger_time: Optional[datetime]
    cooldown_remaining: int
    can_trade: bool


class KellyCriterionCalculator:
    """
    Dynamic Kelly Criterion position sizing.
    
    Calculates optimal position size based on:
    - Win rate
    - Average win/loss ratio
    - Confidence adjustment
    - Fractional Kelly for safety
    """
    
    def __init__(self, 
                 kelly_fraction: float = 0.25,
                 min_trades_required: int = 30,
                 lookback_trades: int = 100):
        self.kelly_fraction = kelly_fraction
        self.min_trades_required = min_trades_required
        self.lookback_trades = lookback_trades
        self.trade_history: deque = deque(maxlen=lookback_trades)
        
    def record_trade(self, pnl: float, is_win: bool):
        """Record a trade result"""
        self.trade_history.append({
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.now()
        })
        
    def calculate_kelly(self, win_rate: Optional[float] = None,
                       win_loss_ratio: Optional[float] = None) -> Tuple[float, Dict]:
        """
        Calculate Kelly Criterion optimal fraction.
        
        Kelly % = W - [(1-W) / R]
        Where:
        - W = Win rate
        - R = Win/Loss ratio (average win / average loss)
        """
        if len(self.trade_history) < self.min_trades_required:
            return 0.0, {"reason": "Insufficient trade history"}
            
        trades = list(self.trade_history)
        
        if win_rate is None:
            wins = [t for t in trades if t["is_win"]]
            win_rate = len(wins) / len(trades)
            
        if win_loss_ratio is None:
            wins = [t["pnl"] for t in trades if t["is_win"]]
            losses = [abs(t["pnl"]) for t in trades if not t["is_win"]]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 1
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
        if win_loss_ratio <= 0:
            return 0.0, {"reason": "Non-positive win/loss ratio"}
            
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        kelly = max(0, kelly)
        
        fractional_kelly = kelly * self.kelly_fraction
        
        return fractional_kelly, {
            "full_kelly": kelly,
            "fractional_kelly": fractional_kelly,
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "trades_analyzed": len(trades)
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for drawdown and risk analysis.
    
    Simulates thousands of potential paths to estimate:
    - Probability of various drawdown levels
    - Expected maximum drawdown
    - Value at Risk (VaR)
    - Conditional VaR (CVaR/Expected Shortfall)
    """
    
    def __init__(self, 
                 num_simulations: int = 10000,
                 simulation_days: int = 252):
        self.num_simulations = num_simulations
        self.simulation_days = simulation_days
        
    def simulate_paths(self, 
                      mean_return: float,
                      volatility: float,
                      initial_value: float = 100000) -> np.ndarray:
        """Simulate price paths using geometric Brownian motion"""
        dt = 1 / 252
        
        random_shocks = np.random.normal(
            0, 1, 
            (self.num_simulations, self.simulation_days)
        )
        
        daily_returns = (mean_return - 0.5 * volatility**2) * dt + \
                       volatility * np.sqrt(dt) * random_shocks
        
        paths = initial_value * np.exp(np.cumsum(daily_returns, axis=1))
        
        paths = np.column_stack([
            np.full(self.num_simulations, initial_value),
            paths
        ])
        
        return paths
        
    def calculate_drawdowns(self, paths: np.ndarray) -> np.ndarray:
        """Calculate drawdowns for all simulated paths"""
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        return drawdowns
        
    def simulate_drawdown_distribution(self,
                                       mean_return: float,
                                       volatility: float) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation and analyze drawdown distribution.
        """
        paths = self.simulate_paths(mean_return, volatility)
        drawdowns = self.calculate_drawdowns(paths)
        
        max_drawdowns = np.max(drawdowns, axis=1)
        
        percentiles = [50, 75, 90, 95, 99]
        dd_percentiles = {
            f"p{p}": float(np.percentile(max_drawdowns, p))
            for p in percentiles
        }
        
        final_values = paths[:, -1]
        initial_value = paths[0, 0]
        
        var_95 = float(np.percentile(final_values, 5))
        var_99 = float(np.percentile(final_values, 1))
        
        cvar_95 = float(np.mean(final_values[final_values <= var_95]))
        cvar_99 = float(np.mean(final_values[final_values <= var_99]))
        
        prob_20_dd = float(np.mean(max_drawdowns >= 0.20))
        prob_30_dd = float(np.mean(max_drawdowns >= 0.30))
        prob_50_dd = float(np.mean(max_drawdowns >= 0.50))
        
        return {
            "expected_max_drawdown": float(np.mean(max_drawdowns)),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "drawdown_percentiles": dd_percentiles,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "prob_20_drawdown": prob_20_dd,
            "prob_30_drawdown": prob_30_dd,
            "prob_50_drawdown": prob_50_dd,
            "num_simulations": self.num_simulations,
            "simulation_days": self.simulation_days
        }


class CircuitBreakerSystem:
    """
    Multi-level circuit breaker system.
    
    Monitors and triggers on:
    - Intraday loss limits
    - Correlation spikes
    - Liquidity drops
    - Volatility explosions
    - Consecutive losses
    """
    
    def __init__(self,
                 max_daily_loss_pct: float = 0.03,
                 max_weekly_loss_pct: float = 0.05,
                 max_drawdown_pct: float = 0.10,
                 max_correlation: float = 0.85,
                 min_liquidity_ratio: float = 0.5,
                 max_consecutive_losses: int = 5,
                 cooldown_minutes: int = 60):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_correlation = max_correlation
        self.min_liquidity_ratio = min_liquidity_ratio
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.consecutive_losses = 0
        
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time: Optional[datetime] = None
        
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().date()
        
        self._lock = threading.Lock()
        
    def update_equity(self, equity: float):
        """Update current equity level"""
        with self._lock:
            self.current_equity = equity
            if equity > self.peak_equity:
                self.peak_equity = equity
                
    def record_trade_result(self, pnl: float, is_win: bool):
        """Record trade result for circuit breaker monitoring"""
        with self._lock:
            self._check_date_reset()
            
            self.daily_pnl += pnl
            self.weekly_pnl += pnl
            
            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                
    def _check_date_reset(self):
        """Reset daily/weekly counters if needed"""
        today = datetime.now().date()
        
        if today != self.last_daily_reset:
            self.daily_pnl = 0.0
            self.last_daily_reset = today
            
        days_since_weekly = (today - self.last_weekly_reset).days
        if days_since_weekly >= 7:
            self.weekly_pnl = 0.0
            self.last_weekly_reset = today
            
    def check_all_breakers(self, 
                          correlation_matrix: Optional[np.ndarray] = None,
                          liquidity_ratio: float = 1.0) -> CircuitBreakerStatus:
        """Check all circuit breakers"""
        with self._lock:
            if self.triggered:
                if self._check_cooldown():
                    self._reset()
                else:
                    return CircuitBreakerStatus(
                        triggered=True,
                        trigger_reason=self.trigger_reason,
                        trigger_time=self.trigger_time,
                        cooldown_remaining=self._get_cooldown_remaining(),
                        can_trade=False
                    )
                    
            if self.current_equity > 0:
                daily_loss_pct = abs(min(0, self.daily_pnl)) / self.current_equity
                if daily_loss_pct >= self.max_daily_loss_pct:
                    return self._trigger(f"Daily loss limit: {daily_loss_pct:.2%}")
                    
                weekly_loss_pct = abs(min(0, self.weekly_pnl)) / self.current_equity
                if weekly_loss_pct >= self.max_weekly_loss_pct:
                    return self._trigger(f"Weekly loss limit: {weekly_loss_pct:.2%}")
                    
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                if drawdown >= self.max_drawdown_pct:
                    return self._trigger(f"Max drawdown: {drawdown:.2%}")
                    
            if self.consecutive_losses >= self.max_consecutive_losses:
                return self._trigger(f"Consecutive losses: {self.consecutive_losses}")
                
            if correlation_matrix is not None:
                upper_tri = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                max_corr = np.max(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
                if max_corr >= self.max_correlation:
                    return self._trigger(f"Correlation spike: {max_corr:.2f}")
                    
            if liquidity_ratio < self.min_liquidity_ratio:
                return self._trigger(f"Liquidity drop: {liquidity_ratio:.2f}")
                
            return CircuitBreakerStatus(
                triggered=False,
                trigger_reason="",
                trigger_time=None,
                cooldown_remaining=0,
                can_trade=True
            )
            
    def _trigger(self, reason: str) -> CircuitBreakerStatus:
        """Trigger circuit breaker"""
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()
        
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")
        
        return CircuitBreakerStatus(
            triggered=True,
            trigger_reason=reason,
            trigger_time=self.trigger_time,
            cooldown_remaining=self.cooldown_minutes * 60,
            can_trade=False
        )
        
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed"""
        if self.trigger_time is None:
            return True
        elapsed = (datetime.now() - self.trigger_time).total_seconds()
        return elapsed >= self.cooldown_minutes * 60
        
    def _get_cooldown_remaining(self) -> int:
        """Get remaining cooldown in seconds"""
        if self.trigger_time is None:
            return 0
        elapsed = (datetime.now() - self.trigger_time).total_seconds()
        remaining = self.cooldown_minutes * 60 - elapsed
        return max(0, int(remaining))
        
    def _reset(self):
        """Reset circuit breaker"""
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_time = None
        self.consecutive_losses = 0
        logger.info("Circuit breaker reset")
        
    def force_reset(self, override_code: str = ""):
        """Force reset circuit breaker (requires override code)"""
        if override_code == "FORCE_RESET_CONFIRMED":
            self._reset()
            return True
        return False


class VolatilityTargeting:
    """
    Volatility targeting position sizing.
    
    Adjusts position sizes to maintain constant portfolio volatility.
    """
    
    def __init__(self, 
                 target_volatility: float = 0.15,
                 lookback_days: int = 20,
                 max_leverage: float = 2.0):
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.max_leverage = max_leverage
        
    def calculate_position_scalar(self, 
                                 realized_volatility: float) -> float:
        """Calculate position scalar to achieve target volatility"""
        if realized_volatility <= 0:
            return 1.0
            
        scalar = self.target_volatility / realized_volatility
        
        scalar = min(scalar, self.max_leverage)
        scalar = max(scalar, 0.1)
        
        return scalar
        
    def calculate_weights(self,
                         returns_matrix: np.ndarray,
                         current_weights: np.ndarray) -> np.ndarray:
        """Calculate volatility-targeted weights"""
        if returns_matrix.shape[0] < self.lookback_days:
            return current_weights
            
        recent_returns = returns_matrix[-self.lookback_days:]
        
        asset_vols = np.std(recent_returns, axis=0) * np.sqrt(252)
        
        portfolio_returns = recent_returns @ current_weights
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        
        scalar = self.calculate_position_scalar(portfolio_vol)
        
        adjusted_weights = current_weights * scalar
        
        if np.sum(np.abs(adjusted_weights)) > self.max_leverage:
            adjusted_weights = adjusted_weights / np.sum(np.abs(adjusted_weights)) * self.max_leverage
            
        return adjusted_weights


class RiskParityAllocator:
    """
    Risk parity allocation.
    
    Allocates capital so each asset contributes equally to portfolio risk.
    """
    
    def __init__(self, target_volatility: float = 0.10):
        self.target_volatility = target_volatility
        
    def calculate_weights(self, 
                         covariance_matrix: np.ndarray,
                         current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate risk parity weights"""
        n_assets = covariance_matrix.shape[0]
        
        if current_weights is None:
            weights = np.ones(n_assets) / n_assets
        else:
            weights = current_weights.copy()
            
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
            
            if portfolio_vol == 0:
                return np.ones(n_assets) / n_assets
                
            marginal_risk = covariance_matrix @ weights / portfolio_vol
            
            risk_contribution = weights * marginal_risk
            
            target_risk = portfolio_vol / n_assets
            
            adjustment = target_risk / (risk_contribution + 1e-10)
            weights = weights * adjustment
            
            weights = weights / np.sum(weights)
            
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        if portfolio_vol > 0:
            weights = weights * (self.target_volatility / portfolio_vol)
            
        return weights


class InstitutionalRiskManager:
    """
    Main institutional risk management system.
    
    Integrates all risk management components:
    - Kelly Criterion sizing
    - Monte Carlo simulation
    - Circuit breakers
    - Volatility targeting
    - Risk parity
    """
    
    def __init__(self,
                 base_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.10,
                 target_volatility: float = 0.15):
        self.base_risk_per_trade = base_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.target_volatility = target_volatility
        
        self.kelly_calculator = KellyCriterionCalculator()
        self.monte_carlo = MonteCarloSimulator()
        self.circuit_breaker = CircuitBreakerSystem()
        self.vol_targeting = VolatilityTargeting(target_volatility=target_volatility)
        self.risk_parity = RiskParityAllocator(target_volatility=target_volatility)
        
        self.risk_history: deque = deque(maxlen=1000)
        self.current_metrics: Optional[RiskMetrics] = None
        
    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               account_balance: float,
                               realized_volatility: float,
                               win_rate: Optional[float] = None,
                               win_loss_ratio: Optional[float] = None) -> PositionSizeResult:
        """
        Calculate optimal position size using multiple methods.
        """
        breaker_status = self.circuit_breaker.check_all_breakers()
        if not breaker_status.can_trade:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=0.0,
                volatility_adjusted_size=0.0,
                risk_parity_weight=0.0,
                max_allowed_size=0.0,
                reasoning=f"Circuit breaker active: {breaker_status.trigger_reason}"
            )
            
        kelly_fraction, kelly_info = self.kelly_calculator.calculate_kelly(
            win_rate, win_loss_ratio
        )
        
        vol_scalar = self.vol_targeting.calculate_position_scalar(realized_volatility)
        
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return PositionSizeResult(
                recommended_size=0.0,
                kelly_fraction=kelly_fraction,
                volatility_adjusted_size=0.0,
                risk_parity_weight=0.0,
                max_allowed_size=0.0,
                reasoning="Invalid stop loss (same as entry)"
            )
            
        base_risk_amount = account_balance * self.base_risk_per_trade
        base_size = base_risk_amount / risk_per_unit
        
        if kelly_fraction > 0:
            kelly_risk_amount = account_balance * kelly_fraction
            kelly_size = kelly_risk_amount / risk_per_unit
        else:
            kelly_size = base_size
            
        vol_adjusted_size = base_size * vol_scalar
        
        max_size = account_balance * self.max_portfolio_risk / risk_per_unit
        
        recommended_size = min(
            base_size,
            kelly_size,
            vol_adjusted_size,
            max_size
        )
        
        return PositionSizeResult(
            recommended_size=recommended_size,
            kelly_fraction=kelly_fraction,
            volatility_adjusted_size=vol_adjusted_size,
            risk_parity_weight=0.0,
            max_allowed_size=max_size,
            reasoning=f"Kelly: {kelly_fraction:.2%}, Vol scalar: {vol_scalar:.2f}"
        )
        
    def run_risk_simulation(self,
                           mean_return: float,
                           volatility: float) -> Dict[str, Any]:
        """Run Monte Carlo risk simulation"""
        return self.monte_carlo.simulate_drawdown_distribution(mean_return, volatility)
        
    def calculate_portfolio_risk(self,
                                positions: Dict[str, float],
                                returns_matrix: np.ndarray,
                                symbols: List[str]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if returns_matrix.shape[0] < 20:
            return self._default_metrics()
            
        weights = np.array([positions.get(s, 0) for s in symbols])
        total_weight = np.sum(np.abs(weights))
        if total_weight > 0:
            weights = weights / total_weight
            
        portfolio_returns = returns_matrix @ weights
        
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        mean_return = np.mean(portfolio_returns) * 252
        
        sharpe = mean_return / portfolio_vol if portfolio_vol > 0 else 0
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else portfolio_vol
        sortino = mean_return / downside_vol if downside_vol > 0 else 0
        
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)
        current_drawdown = drawdowns[-1]
        
        calmar = mean_return / max_drawdown if max_drawdown > 0 else 0
        
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        corr_matrix = np.corrcoef(returns_matrix.T)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        correlation_risk = np.mean(np.abs(upper_tri)) if len(upper_tri) > 0 else 0
        
        concentration_risk = np.sum(weights ** 2)
        
        if max_drawdown >= 0.20 or portfolio_vol >= 0.40:
            risk_level = RiskLevel.CRITICAL
        elif max_drawdown >= 0.15 or portfolio_vol >= 0.30:
            risk_level = RiskLevel.HIGH
        elif max_drawdown >= 0.10 or portfolio_vol >= 0.20:
            risk_level = RiskLevel.ELEVATED
        elif max_drawdown >= 0.05:
            risk_level = RiskLevel.NORMAL
        elif max_drawdown >= 0.02:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
            
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_var=float(var_95),
            portfolio_cvar=float(cvar_95),
            current_drawdown=float(current_drawdown),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            correlation_risk=float(correlation_risk),
            liquidity_risk=0.0,
            concentration_risk=float(concentration_risk),
            overall_risk_level=risk_level
        )
        
        self.current_metrics = metrics
        self.risk_history.append(metrics)
        
        return metrics
        
    def _default_metrics(self) -> RiskMetrics:
        """Return default metrics when insufficient data"""
        return RiskMetrics(
            timestamp=datetime.now(),
            portfolio_var=0.0,
            portfolio_cvar=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
            concentration_risk=0.0,
            overall_risk_level=RiskLevel.NORMAL
        )
        
    def record_trade(self, pnl: float, is_win: bool):
        """Record trade for risk tracking"""
        self.kelly_calculator.record_trade(pnl, is_win)
        self.circuit_breaker.record_trade_result(pnl, is_win)
        
    def update_equity(self, equity: float):
        """Update current equity"""
        self.circuit_breaker.update_equity(equity)
        
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        breaker_status = self.circuit_breaker.check_all_breakers()
        
        return {
            "can_trade": breaker_status.can_trade,
            "circuit_breaker_triggered": breaker_status.triggered,
            "trigger_reason": breaker_status.trigger_reason,
            "cooldown_remaining": breaker_status.cooldown_remaining,
            "current_metrics": self.current_metrics.__dict__ if self.current_metrics else None,
            "risk_level": self.current_metrics.overall_risk_level.value if self.current_metrics else "unknown"
        }


def main():
    """Demo of institutional risk management"""
    risk_manager = InstitutionalRiskManager(
        base_risk_per_trade=0.02,
        max_portfolio_risk=0.10,
        target_volatility=0.15
    )
    
    np.random.seed(42)
    for i in range(50):
        pnl = np.random.normal(100, 500)
        is_win = pnl > 0
        risk_manager.record_trade(pnl, is_win)
        
    risk_manager.update_equity(100000)
    
    print("=== Position Sizing ===")
    size_result = risk_manager.calculate_position_size(
        symbol="BTCUSD",
        entry_price=45000,
        stop_loss=44000,
        account_balance=100000,
        realized_volatility=0.60
    )
    print(f"Recommended Size: {size_result.recommended_size:.4f}")
    print(f"Kelly Fraction: {size_result.kelly_fraction:.2%}")
    print(f"Vol Adjusted Size: {size_result.volatility_adjusted_size:.4f}")
    print(f"Reasoning: {size_result.reasoning}")
    
    print("\n=== Monte Carlo Simulation ===")
    mc_results = risk_manager.run_risk_simulation(
        mean_return=0.10,
        volatility=0.20
    )
    print(f"Expected Max Drawdown: {mc_results['expected_max_drawdown']:.2%}")
    print(f"VaR 95%: ${mc_results['var_95']:,.0f}")
    print(f"CVaR 95%: ${mc_results['cvar_95']:,.0f}")
    print(f"Prob of 20% DD: {mc_results['prob_20_drawdown']:.2%}")
    
    print("\n=== Portfolio Risk ===")
    returns = np.random.normal(0.0005, 0.02, (252, 3))
    positions = {"BTCUSD": 0.4, "ETHUSD": 0.3, "EURUSD": 0.3}
    symbols = ["BTCUSD", "ETHUSD", "EURUSD"]
    
    metrics = risk_manager.calculate_portfolio_risk(positions, returns, symbols)
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Risk Level: {metrics.overall_risk_level.value}")
    
    print("\n=== Risk Status ===")
    status = risk_manager.get_risk_status()
    print(f"Can Trade: {status['can_trade']}")
    print(f"Circuit Breaker: {status['circuit_breaker_triggered']}")


if __name__ == "__main__":
    main()
