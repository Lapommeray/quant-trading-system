"""
MetaTrader 5 Deployment Script

Handles deployment and configuration of the trading system for MT5 live execution.
Includes symbol mapping, timeframe alignment, error recovery, and monitoring.
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mt5_live_engine import (
    MT5LiveEngine, 
    Order, 
    OrderType, 
    ExecutionMode,
    RiskLimits,
    CircuitBreaker
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deploy_mt5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeployMT5")


@dataclass
class DeploymentConfig:
    """Configuration for MT5 deployment"""
    symbols: List[str]
    timeframes: List[str]
    risk_per_trade: float = 0.02
    max_positions: int = 5
    max_daily_trades: int = 20
    execution_mode: str = "MARKET"
    require_human_confirmation: bool = True
    human_confirmation_trades: int = 100
    enable_circuit_breaker: bool = True
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.05
    paper_trading: bool = True
    signal_confidence_threshold: float = 0.7
    check_interval_seconds: int = 60
    
    @classmethod
    def from_file(cls, filepath: str) -> 'DeploymentConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
        
    def to_file(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class SignalGenerator:
    """Interface to the trading signal generation system"""
    
    def __init__(self, engine: MT5LiveEngine):
        self.engine = engine
        self._load_signal_modules()
        
    def _load_signal_modules(self):
        """Load signal generation modules"""
        try:
            from core.qmp_engine_v3 import QMPUltraEngine
            self.qmp_available = True
        except ImportError:
            self.qmp_available = False
            logger.warning("QMPUltraEngine not available, using basic signals")
            
        try:
            from advanced_modules.mathematical_integration_layer import MathematicalIntegrationLayer
            self.math_layer_available = True
        except ImportError:
            self.math_layer_available = False
            
    def generate_signal(self, symbol: str, timeframes: List[str]) -> Dict[str, Any]:
        """
        Generate trading signal for symbol
        
        Returns:
            Dictionary with signal direction, confidence, and metadata
        """
        history_data = {}
        for tf in timeframes:
            df = self.engine.get_ohlcv(symbol, tf, 100)
            if not df.empty:
                history_data[tf] = df
                
        if not history_data:
            return {"signal": None, "confidence": 0.0, "reason": "No data"}
            
        signals = []
        
        trend_signal = self._analyze_trend(history_data)
        signals.append(trend_signal)
        
        momentum_signal = self._analyze_momentum(history_data)
        signals.append(momentum_signal)
        
        volatility_signal = self._analyze_volatility(history_data)
        signals.append(volatility_signal)
        
        buy_votes = sum(1 for s in signals if s.get("direction") == "BUY")
        sell_votes = sum(1 for s in signals if s.get("direction") == "SELL")
        
        if buy_votes > sell_votes:
            direction = "BUY"
            confidence = buy_votes / len(signals)
        elif sell_votes > buy_votes:
            direction = "SELL"
            confidence = sell_votes / len(signals)
        else:
            direction = None
            confidence = 0.0
            
        avg_confidence = sum(s.get("confidence", 0) for s in signals) / len(signals)
        final_confidence = (confidence + avg_confidence) / 2
        
        return {
            "signal": direction,
            "confidence": final_confidence,
            "components": signals,
            "timestamp": datetime.now().isoformat()
        }
        
    def _analyze_trend(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend using moving averages"""
        import numpy as np
        
        for tf in ['1h', '15m', '5m', '1m']:
            if tf in history_data:
                df = history_data[tf]
                break
        else:
            return {"direction": None, "confidence": 0.0}
            
        closes = df['close'].values
        
        if len(closes) < 50:
            return {"direction": None, "confidence": 0.0}
            
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        current_price = closes[-1]
        
        if current_price > sma_20 > sma_50:
            direction = "BUY"
            confidence = min(1.0, (current_price - sma_50) / sma_50 * 10)
        elif current_price < sma_20 < sma_50:
            direction = "SELL"
            confidence = min(1.0, (sma_50 - current_price) / sma_50 * 10)
        else:
            direction = None
            confidence = 0.3
            
        return {"direction": direction, "confidence": confidence, "type": "trend"}
        
    def _analyze_momentum(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum using RSI"""
        import numpy as np
        
        for tf in ['1h', '15m', '5m', '1m']:
            if tf in history_data:
                df = history_data[tf]
                break
        else:
            return {"direction": None, "confidence": 0.0}
            
        closes = df['close'].values
        
        if len(closes) < 15:
            return {"direction": None, "confidence": 0.0}
            
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        if rsi < 30:
            direction = "BUY"
            confidence = (30 - rsi) / 30
        elif rsi > 70:
            direction = "SELL"
            confidence = (rsi - 70) / 30
        else:
            direction = None
            confidence = 0.3
            
        return {"direction": direction, "confidence": confidence, "type": "momentum", "rsi": rsi}
        
    def _analyze_volatility(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility for position sizing guidance"""
        import numpy as np
        
        for tf in ['1h', '15m', '5m', '1m']:
            if tf in history_data:
                df = history_data[tf]
                break
        else:
            return {"direction": None, "confidence": 0.0}
            
        closes = df['close'].values
        
        if len(closes) < 20:
            return {"direction": None, "confidence": 0.5}
            
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:])
        
        avg_volatility = np.std(returns)
        
        if volatility < avg_volatility * 0.8:
            confidence = 0.7
        elif volatility > avg_volatility * 1.5:
            confidence = 0.3
        else:
            confidence = 0.5
            
        return {
            "direction": None, 
            "confidence": confidence, 
            "type": "volatility",
            "current_vol": volatility,
            "avg_vol": avg_volatility
        }


class MT5Deployer:
    """
    Main deployment class for MT5 live trading
    
    Handles:
    - Connection management
    - Signal generation and execution
    - Risk management
    - Error recovery
    - Monitoring and logging
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.engine: Optional[MT5LiveEngine] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.running = False
        self.daily_trades = 0
        self.last_trade_date = None
        self.error_count = 0
        self.max_errors = 10
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
        
    def initialize(self) -> bool:
        """Initialize the deployment"""
        logger.info("Initializing MT5 deployment...")
        
        risk_limits = RiskLimits(
            max_position_size=0.1,
            max_daily_loss=self.config.max_daily_loss_pct,
            max_drawdown=self.config.max_drawdown_pct,
            max_single_trade_risk=self.config.risk_per_trade,
            require_human_confirmation=self.config.require_human_confirmation,
            human_confirmation_trades=self.config.human_confirmation_trades
        )
        
        self.engine = MT5LiveEngine(
            risk_limits=risk_limits,
            simulation_mode=self.config.paper_trading
        )
        
        if not self.engine.connect():
            logger.error("Failed to connect to MT5")
            return False
            
        self.signal_generator = SignalGenerator(self.engine)
        
        logger.info("MT5 deployment initialized successfully")
        logger.info(f"Paper trading mode: {self.config.paper_trading}")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Timeframes: {self.config.timeframes}")
        
        return True
        
    def start(self):
        """Start the deployment"""
        if not self.engine:
            if not self.initialize():
                return
                
        self.running = True
        logger.info("Starting MT5 deployment...")
        
        self.engine.start_tick_streaming()
        self.engine.start_portfolio_sync()
        
        self._main_loop()
        
    def stop(self):
        """Stop the deployment"""
        logger.info("Stopping MT5 deployment...")
        self.running = False
        
        if self.engine:
            self.engine.disconnect()
            
        logger.info("MT5 deployment stopped")
        
    def _main_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                self._check_daily_reset()
                
                if self.daily_trades >= self.config.max_daily_trades:
                    logger.info("Daily trade limit reached, waiting...")
                    time.sleep(self.config.check_interval_seconds)
                    continue
                    
                positions = self.engine.get_positions()
                if len(positions) >= self.config.max_positions:
                    logger.info("Max positions reached, monitoring only...")
                    self._monitor_positions(positions)
                    time.sleep(self.config.check_interval_seconds)
                    continue
                    
                for symbol in self.config.symbols:
                    if not self.running:
                        break
                        
                    if symbol in positions:
                        continue
                        
                    self._process_symbol(symbol)
                    
                self.error_count = 0
                
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in main loop: {e}")
                
                if self.error_count >= self.max_errors:
                    logger.critical("Too many errors, initiating emergency stop")
                    self.engine.emergency_stop()
                    self.running = False
                    break
                    
                time.sleep(10)
                
    def _check_daily_reset(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
            logger.info("Daily counters reset")
            
    def _process_symbol(self, symbol: str):
        """Process trading signal for a symbol"""
        signal_result = self.signal_generator.generate_signal(
            symbol, 
            self.config.timeframes
        )
        
        if not signal_result.get("signal"):
            return
            
        confidence = signal_result.get("confidence", 0)
        if confidence < self.config.signal_confidence_threshold:
            logger.debug(f"{symbol}: Signal confidence too low ({confidence:.2f})")
            return
            
        direction = signal_result["signal"]
        
        tick = self.engine.get_tick(symbol)
        if not tick:
            logger.warning(f"No tick data for {symbol}")
            return
            
        volume = self._calculate_position_size(symbol, tick)
        
        stop_loss, take_profit = self._calculate_sl_tp(symbol, direction, tick)
        
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET_BUY if direction == "BUY" else OrderType.MARKET_SELL,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"Signal_{confidence:.2f}"
        )
        
        execution_mode = ExecutionMode[self.config.execution_mode]
        
        logger.info(f"Executing {direction} order for {symbol} (confidence: {confidence:.2f})")
        
        result = self.engine.execute_order(order, execution_mode)
        
        if result.get("success"):
            self.daily_trades += 1
            logger.info(f"Order executed: {result}")
        else:
            logger.warning(f"Order failed: {result.get('error')}")
            
    def _calculate_position_size(self, symbol: str, tick) -> float:
        """Calculate position size based on risk parameters"""
        account = self.engine.account_info
        balance = account.get("balance", 10000)
        
        risk_amount = balance * self.config.risk_per_trade
        
        price = tick.ask
        
        if "BTC" in symbol or "ETH" in symbol:
            min_volume = 0.001
            volume = risk_amount / price
        elif "XAU" in symbol:
            min_volume = 0.01
            volume = risk_amount / (price * 100)
        else:
            min_volume = 0.01
            volume = risk_amount / (price * 100000)
            
        volume = max(min_volume, round(volume, 3))
        
        return volume
        
    def _calculate_sl_tp(self, symbol: str, direction: str, tick) -> tuple:
        """Calculate stop loss and take profit levels"""
        price = tick.ask if direction == "BUY" else tick.bid
        
        if "BTC" in symbol:
            sl_distance = price * 0.02
            tp_distance = price * 0.04
        elif "ETH" in symbol:
            sl_distance = price * 0.025
            tp_distance = price * 0.05
        elif "XAU" in symbol:
            sl_distance = price * 0.01
            tp_distance = price * 0.02
        else:
            sl_distance = price * 0.005
            tp_distance = price * 0.01
            
        if direction == "BUY":
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + sl_distance
            take_profit = price - tp_distance
            
        return stop_loss, take_profit
        
    def _monitor_positions(self, positions: Dict):
        """Monitor open positions"""
        for symbol, position in positions.items():
            pnl_pct = position.profit / self.engine.account_info.get("balance", 1) * 100
            
            logger.info(
                f"Position {symbol}: {position.direction} {position.volume} @ {position.entry_price}, "
                f"Current: {position.current_price}, PnL: {position.profit:.2f} ({pnl_pct:.2f}%)"
            )
            
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            "running": self.running,
            "paper_trading": self.config.paper_trading,
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.config.max_daily_trades,
            "error_count": self.error_count,
            "account": self.engine.get_account_summary() if self.engine else None,
            "config": asdict(self.config)
        }


def create_default_config() -> DeploymentConfig:
    """Create default deployment configuration"""
    return DeploymentConfig(
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "EUR/USD",
            "GBP/USD",
            "XAU/USD"
        ],
        timeframes=["1m", "5m", "15m", "1h"],
        risk_per_trade=0.02,
        max_positions=5,
        max_daily_trades=20,
        execution_mode="MARKET",
        require_human_confirmation=True,
        human_confirmation_trades=100,
        enable_circuit_breaker=True,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.05,
        paper_trading=True,
        signal_confidence_threshold=0.7,
        check_interval_seconds=60
    )


def main():
    """Main entry point for MT5 deployment"""
    parser = argparse.ArgumentParser(description="MT5 Trading System Deployment")
    
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols")
    parser.add_argument("--create-config", type=str, help="Create default config file")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config.to_file(args.create_config)
        print(f"Default config created: {args.create_config}")
        return
        
    if args.config and os.path.exists(args.config):
        config = DeploymentConfig.from_file(args.config)
    else:
        config = create_default_config()
        
    if args.paper:
        config.paper_trading = True
    elif args.live:
        config.paper_trading = False
        
    if args.symbols:
        config.symbols = [s.strip() for s in args.symbols.split(",")]
        
    print("=" * 60)
    print("MT5 TRADING SYSTEM DEPLOYMENT")
    print("=" * 60)
    print(f"Mode: {'PAPER TRADING' if config.paper_trading else 'LIVE TRADING'}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Risk per trade: {config.risk_per_trade * 100}%")
    print(f"Max daily loss: {config.max_daily_loss_pct * 100}%")
    print("=" * 60)
    
    if not config.paper_trading:
        print("\n*** WARNING: LIVE TRADING MODE ***")
        print("This will execute real trades with real money.")
        confirm = input("Type 'CONFIRM_LIVE_TRADING' to proceed: ")
        if confirm != "CONFIRM_LIVE_TRADING":
            print("Live trading not confirmed. Exiting.")
            return
            
    deployer = MT5Deployer(config)
    
    try:
        deployer.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        deployer.stop()
        
    print("\nDeployment complete.")


if __name__ == "__main__":
    main()
