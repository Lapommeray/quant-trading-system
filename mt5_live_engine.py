"""
MetaTrader 5 Live Trading Engine

Institutional-grade live trading integration with MetaTrader 5.
Handles real-time data streaming, order execution, and portfolio synchronization.
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mt5_live_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MT5LiveEngine")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available. Running in simulation mode.")


class OrderType(Enum):
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"


class ExecutionMode(Enum):
    MARKET = "MARKET"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"


@dataclass
class Position:
    symbol: str
    volume: float
    entry_price: float
    current_price: float
    profit: float
    swap: float
    ticket: int
    direction: str
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Order:
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""
    magic_number: int = 123456
    deviation: int = 20


@dataclass
class RiskLimits:
    max_position_size: float = 0.1
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.05
    max_correlation_exposure: float = 0.3
    max_single_trade_risk: float = 0.02
    require_human_confirmation: bool = True
    human_confirmation_trades: int = 100


@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: datetime
    flags: int = 0


class CircuitBreaker:
    """Circuit breaker for risk management"""
    
    def __init__(self, 
                 max_daily_loss_pct: float = 0.03,
                 max_hourly_loss_pct: float = 0.01,
                 max_consecutive_losses: int = 5,
                 cooldown_minutes: int = 30):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_hourly_loss_pct = max_hourly_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        
        self.daily_pnl = 0.0
        self.hourly_pnl = 0.0
        self.consecutive_losses = 0
        self.last_reset_daily = datetime.now()
        self.last_reset_hourly = datetime.now()
        self.tripped = False
        self.trip_time: Optional[datetime] = None
        self.trip_reason: str = ""
        
    def record_trade(self, pnl: float, is_win: bool):
        """Record trade result"""
        now = datetime.now()
        
        if (now - self.last_reset_daily).days >= 1:
            self.daily_pnl = 0.0
            self.last_reset_daily = now
            
        if (now - self.last_reset_hourly).seconds >= 3600:
            self.hourly_pnl = 0.0
            self.last_reset_hourly = now
            
        self.daily_pnl += pnl
        self.hourly_pnl += pnl
        
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
    def check(self, account_balance: float) -> Tuple[bool, str]:
        """Check if circuit breaker should trip"""
        if self.tripped:
            if self.trip_time and (datetime.now() - self.trip_time).seconds >= self.cooldown_minutes * 60:
                self.reset()
            else:
                return False, f"Circuit breaker tripped: {self.trip_reason}"
                
        if account_balance > 0:
            daily_loss_pct = abs(min(0, self.daily_pnl)) / account_balance
            hourly_loss_pct = abs(min(0, self.hourly_pnl)) / account_balance
            
            if daily_loss_pct >= self.max_daily_loss_pct:
                self._trip(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                return False, self.trip_reason
                
            if hourly_loss_pct >= self.max_hourly_loss_pct:
                self._trip(f"Hourly loss limit exceeded: {hourly_loss_pct:.2%}")
                return False, self.trip_reason
                
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"Consecutive losses limit: {self.consecutive_losses}")
            return False, self.trip_reason
            
        return True, "OK"
        
    def _trip(self, reason: str):
        """Trip the circuit breaker"""
        self.tripped = True
        self.trip_time = datetime.now()
        self.trip_reason = reason
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")
        
    def reset(self):
        """Reset the circuit breaker"""
        self.tripped = False
        self.trip_time = None
        self.trip_reason = ""
        self.consecutive_losses = 0
        logger.info("Circuit breaker reset")


class MT5LiveEngine:
    """
    MetaTrader 5 Live Trading Engine
    
    Features:
    - Real-time tick data streaming
    - Smart order execution (TWAP/VWAP/Iceberg)
    - Risk management with circuit breakers
    - Portfolio state synchronization
    - Human confirmation for initial trades
    """
    
    SYMBOL_MAPPING = {
        "BTC/USDT": "BTCUSD",
        "ETH/USDT": "ETHUSD",
        "EUR/USD": "EURUSD",
        "GBP/USD": "GBPUSD",
        "USD/JPY": "USDJPY",
        "XAU/USD": "XAUUSD",
        "US30": "US30",
        "SPX500": "US500",
        "NASDAQ": "USTEC",
    }
    
    TIMEFRAME_MAPPING = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    
    def __init__(self,
                 login: Optional[int] = None,
                 password: Optional[str] = None,
                 server: Optional[str] = None,
                 risk_limits: Optional[RiskLimits] = None,
                 simulation_mode: bool = False):
        """
        Initialize MT5 Live Engine
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            risk_limits: Risk management limits
            simulation_mode: Run in simulation mode without real trades
        """
        self.login = login or int(os.environ.get("MT5_LOGIN", 0))
        self.password = password or os.environ.get("MT5_PASSWORD", "")
        self.server = server or os.environ.get("MT5_SERVER", "")
        self.risk_limits = risk_limits or RiskLimits()
        self.simulation_mode = simulation_mode or not MT5_AVAILABLE
        
        self.connected = False
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[int, Order] = {}
        self.trade_history: List[Dict] = []
        self.tick_callbacks: Dict[str, List[Callable]] = {}
        self.tick_buffer: Dict[str, deque] = {}
        
        self.circuit_breaker = CircuitBreaker()
        self.trades_executed = 0
        self.human_confirmed = False
        
        self._running = False
        self._tick_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        
        self.account_info: Dict[str, Any] = {}
        
        logger.info(f"MT5LiveEngine initialized (simulation_mode={self.simulation_mode})")
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if self.simulation_mode:
            logger.info("Running in simulation mode - no MT5 connection")
            self.connected = True
            self._initialize_simulation()
            return True
            
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed")
            return False
            
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        if self.login and self.password and self.server:
            authorized = mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            )
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
                
        self.connected = True
        self._sync_account_info()
        logger.info(f"Connected to MT5: {self.account_info.get('name', 'Unknown')}")
        return True
        
    def disconnect(self):
        """Disconnect from MT5 terminal"""
        self._running = False
        
        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=5)
            
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
            
        if not self.simulation_mode and MT5_AVAILABLE:
            mt5.shutdown()
            
        self.connected = False
        logger.info("Disconnected from MT5")
        
    def _initialize_simulation(self):
        """Initialize simulation mode with mock data"""
        self.account_info = {
            "login": 12345678,
            "name": "Simulation Account",
            "balance": 100000.0,
            "equity": 100000.0,
            "margin": 0.0,
            "free_margin": 100000.0,
            "leverage": 100,
            "currency": "USD"
        }
        
    def _sync_account_info(self):
        """Synchronize account information"""
        if self.simulation_mode:
            return
            
        info = mt5.account_info()
        if info:
            self.account_info = {
                "login": info.login,
                "name": info.name,
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "leverage": info.leverage,
                "currency": info.currency
            }
            
    def _map_symbol(self, symbol: str) -> str:
        """Map internal symbol to MT5 symbol"""
        return self.SYMBOL_MAPPING.get(symbol, symbol)
        
    def _reverse_map_symbol(self, mt5_symbol: str) -> str:
        """Map MT5 symbol back to internal symbol"""
        for internal, mt5_sym in self.SYMBOL_MAPPING.items():
            if mt5_sym == mt5_symbol:
                return internal
        return mt5_symbol
        
    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get current tick for symbol"""
        mt5_symbol = self._map_symbol(symbol)
        
        if self.simulation_mode:
            base_price = 100.0
            if "BTC" in symbol:
                base_price = 45000.0
            elif "ETH" in symbol:
                base_price = 2500.0
            elif "XAU" in symbol:
                base_price = 2000.0
            elif "EUR" in symbol:
                base_price = 1.08
            elif "GBP" in symbol:
                base_price = 1.26
                
            spread = base_price * 0.0001
            return TickData(
                symbol=symbol,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                last=base_price,
                volume=1000.0,
                time=datetime.now()
            )
            
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick:
            return TickData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                last=tick.last,
                volume=tick.volume,
                time=datetime.fromtimestamp(tick.time),
                flags=tick.flags
            )
        return None
        
    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get OHLCV data for symbol"""
        mt5_symbol = self._map_symbol(symbol)
        
        if self.simulation_mode:
            dates = pd.date_range(end=datetime.now(), periods=count, freq=timeframe)
            base_price = 100.0
            if "BTC" in symbol:
                base_price = 45000.0
            elif "ETH" in symbol:
                base_price = 2500.0
                
            np.random.seed(42)
            returns = np.random.normal(0, 0.01, count)
            prices = base_price * np.exp(np.cumsum(returns))
            
            df = pd.DataFrame({
                'time': dates,
                'open': prices * (1 + np.random.uniform(-0.005, 0.005, count)),
                'high': prices * (1 + np.random.uniform(0, 0.01, count)),
                'low': prices * (1 - np.random.uniform(0, 0.01, count)),
                'close': prices,
                'volume': np.random.uniform(100, 1000, count)
            })
            df.set_index('time', inplace=True)
            return df
            
        tf_minutes = self.TIMEFRAME_MAPPING.get(timeframe, 1)
        mt5_timeframe = getattr(mt5, f"TIMEFRAME_M{tf_minutes}", mt5.TIMEFRAME_M1)
        
        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        df['volume'] = df['tick_volume']
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    def get_positions(self) -> Dict[str, Position]:
        """Get all open positions"""
        if self.simulation_mode:
            return self.positions
            
        positions = mt5.positions_get()
        if positions is None:
            return {}
            
        result = {}
        for pos in positions:
            symbol = self._reverse_map_symbol(pos.symbol)
            result[symbol] = Position(
                symbol=symbol,
                volume=pos.volume,
                entry_price=pos.price_open,
                current_price=pos.price_current,
                profit=pos.profit,
                swap=pos.swap,
                ticket=pos.ticket,
                direction="BUY" if pos.type == 0 else "SELL",
                open_time=datetime.fromtimestamp(pos.time),
                stop_loss=pos.sl if pos.sl > 0 else None,
                take_profit=pos.tp if pos.tp > 0 else None
            )
        self.positions = result
        return result
        
    def _check_risk_limits(self, order: Order) -> Tuple[bool, str]:
        """Check if order passes risk limits"""
        can_trade, reason = self.circuit_breaker.check(
            self.account_info.get("balance", 0)
        )
        if not can_trade:
            return False, reason
            
        if self.risk_limits.require_human_confirmation:
            if self.trades_executed < self.risk_limits.human_confirmation_trades:
                if not self.human_confirmed:
                    return False, f"Human confirmation required for first {self.risk_limits.human_confirmation_trades} trades"
                    
        tick = self.get_tick(order.symbol)
        if tick:
            price = tick.ask if "BUY" in order.order_type.value else tick.bid
            position_value = order.volume * price
            balance = self.account_info.get("balance", 0)
            
            if balance > 0 and position_value / balance > self.risk_limits.max_position_size:
                return False, f"Position size exceeds limit: {position_value/balance:.2%} > {self.risk_limits.max_position_size:.2%}"
                
            if order.stop_loss:
                risk_per_unit = abs(price - order.stop_loss)
                total_risk = risk_per_unit * order.volume
                if balance > 0 and total_risk / balance > self.risk_limits.max_single_trade_risk:
                    return False, f"Single trade risk exceeds limit: {total_risk/balance:.2%}"
                    
        return True, "OK"
        
    def execute_order(self, order: Order, 
                     execution_mode: ExecutionMode = ExecutionMode.MARKET) -> Dict[str, Any]:
        """
        Execute an order with specified execution mode
        
        Args:
            order: Order to execute
            execution_mode: Execution strategy (MARKET, ICEBERG, TWAP, VWAP)
            
        Returns:
            Execution result dictionary
        """
        can_trade, reason = self._check_risk_limits(order)
        if not can_trade:
            logger.warning(f"Order rejected: {reason}")
            return {"success": False, "error": reason}
            
        if execution_mode == ExecutionMode.MARKET:
            return self._execute_market_order(order)
        elif execution_mode == ExecutionMode.ICEBERG:
            return self._execute_iceberg_order(order)
        elif execution_mode == ExecutionMode.TWAP:
            return self._execute_twap_order(order)
        elif execution_mode == ExecutionMode.VWAP:
            return self._execute_vwap_order(order)
        else:
            return {"success": False, "error": f"Unknown execution mode: {execution_mode}"}
            
    def _execute_market_order(self, order: Order) -> Dict[str, Any]:
        """Execute a market order"""
        mt5_symbol = self._map_symbol(order.symbol)
        
        if self.simulation_mode:
            tick = self.get_tick(order.symbol)
            if not tick:
                return {"success": False, "error": "No tick data"}
                
            fill_price = tick.ask if "BUY" in order.order_type.value else tick.bid
            
            result = {
                "success": True,
                "ticket": int(time.time() * 1000),
                "symbol": order.symbol,
                "volume": order.volume,
                "price": fill_price,
                "order_type": order.order_type.value,
                "time": datetime.now().isoformat(),
                "comment": order.comment
            }
            
            if "BUY" in order.order_type.value:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    volume=order.volume,
                    entry_price=fill_price,
                    current_price=fill_price,
                    profit=0.0,
                    swap=0.0,
                    ticket=result["ticket"],
                    direction="BUY",
                    open_time=datetime.now(),
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit
                )
                
            self.trades_executed += 1
            self.trade_history.append(result)
            logger.info(f"Simulated order executed: {result}")
            return result
            
        tick = mt5.symbol_info_tick(mt5_symbol)
        if not tick:
            return {"success": False, "error": f"No tick data for {mt5_symbol}"}
            
        if "BUY" in order.order_type.value:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": order.volume,
            "type": order_type,
            "price": price,
            "deviation": order.deviation,
            "magic": order.magic_number,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if order.stop_loss:
            request["sl"] = order.stop_loss
        if order.take_profit:
            request["tp"] = order.take_profit
            
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.retcode} - {result.comment}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "retcode": result.retcode}
            
        self.trades_executed += 1
        
        execution_result = {
            "success": True,
            "ticket": result.order,
            "symbol": order.symbol,
            "volume": result.volume,
            "price": result.price,
            "order_type": order.order_type.value,
            "time": datetime.now().isoformat(),
            "comment": order.comment
        }
        
        self.trade_history.append(execution_result)
        logger.info(f"Order executed: {execution_result}")
        return execution_result
        
    def _execute_iceberg_order(self, order: Order, 
                               slice_size: float = 0.1,
                               delay_seconds: float = 5.0) -> Dict[str, Any]:
        """Execute iceberg order in slices"""
        total_volume = order.volume
        slice_volume = total_volume * slice_size
        num_slices = int(1 / slice_size)
        
        results = []
        total_filled = 0.0
        weighted_price = 0.0
        
        for i in range(num_slices):
            remaining = total_volume - total_filled
            current_slice = min(slice_volume, remaining)
            
            if current_slice <= 0:
                break
                
            slice_order = Order(
                symbol=order.symbol,
                order_type=order.order_type,
                volume=current_slice,
                stop_loss=order.stop_loss if i == num_slices - 1 else None,
                take_profit=order.take_profit if i == num_slices - 1 else None,
                comment=f"{order.comment}_slice_{i+1}",
                magic_number=order.magic_number,
                deviation=order.deviation
            )
            
            result = self._execute_market_order(slice_order)
            results.append(result)
            
            if result["success"]:
                total_filled += result["volume"]
                weighted_price += result["price"] * result["volume"]
                
            if i < num_slices - 1:
                time.sleep(delay_seconds)
                
        avg_price = weighted_price / total_filled if total_filled > 0 else 0
        
        return {
            "success": total_filled > 0,
            "execution_mode": "ICEBERG",
            "total_volume": total_filled,
            "avg_price": avg_price,
            "num_slices": len(results),
            "slices": results
        }
        
    def _execute_twap_order(self, order: Order,
                           duration_minutes: int = 30,
                           num_slices: int = 10) -> Dict[str, Any]:
        """Execute TWAP order over time period"""
        slice_volume = order.volume / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        results = []
        total_filled = 0.0
        weighted_price = 0.0
        
        for i in range(num_slices):
            slice_order = Order(
                symbol=order.symbol,
                order_type=order.order_type,
                volume=slice_volume,
                stop_loss=order.stop_loss if i == num_slices - 1 else None,
                take_profit=order.take_profit if i == num_slices - 1 else None,
                comment=f"{order.comment}_twap_{i+1}",
                magic_number=order.magic_number,
                deviation=order.deviation
            )
            
            result = self._execute_market_order(slice_order)
            results.append(result)
            
            if result["success"]:
                total_filled += result["volume"]
                weighted_price += result["price"] * result["volume"]
                
            if i < num_slices - 1:
                time.sleep(interval_seconds)
                
        avg_price = weighted_price / total_filled if total_filled > 0 else 0
        
        return {
            "success": total_filled > 0,
            "execution_mode": "TWAP",
            "total_volume": total_filled,
            "avg_price": avg_price,
            "duration_minutes": duration_minutes,
            "num_slices": len(results),
            "slices": results
        }
        
    def _execute_vwap_order(self, order: Order,
                           duration_minutes: int = 30) -> Dict[str, Any]:
        """Execute VWAP order based on volume profile"""
        historical_data = self.get_ohlcv(order.symbol, "1m", duration_minutes)
        
        if historical_data.empty:
            logger.warning("No historical data for VWAP, falling back to TWAP")
            return self._execute_twap_order(order, duration_minutes)
            
        volume_profile = historical_data['volume'].values
        total_volume = volume_profile.sum()
        
        if total_volume == 0:
            return self._execute_twap_order(order, duration_minutes)
            
        volume_weights = volume_profile / total_volume
        
        results = []
        total_filled = 0.0
        weighted_price = 0.0
        
        for i, weight in enumerate(volume_weights):
            slice_volume = order.volume * weight
            
            if slice_volume < 0.01:
                continue
                
            slice_order = Order(
                symbol=order.symbol,
                order_type=order.order_type,
                volume=slice_volume,
                comment=f"{order.comment}_vwap_{i+1}",
                magic_number=order.magic_number,
                deviation=order.deviation
            )
            
            result = self._execute_market_order(slice_order)
            results.append(result)
            
            if result["success"]:
                total_filled += result["volume"]
                weighted_price += result["price"] * result["volume"]
                
            time.sleep(60)
            
        avg_price = weighted_price / total_filled if total_filled > 0 else 0
        
        return {
            "success": total_filled > 0,
            "execution_mode": "VWAP",
            "total_volume": total_filled,
            "avg_price": avg_price,
            "num_slices": len(results),
            "slices": results
        }
        
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close position for symbol"""
        positions = self.get_positions()
        
        if symbol not in positions:
            return {"success": False, "error": f"No position for {symbol}"}
            
        position = positions[symbol]
        
        close_order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET_SELL if position.direction == "BUY" else OrderType.MARKET_BUY,
            volume=position.volume,
            comment=f"Close_{symbol}"
        )
        
        result = self._execute_market_order(close_order)
        
        if result["success"] and self.simulation_mode:
            del self.positions[symbol]
            
        return result
        
    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions"""
        results = []
        positions = self.get_positions()
        
        for symbol in list(positions.keys()):
            result = self.close_position(symbol)
            results.append(result)
            
        return results
        
    def subscribe_ticks(self, symbol: str, callback: Callable[[TickData], None]):
        """Subscribe to tick updates for symbol"""
        if symbol not in self.tick_callbacks:
            self.tick_callbacks[symbol] = []
            self.tick_buffer[symbol] = deque(maxlen=1000)
            
        self.tick_callbacks[symbol].append(callback)
        logger.info(f"Subscribed to ticks for {symbol}")
        
    def start_tick_streaming(self):
        """Start tick data streaming thread"""
        if self._running:
            return
            
        self._running = True
        self._tick_thread = threading.Thread(target=self._tick_streaming_loop, daemon=True)
        self._tick_thread.start()
        logger.info("Tick streaming started")
        
    def _tick_streaming_loop(self):
        """Main tick streaming loop"""
        while self._running:
            for symbol in list(self.tick_callbacks.keys()):
                tick = self.get_tick(symbol)
                if tick:
                    self.tick_buffer[symbol].append(tick)
                    for callback in self.tick_callbacks[symbol]:
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"Tick callback error: {e}")
                            
            time.sleep(0.1)
            
    def start_portfolio_sync(self, interval_seconds: int = 5):
        """Start portfolio synchronization thread"""
        self._sync_thread = threading.Thread(
            target=self._portfolio_sync_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._sync_thread.start()
        logger.info("Portfolio sync started")
        
    def _portfolio_sync_loop(self, interval: int):
        """Portfolio synchronization loop"""
        while self._running:
            try:
                self._sync_account_info()
                self.get_positions()
            except Exception as e:
                logger.error(f"Portfolio sync error: {e}")
            time.sleep(interval)
            
    def confirm_human_trading(self, confirmation_code: str = "CONFIRM_LIVE_TRADING"):
        """Confirm human authorization for live trading"""
        if confirmation_code == "CONFIRM_LIVE_TRADING":
            self.human_confirmed = True
            logger.info("Human trading confirmation received")
            return True
        return False
        
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        self._sync_account_info()
        positions = self.get_positions()
        
        total_profit = sum(p.profit for p in positions.values())
        total_volume = sum(p.volume for p in positions.values())
        
        return {
            "account": self.account_info,
            "positions_count": len(positions),
            "total_profit": total_profit,
            "total_volume": total_volume,
            "trades_executed": self.trades_executed,
            "circuit_breaker_status": "OK" if not self.circuit_breaker.tripped else self.circuit_breaker.trip_reason,
            "human_confirmed": self.human_confirmed,
            "simulation_mode": self.simulation_mode
        }
        
    def emergency_stop(self):
        """Emergency stop - close all positions and halt trading"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        
        self.circuit_breaker._trip("Emergency stop activated")
        
        results = self.close_all_positions()
        
        self._running = False
        
        return {
            "emergency_stop": True,
            "positions_closed": results,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Demo of MT5 Live Engine"""
    engine = MT5LiveEngine(simulation_mode=True)
    
    if not engine.connect():
        print("Failed to connect")
        return
        
    print("\n=== Account Summary ===")
    summary = engine.get_account_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    print("\n=== Getting Tick Data ===")
    tick = engine.get_tick("BTC/USDT")
    if tick:
        print(f"BTC/USDT: Bid={tick.bid}, Ask={tick.ask}")
        
    print("\n=== Getting OHLCV Data ===")
    ohlcv = engine.get_ohlcv("BTC/USDT", "1h", 10)
    print(ohlcv.tail())
    
    engine.confirm_human_trading("CONFIRM_LIVE_TRADING")
    
    print("\n=== Executing Test Order ===")
    order = Order(
        symbol="BTC/USDT",
        order_type=OrderType.MARKET_BUY,
        volume=0.01,
        stop_loss=44000.0,
        take_profit=46000.0,
        comment="Test order"
    )
    
    result = engine.execute_order(order)
    print(json.dumps(result, indent=2, default=str))
    
    print("\n=== Positions ===")
    positions = engine.get_positions()
    for symbol, pos in positions.items():
        print(f"{symbol}: {pos.direction} {pos.volume} @ {pos.entry_price}")
        
    engine.disconnect()
    print("\nDemo complete")


if __name__ == "__main__":
    main()
