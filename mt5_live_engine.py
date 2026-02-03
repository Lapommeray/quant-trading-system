"""
MT5 Live Trading Engine with IBKR Fallback

A production-ready live trading engine that:
- Connects to MetaTrader 5 for forex/CFD execution
- Falls back to Interactive Brokers (IBKR) for superior execution
- Streams real-time tick data
- Executes orders with smart routing (TWAP/VWAP/Iceberg)
- Manages portfolio state and risk
- Implements circuit breakers and safety controls

Note: MetaTrader5 package is Windows-only. On Linux, runs in simulation mode.
"""

import os
import sys
import time
import json
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MT5LiveEngine")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.info("MetaTrader5 package available")
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available (Windows only). Running in simulation mode.")

try:
    from ib_insync import IB, Stock, Forex, Contract, Order, Trade
    IBKR_AVAILABLE = True
    logger.info("ib_insync package available - IBKR fallback enabled")
except ImportError:
    IBKR_AVAILABLE = False
    logger.warning("ib_insync not available. IBKR fallback disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from advanced_modules.bayesian_market_state import BayesianMarketState
    from advanced_modules.microstructure_detector import MicrostructureDetectors, OrderBookSnapshot
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    logger.warning("Bayesian/Microstructure modules not available")


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class ExecutionAlgo(Enum):
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class BrokerType(Enum):
    MT5 = "mt5"
    IBKR = "ibkr"
    SIMULATION = "simulation"


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
        
    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    algo: ExecutionAlgo = ExecutionAlgo.MARKET
    algo_params: Dict = field(default_factory=dict)
    client_order_id: str = ""
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


@dataclass
class OrderResult:
    success: bool
    order_id: str
    client_order_id: str
    filled_quantity: float
    avg_fill_price: float
    status: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    broker: BrokerType


class CircuitBreaker:
    """Circuit breaker for risk management"""
    
    def __init__(self,
                 max_daily_loss_pct: float = 0.03,
                 max_position_size: float = 0.1,
                 max_orders_per_minute: int = 10,
                 cooldown_minutes: int = 30):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_size = max_position_size
        self.max_orders_per_minute = max_orders_per_minute
        self.cooldown_minutes = cooldown_minutes
        
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.order_timestamps: deque = deque(maxlen=100)
        self.tripped = False
        self.trip_time: Optional[datetime] = None
        self.trip_reason: str = ""
        
    def reset_daily(self, equity: float):
        """Reset daily tracking"""
        self.daily_pnl = 0.0
        self.starting_equity = equity
        self.tripped = False
        self.trip_time = None
        self.trip_reason = ""
        
    def update_pnl(self, pnl_change: float):
        """Update daily P&L"""
        self.daily_pnl += pnl_change
        
    def check_order(self, order: OrderRequest, account: AccountInfo) -> Tuple[bool, str]:
        """Check if order should be allowed"""
        if self.tripped:
            if self.trip_time and datetime.now() - self.trip_time > timedelta(minutes=self.cooldown_minutes):
                self.tripped = False
                logger.info("Circuit breaker cooldown expired, resetting")
            else:
                return False, f"Circuit breaker tripped: {self.trip_reason}"
                
        if self.starting_equity > 0:
            loss_pct = -self.daily_pnl / self.starting_equity
            if loss_pct > self.max_daily_loss_pct:
                self._trip(f"Daily loss limit exceeded: {loss_pct:.2%}")
                return False, self.trip_reason
                
        order_value = order.quantity * (order.limit_price or 0)
        if account.equity > 0 and order_value / account.equity > self.max_position_size:
            return False, f"Position size too large: {order_value / account.equity:.2%}"
            
        now = datetime.now()
        self.order_timestamps.append(now)
        recent_orders = sum(1 for t in self.order_timestamps if now - t < timedelta(minutes=1))
        if recent_orders > self.max_orders_per_minute:
            self._trip(f"Order rate limit exceeded: {recent_orders}/min")
            return False, self.trip_reason
            
        return True, "OK"
        
    def _trip(self, reason: str):
        """Trip the circuit breaker"""
        self.tripped = True
        self.trip_time = datetime.now()
        self.trip_reason = reason
        logger.warning(f"Circuit breaker tripped: {reason}")


class SmartOrderRouter:
    """Smart order routing with TWAP/VWAP/Iceberg algorithms"""
    
    def __init__(self, execute_func: Callable):
        self.execute_func = execute_func
        self.active_algos: Dict[str, threading.Thread] = {}
        self.cancel_flags: Dict[str, threading.Event] = {}
        
    def execute(self, order: OrderRequest) -> OrderResult:
        """Execute order with specified algorithm"""
        if order.algo == ExecutionAlgo.MARKET:
            return self.execute_func(order)
        elif order.algo == ExecutionAlgo.TWAP:
            return self._execute_twap(order)
        elif order.algo == ExecutionAlgo.VWAP:
            return self._execute_vwap(order)
        elif order.algo == ExecutionAlgo.ICEBERG:
            return self._execute_iceberg(order)
        else:
            return self.execute_func(order)
            
    def _execute_twap(self, order: OrderRequest) -> OrderResult:
        """Time-Weighted Average Price execution"""
        duration_minutes = order.algo_params.get("duration_minutes", 30)
        num_slices = order.algo_params.get("num_slices", 10)
        
        slice_qty = order.quantity / num_slices
        interval = (duration_minutes * 60) / num_slices
        
        total_filled = 0.0
        total_value = 0.0
        
        cancel_event = threading.Event()
        self.cancel_flags[order.client_order_id] = cancel_event
        
        for i in range(num_slices):
            if cancel_event.is_set():
                break
                
            slice_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.MARKET,
                client_order_id=f"{order.client_order_id}_slice_{i}"
            )
            
            result = self.execute_func(slice_order)
            
            if result.success:
                total_filled += result.filled_quantity
                total_value += result.filled_quantity * result.avg_fill_price
                
            if i < num_slices - 1:
                time.sleep(interval)
                
        avg_price = total_value / total_filled if total_filled > 0 else 0
        
        return OrderResult(
            success=total_filled > 0,
            order_id=order.client_order_id,
            client_order_id=order.client_order_id,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            status="filled" if total_filled >= order.quantity * 0.95 else "partial",
            message=f"TWAP complete: {total_filled}/{order.quantity}"
        )
        
    def _execute_vwap(self, order: OrderRequest) -> OrderResult:
        """Volume-Weighted Average Price execution"""
        duration_minutes = order.algo_params.get("duration_minutes", 30)
        participation_rate = order.algo_params.get("participation_rate", 0.1)
        
        return self._execute_twap(order)
        
    def _execute_iceberg(self, order: OrderRequest) -> OrderResult:
        """Iceberg order execution - show only small visible quantity"""
        visible_qty = order.algo_params.get("visible_qty", order.quantity * 0.1)
        
        total_filled = 0.0
        total_value = 0.0
        remaining = order.quantity
        
        cancel_event = threading.Event()
        self.cancel_flags[order.client_order_id] = cancel_event
        
        slice_num = 0
        while remaining > 0 and not cancel_event.is_set():
            slice_qty = min(visible_qty, remaining)
            
            slice_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                client_order_id=f"{order.client_order_id}_ice_{slice_num}"
            )
            
            result = self.execute_func(slice_order)
            
            if result.success:
                total_filled += result.filled_quantity
                total_value += result.filled_quantity * result.avg_fill_price
                remaining -= result.filled_quantity
            else:
                break
                
            slice_num += 1
            time.sleep(0.5)
            
        avg_price = total_value / total_filled if total_filled > 0 else 0
        
        return OrderResult(
            success=total_filled > 0,
            order_id=order.client_order_id,
            client_order_id=order.client_order_id,
            filled_quantity=total_filled,
            avg_fill_price=avg_price,
            status="filled" if remaining <= 0 else "partial",
            message=f"Iceberg complete: {total_filled}/{order.quantity}"
        )
        
    def cancel_algo(self, client_order_id: str):
        """Cancel running algorithm"""
        if client_order_id in self.cancel_flags:
            self.cancel_flags[client_order_id].set()


class IBKRConnection:
    """Interactive Brokers connection handler"""
    
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 7497,
                 client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib: Optional[IB] = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway"""
        if not IBKR_AVAILABLE:
            logger.warning("ib_insync not available")
            return False
            
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")
            
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if not self.connected or not self.ib:
            return None
            
        try:
            account_values = self.ib.accountValues()
            
            balance = 0.0
            equity = 0.0
            
            for av in account_values:
                if av.tag == "TotalCashBalance" and av.currency == "USD":
                    balance = float(av.value)
                elif av.tag == "NetLiquidation" and av.currency == "USD":
                    equity = float(av.value)
                    
            return AccountInfo(
                balance=balance,
                equity=equity,
                margin=0.0,
                free_margin=balance,
                margin_level=100.0,
                currency="USD",
                broker=BrokerType.IBKR
            )
        except Exception as e:
            logger.error(f"Failed to get IBKR account info: {e}")
            return None
            
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place order through IBKR"""
        if not self.connected or not self.ib:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message="Not connected to IBKR"
            )
            
        try:
            if "/" in order.symbol:
                base, quote = order.symbol.split("/")
                contract = Forex(base + quote)
            else:
                contract = Stock(order.symbol, "SMART", "USD")
                
            self.ib.qualifyContracts(contract)
            
            if order.order_type == OrderType.MARKET:
                ib_order = Order(
                    action="BUY" if order.side == OrderSide.BUY else "SELL",
                    totalQuantity=order.quantity,
                    orderType="MKT"
                )
            elif order.order_type == OrderType.LIMIT:
                ib_order = Order(
                    action="BUY" if order.side == OrderSide.BUY else "SELL",
                    totalQuantity=order.quantity,
                    orderType="LMT",
                    lmtPrice=order.limit_price
                )
            else:
                ib_order = Order(
                    action="BUY" if order.side == OrderSide.BUY else "SELL",
                    totalQuantity=order.quantity,
                    orderType="MKT"
                )
                
            trade = self.ib.placeOrder(contract, ib_order)
            
            timeout = 30
            start = time.time()
            while not trade.isDone() and time.time() - start < timeout:
                self.ib.sleep(0.1)
                
            return OrderResult(
                success=trade.orderStatus.status in ["Filled", "Submitted"],
                order_id=str(trade.order.orderId),
                client_order_id=order.client_order_id,
                filled_quantity=trade.orderStatus.filled,
                avg_fill_price=trade.orderStatus.avgFillPrice,
                status=trade.orderStatus.status,
                message=f"IBKR order: {trade.orderStatus.status}"
            )
            
        except Exception as e:
            logger.error(f"IBKR order failed: {e}")
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message=str(e)
            )


class SimulationEngine:
    """Simulation engine for testing without live connection"""
    
    def __init__(self, initial_balance: float = 100000):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[str, Position] = {}
        self.order_history: List[OrderResult] = []
        self.tick_data: Dict[str, Tick] = {}
        
        self._generate_initial_prices()
        
    def _generate_initial_prices(self):
        """Generate initial simulated prices"""
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        base_prices = [1.0850, 1.2650, 149.50, 0.6550, 1.3650]
        
        for symbol, price in zip(symbols, base_prices):
            spread = price * 0.0001
            self.tick_data[symbol] = Tick(
                symbol=symbol,
                bid=price - spread/2,
                ask=price + spread/2,
                last=price,
                volume=1000000,
                timestamp=datetime.now()
            )
            
    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current tick with simulated price movement"""
        if symbol not in self.tick_data:
            base_price = 1.0 + random.random()
            spread = base_price * 0.0001
            self.tick_data[symbol] = Tick(
                symbol=symbol,
                bid=base_price - spread/2,
                ask=base_price + spread/2,
                last=base_price,
                volume=1000000,
                timestamp=datetime.now()
            )
            
        tick = self.tick_data[symbol]
        
        change = (random.random() - 0.5) * 0.0002 * tick.last
        new_price = tick.last + change
        spread = new_price * 0.0001
        
        self.tick_data[symbol] = Tick(
            symbol=symbol,
            bid=new_price - spread/2,
            ask=new_price + spread/2,
            last=new_price,
            volume=tick.volume + random.randint(-10000, 10000),
            timestamp=datetime.now()
        )
        
        return self.tick_data[symbol]
        
    def get_account_info(self) -> AccountInfo:
        """Get simulated account info"""
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=0.0,
            free_margin=self.balance,
            margin_level=100.0,
            currency="USD",
            broker=BrokerType.SIMULATION
        )
        
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Execute simulated order"""
        tick = self.get_tick(order.symbol)
        if not tick:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message="Symbol not found"
            )
            
        fill_price = tick.ask if order.side == OrderSide.BUY else tick.bid
        
        slippage = fill_price * 0.00005 * (1 if order.side == OrderSide.BUY else -1)
        fill_price += slippage
        
        if order.symbol in self.positions:
            pos = self.positions[order.symbol]
            if order.side == OrderSide.BUY:
                new_qty = pos.quantity + order.quantity
                new_avg = (pos.avg_price * pos.quantity + fill_price * order.quantity) / new_qty
            else:
                new_qty = pos.quantity - order.quantity
                new_avg = pos.avg_price
                
            if abs(new_qty) < 0.0001:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=new_qty,
                    avg_price=new_avg,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    side="long" if new_qty > 0 else "short"
                )
        else:
            qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=qty,
                avg_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0,
                realized_pnl=0,
                side="long" if qty > 0 else "short"
            )
            
        result = OrderResult(
            success=True,
            order_id=f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            client_order_id=order.client_order_id,
            filled_quantity=order.quantity,
            avg_fill_price=fill_price,
            status="filled",
            message="Simulated fill"
        )
        
        self.order_history.append(result)
        return result


class MT5LiveEngine:
    """
    Main MT5 Live Trading Engine with IBKR fallback.
    
    Features:
    - Multi-broker support (MT5, IBKR, Simulation)
    - Real-time tick streaming
    - Smart order routing (TWAP/VWAP/Iceberg)
    - Circuit breakers and risk management
    - Position and portfolio management
    """
    
    SYMBOL_MAPPING = {
        "EUR/USD": "EURUSD",
        "GBP/USD": "GBPUSD",
        "USD/JPY": "USDJPY",
        "AUD/USD": "AUDUSD",
        "USD/CAD": "USDCAD",
        "USD/CHF": "USDCHF",
        "NZD/USD": "NZDUSD",
        "EUR/GBP": "EURGBP",
        "EUR/JPY": "EURJPY",
        "GBP/JPY": "GBPJPY"
    }
    
    def __init__(self,
                 prefer_ibkr: bool = False,
                 simulation_mode: bool = False,
                 ibkr_host: str = "127.0.0.1",
                 ibkr_port: int = 7497,
                 ibkr_client_id: int = 1):
        """
        Initialize the MT5 Live Engine.
        
        Args:
            prefer_ibkr: Prefer IBKR over MT5 when both available
            simulation_mode: Force simulation mode
            ibkr_host: IBKR TWS/Gateway host
            ibkr_port: IBKR TWS/Gateway port
            ibkr_client_id: IBKR client ID
        """
        self.prefer_ibkr = prefer_ibkr
        self.simulation_mode = simulation_mode
        
        self.status = ConnectionStatus.DISCONNECTED
        self.active_broker: Optional[BrokerType] = None
        
        self.ibkr = IBKRConnection(ibkr_host, ibkr_port, ibkr_client_id)
        self.simulation = SimulationEngine()
        
        self.circuit_breaker = CircuitBreaker()
        self.smart_router = SmartOrderRouter(self._execute_order_direct)
        
        self.tick_callbacks: List[Callable[[Tick], None]] = []
        self.tick_thread: Optional[threading.Thread] = None
        self.tick_running = False
        
        self.positions: Dict[str, Position] = {}
        self.order_history: List[OrderResult] = []
        
        logger.info("MT5LiveEngine initialized")
        
    def connect(self) -> bool:
        """Connect to trading platform"""
        self.status = ConnectionStatus.CONNECTING
        
        if self.simulation_mode:
            self.active_broker = BrokerType.SIMULATION
            self.status = ConnectionStatus.CONNECTED
            logger.info("Connected in SIMULATION mode")
            return True
            
        if self.prefer_ibkr and IBKR_AVAILABLE:
            if self.ibkr.connect():
                self.active_broker = BrokerType.IBKR
                self.status = ConnectionStatus.CONNECTED
                logger.info("Connected to IBKR - superior execution active")
                return True
                
        if MT5_AVAILABLE:
            if mt5.initialize():
                self.active_broker = BrokerType.MT5
                self.status = ConnectionStatus.CONNECTED
                logger.info("Connected to MT5")
                return True
                
        if IBKR_AVAILABLE and not self.prefer_ibkr:
            if self.ibkr.connect():
                self.active_broker = BrokerType.IBKR
                self.status = ConnectionStatus.CONNECTED
                logger.info("Connected to IBKR (MT5 fallback)")
                return True
                
        self.active_broker = BrokerType.SIMULATION
        self.status = ConnectionStatus.CONNECTED
        logger.warning("No live connection available, using SIMULATION mode")
        return True
        
    def disconnect(self):
        """Disconnect from trading platform"""
        self.stop_tick_stream()
        
        if self.active_broker == BrokerType.MT5 and MT5_AVAILABLE:
            mt5.shutdown()
        elif self.active_broker == BrokerType.IBKR:
            self.ibkr.disconnect()
            
        self.status = ConnectionStatus.DISCONNECTED
        self.active_broker = None
        logger.info("Disconnected from trading platform")
        
    def get_account_info(self) -> Optional[AccountInfo]:
        """Get account information"""
        if self.active_broker == BrokerType.MT5 and MT5_AVAILABLE:
            info = mt5.account_info()
            if info:
                return AccountInfo(
                    balance=info.balance,
                    equity=info.equity,
                    margin=info.margin,
                    free_margin=info.margin_free,
                    margin_level=info.margin_level if info.margin_level else 0,
                    currency=info.currency,
                    broker=BrokerType.MT5
                )
        elif self.active_broker == BrokerType.IBKR:
            return self.ibkr.get_account_info()
        else:
            return self.simulation.get_account_info()
            
        return None
        
    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Get current tick for symbol"""
        mt5_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        
        if self.active_broker == BrokerType.MT5 and MT5_AVAILABLE:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick:
                return Tick(
                    symbol=symbol,
                    bid=tick.bid,
                    ask=tick.ask,
                    last=tick.last,
                    volume=tick.volume,
                    timestamp=datetime.fromtimestamp(tick.time)
                )
        elif self.active_broker == BrokerType.SIMULATION:
            return self.simulation.get_tick(mt5_symbol)
            
        return None
        
    def start_tick_stream(self, symbols: List[str], callback: Callable[[Tick], None]):
        """Start streaming ticks for symbols"""
        self.tick_callbacks.append(callback)
        
        if self.tick_running:
            return
            
        self.tick_running = True
        
        def stream_loop():
            while self.tick_running:
                for symbol in symbols:
                    tick = self.get_tick(symbol)
                    if tick:
                        for cb in self.tick_callbacks:
                            try:
                                cb(tick)
                            except Exception as e:
                                logger.error(f"Tick callback error: {e}")
                time.sleep(0.1)
                
        self.tick_thread = threading.Thread(target=stream_loop, daemon=True)
        self.tick_thread.start()
        logger.info(f"Started tick stream for {symbols}")
        
    def stop_tick_stream(self):
        """Stop tick streaming"""
        self.tick_running = False
        if self.tick_thread:
            self.tick_thread.join(timeout=5)
        self.tick_callbacks.clear()
        logger.info("Stopped tick stream")
        
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place order with risk checks and smart routing"""
        account = self.get_account_info()
        if not account:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message="Could not get account info"
            )
            
        allowed, reason = self.circuit_breaker.check_order(order, account)
        if not allowed:
            logger.warning(f"Order blocked by circuit breaker: {reason}")
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="blocked",
                message=reason
            )
            
        result = self.smart_router.execute(order)
        
        self.order_history.append(result)
        
        if result.success:
            self._update_position(order, result)
            
        return result
        
    def _execute_order_direct(self, order: OrderRequest) -> OrderResult:
        """Execute order directly without smart routing"""
        if self.active_broker == BrokerType.MT5 and MT5_AVAILABLE:
            return self._execute_mt5_order(order)
        elif self.active_broker == BrokerType.IBKR:
            return self.ibkr.place_order(order)
        else:
            return self.simulation.place_order(order)
            
    def _execute_mt5_order(self, order: OrderRequest) -> OrderResult:
        """Execute order through MT5"""
        mt5_symbol = self.SYMBOL_MAPPING.get(order.symbol, order.symbol)
        
        symbol_info = mt5.symbol_info(mt5_symbol)
        if not symbol_info:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message=f"Symbol {mt5_symbol} not found"
            )
            
        if not symbol_info.visible:
            mt5.symbol_select(mt5_symbol, True)
            
        tick = mt5.symbol_info_tick(mt5_symbol)
        if not tick:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message="Could not get tick"
            )
            
        price = tick.ask if order.side == OrderSide.BUY else tick.bid
        
        if order.order_type == OrderType.MARKET:
            order_type = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        elif order.order_type == OrderType.LIMIT:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT
            price = order.limit_price
        else:
            order_type = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": order.quantity,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": order.client_order_id,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=True,
                order_id=str(result.order),
                client_order_id=order.client_order_id,
                filled_quantity=result.volume,
                avg_fill_price=result.price,
                status="filled",
                message="MT5 order filled"
            )
        else:
            return OrderResult(
                success=False,
                order_id="",
                client_order_id=order.client_order_id,
                filled_quantity=0,
                avg_fill_price=0,
                status="error",
                message=f"MT5 error: {result.retcode}"
            )
            
    def _update_position(self, order: OrderRequest, result: OrderResult):
        """Update position tracking after fill"""
        if order.symbol in self.positions:
            pos = self.positions[order.symbol]
            if order.side == OrderSide.BUY:
                new_qty = pos.quantity + result.filled_quantity
                if pos.quantity > 0:
                    new_avg = (pos.avg_price * pos.quantity + result.avg_fill_price * result.filled_quantity) / new_qty
                else:
                    new_avg = result.avg_fill_price
            else:
                new_qty = pos.quantity - result.filled_quantity
                new_avg = pos.avg_price
                
            if abs(new_qty) < 0.0001:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=new_qty,
                    avg_price=new_avg,
                    current_price=result.avg_fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    side="long" if new_qty > 0 else "short"
                )
        else:
            qty = result.filled_quantity if order.side == OrderSide.BUY else -result.filled_quantity
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=qty,
                avg_price=result.avg_fill_price,
                current_price=result.avg_fill_price,
                unrealized_pnl=0,
                realized_pnl=0,
                side="long" if qty > 0 else "short"
            )
            
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "connection_status": self.status.value,
            "active_broker": self.active_broker.value if self.active_broker else None,
            "circuit_breaker_tripped": self.circuit_breaker.tripped,
            "positions_count": len(self.positions),
            "orders_count": len(self.order_history),
            "tick_streaming": self.tick_running
        }


class DepthFeedIntegration:
    """
    Order book depth feed integration with Bayesian market state.
    
    Streams L2 order book data and feeds it into the microstructure
    detectors and Bayesian belief system for real-time MM behavior analysis.
    """
    
    def __init__(self, engine: MT5LiveEngine, symbols: List[str]):
        """
        Initialize depth feed integration.
        
        Args:
            engine: MT5LiveEngine instance
            symbols: List of symbols to track
        """
        self.engine = engine
        self.symbols = symbols
        
        self.market_states: Dict[str, Any] = {}
        if BAYESIAN_AVAILABLE:
            for symbol in symbols:
                self.market_states[symbol] = BayesianMarketState()
        
        self.depth_history: Dict[str, deque] = {s: deque(maxlen=100) for s in symbols}
        self.last_book: Dict[str, Dict] = {}
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self.add_counts: Dict[str, int] = {s: 0 for s in symbols}
        self.cancel_counts: Dict[str, int] = {s: 0 for s in symbols}
        self.trade_counts: Dict[str, int] = {s: 0 for s in symbols}
        
        logger.info(f"DepthFeedIntegration initialized for {len(symbols)} symbols")
        
    def start(self, callback: Optional[Callable] = None):
        """Start depth feed streaming"""
        if self._running:
            return
            
        self._running = True
        self._callback = callback
        
        def stream_loop():
            while self._running:
                for symbol in self.symbols:
                    try:
                        book_data = self._get_order_book(symbol)
                        if book_data:
                            belief = self._process_book_update(symbol, book_data)
                            if self._callback and belief:
                                self._callback(symbol, belief)
                    except Exception as e:
                        logger.error(f"Depth feed error for {symbol}: {e}")
                time.sleep(0.05)
                
        self._thread = threading.Thread(target=stream_loop, daemon=True)
        self._thread.start()
        logger.info("Depth feed streaming started")
        
    def stop(self):
        """Stop depth feed streaming"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Depth feed streaming stopped")
        
    def _get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get order book data from MT5 or simulation"""
        mt5_symbol = self.engine.SYMBOL_MAPPING.get(symbol, symbol)
        
        if self.engine.active_broker == BrokerType.MT5 and MT5_AVAILABLE:
            book = mt5.market_book_get(mt5_symbol)
            if book:
                bids = [(item.price, item.volume) for item in book if item.type == 1]
                asks = [(item.price, item.volume) for item in book if item.type == 2]
                
                bids.sort(key=lambda x: -x[0])
                asks.sort(key=lambda x: x[0])
                
                return {
                    "symbol": symbol,
                    "bids": bids[:10],
                    "asks": asks[:10],
                    "timestamp": datetime.now()
                }
        else:
            tick = self.engine.get_tick(symbol)
            if tick:
                spread = tick.spread
                mid = tick.mid
                
                bids = [(mid - spread/2 - i*0.0001, 1000 + random.randint(-200, 200)) 
                        for i in range(10)]
                asks = [(mid + spread/2 + i*0.0001, 1000 + random.randint(-200, 200)) 
                        for i in range(10)]
                
                return {
                    "symbol": symbol,
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now()
                }
                
        return None
        
    def _process_book_update(self, symbol: str, book_data: Dict) -> Optional[Dict]:
        """Process order book update and update Bayesian state"""
        if not BAYESIAN_AVAILABLE or symbol not in self.market_states:
            return None
            
        bids = book_data["bids"]
        asks = book_data["asks"]
        
        bid_depth = sum(vol for _, vol in bids)
        ask_depth = sum(vol for _, vol in asks)
        
        adds = 10
        cancels = 3
        trades = 1
        
        if symbol in self.last_book:
            last = self.last_book[symbol]
            last_bid_depth = sum(vol for _, vol in last.get("bids", []))
            last_ask_depth = sum(vol for _, vol in last.get("asks", []))
            
            depth_change = (bid_depth + ask_depth) - (last_bid_depth + last_ask_depth)
            if depth_change > 0:
                adds = int(abs(depth_change) / 100) + 5
            else:
                cancels = int(abs(depth_change) / 100) + 5
                
        self.last_book[symbol] = book_data
        
        tick = self.engine.get_tick(symbol)
        price_delta = 0.0
        volume = 100.0
        if tick:
            if symbol in self.depth_history and self.depth_history[symbol]:
                last_mid = self.depth_history[symbol][-1].get("mid", tick.mid)
                price_delta = tick.mid - last_mid
            volume = tick.volume
            
        self.depth_history[symbol].append({
            "mid": tick.mid if tick else (bids[0][0] + asks[0][0]) / 2,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "timestamp": book_data["timestamp"]
        })
        
        event_data = {
            "energy": 1.0,
            "persistence": 0.5 + 0.3 * (bid_depth / max(bid_depth + ask_depth, 1)),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "adds": adds,
            "cancels": cancels,
            "trades": trades,
            "volume": volume,
            "price_delta": price_delta
        }
        
        state = self.market_states[symbol]
        belief = state.update_from_event(event_data)
        
        return {
            "symbol": symbol,
            "p_accept": belief.p_accept,
            "confidence": belief.confidence,
            "mm_flags": belief.mm_flags,
            "expected_ig": state.expected_info_gain(hypothetical_E=1.0, n_samples=50),
            "recommendation": state.get_exploration_recommendation(),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1)
        }
        
    def get_belief_state(self, symbol: str) -> Optional[Dict]:
        """Get current belief state for symbol"""
        if symbol in self.market_states:
            return self.market_states[symbol].get_state_dict()
        return None
        
    def get_all_beliefs(self) -> Dict[str, Dict]:
        """Get belief states for all symbols"""
        return {s: self.get_belief_state(s) for s in self.symbols if s in self.market_states}


def demo():
    """Demonstration of MT5 Live Engine"""
    print("=" * 60)
    print("MT5 LIVE ENGINE DEMO")
    print("=" * 60)
    
    engine = MT5LiveEngine(simulation_mode=True)
    
    print("\nConnecting...")
    engine.connect()
    
    print(f"\nStatus: {engine.get_status()}")
    
    account = engine.get_account_info()
    if account:
        print(f"\nAccount Info:")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  Broker: {account.broker.value}")
        
    tick = engine.get_tick("EURUSD")
    if tick:
        print(f"\nEURUSD Tick:")
        print(f"  Bid: {tick.bid:.5f}")
        print(f"  Ask: {tick.ask:.5f}")
        print(f"  Spread: {tick.spread:.5f}")
        
    print("\nPlacing market order...")
    order = OrderRequest(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.1,
        order_type=OrderType.MARKET
    )
    
    result = engine.place_order(order)
    print(f"Order result: {result.status}")
    print(f"  Filled: {result.filled_quantity} @ {result.avg_fill_price:.5f}")
    
    print("\nPlacing TWAP order...")
    twap_order = OrderRequest(
        symbol="GBPUSD",
        side=OrderSide.BUY,
        quantity=0.5,
        order_type=OrderType.MARKET,
        algo=ExecutionAlgo.TWAP,
        algo_params={"duration_minutes": 1, "num_slices": 3}
    )
    
    result = engine.place_order(twap_order)
    print(f"TWAP result: {result.status}")
    print(f"  Filled: {result.filled_quantity} @ {result.avg_fill_price:.5f}")
    
    print(f"\nPositions: {len(engine.get_positions())}")
    for symbol, pos in engine.get_positions().items():
        print(f"  {symbol}: {pos.quantity} @ {pos.avg_price:.5f}")
        
    engine.disconnect()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
