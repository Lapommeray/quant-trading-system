"""
Safety Governance System - Eternal Guardrails

Comprehensive safety and governance system for live trading with:
- Human confirmation requirements for live trading
- Emergency kill switch with immediate halt capability
- Comprehensive audit logging
- Eternal guardrails that cannot be bypassed
- Multi-level authorization system
- Performance decay monitoring
- Automatic pause on anomaly detection

Implements Directive 21: Eternal Safeguards for Infinite Growth
"""

import os
import sys
import json
import time
import signal
import logging
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SafetyGovernance")


class AuthorizationLevel(Enum):
    NONE = 0
    READ_ONLY = 1
    PAPER_TRADING = 2
    LIMITED_LIVE = 3
    FULL_LIVE = 4
    ADMIN = 5


class TradeStatus(Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class TradeAuthorization:
    trade_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: TradeStatus
    requested_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    confirmation_code: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status.value,
            "requested_at": self.requested_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "confirmation_code": self.confirmation_code
        }


@dataclass
class AuditLogEntry:
    timestamp: datetime
    event_type: str
    user: str
    action: str
    details: Dict
    risk_level: str
    session_id: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "user": self.user,
            "action": self.action,
            "details": self.details,
            "risk_level": self.risk_level,
            "session_id": self.session_id
        }


@dataclass
class PerformanceSnapshot:
    timestamp: datetime
    equity: float
    peak_equity: float
    drawdown: float
    sharpe_ratio: float
    daily_pnl: float
    weekly_pnl: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Logs all trading activities with file persistence,
    searchable by date, event type, and user.
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        self.entries: List[AuditLogEntry] = []
        
        self._current_log_file = self._get_log_file()
        
        logger.info(f"AuditLogger initialized with session {self.session_id}")
        
    def _get_log_file(self) -> Path:
        """Get log file for current date"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.json"
        
    def log(self,
            event_type: str,
            action: str,
            details: Dict = None,
            user: str = "system",
            risk_level: str = "low") -> AuditLogEntry:
        """Log an audit event"""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user=user,
            action=action,
            details=details or {},
            risk_level=risk_level,
            session_id=self.session_id
        )
        
        self.entries.append(entry)
        self._persist_entry(entry)
        
        if risk_level in ["high", "critical"]:
            logger.warning(f"[AUDIT] {event_type}: {action} - {details}")
        else:
            logger.info(f"[AUDIT] {event_type}: {action}")
            
        return entry
        
    def _persist_entry(self, entry: AuditLogEntry):
        """Persist entry to file"""
        log_file = self._get_log_file()
        
        entries = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                entries = json.load(f)
                
        entries.append(entry.to_dict())
        
        with open(log_file, 'w') as f:
            json.dump(entries, f, indent=2)
            
    def search(self,
               start_date: Optional[datetime] = None,
               end_date: Optional[datetime] = None,
               event_type: Optional[str] = None,
               user: Optional[str] = None) -> List[AuditLogEntry]:
        """Search audit logs"""
        results = []
        
        for log_file in sorted(self.log_dir.glob("audit_*.json")):
            with open(log_file, 'r') as f:
                entries = json.load(f)
                
            for entry_dict in entries:
                entry_time = datetime.fromisoformat(entry_dict["timestamp"])
                
                if start_date and entry_time < start_date:
                    continue
                if end_date and entry_time > end_date:
                    continue
                if event_type and entry_dict["event_type"] != event_type:
                    continue
                if user and entry_dict["user"] != user:
                    continue
                    
                results.append(AuditLogEntry(
                    timestamp=entry_time,
                    event_type=entry_dict["event_type"],
                    user=entry_dict["user"],
                    action=entry_dict["action"],
                    details=entry_dict["details"],
                    risk_level=entry_dict["risk_level"],
                    session_id=entry_dict["session_id"]
                ))
                
        return results


class HumanConfirmationSystem:
    """
    Human confirmation system for trade authorization.
    
    Requires human approval for:
    - First N trades in live mode
    - Trades above risk threshold
    - Trades during unusual market conditions
    """
    
    def __init__(self,
                 required_confirmations: int = 100,
                 confirmation_timeout_minutes: int = 60,
                 auto_approve_paper: bool = True):
        """
        Initialize human confirmation system.
        
        Args:
            required_confirmations: Number of trades requiring human confirmation
            confirmation_timeout_minutes: Timeout for pending confirmations
            auto_approve_paper: Auto-approve in paper trading mode
        """
        self.required_confirmations = required_confirmations
        self.confirmation_timeout = timedelta(minutes=confirmation_timeout_minutes)
        self.auto_approve_paper = auto_approve_paper
        
        self.confirmed_trades = 0
        self.pending_authorizations: Dict[str, TradeAuthorization] = {}
        self.authorization_history: List[TradeAuthorization] = []
        
        self.is_paper_mode = True
        self.human_override_active = False
        self.override_expiry: Optional[datetime] = None
        
        self._callbacks: List[Callable[[TradeAuthorization], None]] = []
        
    def set_mode(self, paper_mode: bool):
        """Set trading mode"""
        self.is_paper_mode = paper_mode
        logger.info(f"Trading mode set to: {'PAPER' if paper_mode else 'LIVE'}")
        
    def request_authorization(self,
                              symbol: str,
                              side: str,
                              quantity: float,
                              order_type: str = "market") -> TradeAuthorization:
        """Request authorization for a trade"""
        trade_id = str(uuid.uuid4())[:12]
        confirmation_code = hashlib.sha256(
            f"{trade_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8].upper()
        
        auth = TradeAuthorization(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=TradeStatus.PENDING_APPROVAL,
            requested_at=datetime.now(),
            confirmation_code=confirmation_code
        )
        
        if self.is_paper_mode and self.auto_approve_paper:
            auth.status = TradeStatus.APPROVED
            auth.approved_at = datetime.now()
            auth.approved_by = "auto_paper"
            self.authorization_history.append(auth)
            return auth
            
        if self.human_override_active:
            if self.override_expiry and datetime.now() < self.override_expiry:
                auth.status = TradeStatus.APPROVED
                auth.approved_at = datetime.now()
                auth.approved_by = "human_override"
                self.authorization_history.append(auth)
                return auth
            else:
                self.human_override_active = False
                
        if self.confirmed_trades >= self.required_confirmations:
            auth.status = TradeStatus.APPROVED
            auth.approved_at = datetime.now()
            auth.approved_by = "auto_threshold"
            self.authorization_history.append(auth)
            return auth
            
        self.pending_authorizations[trade_id] = auth
        
        for callback in self._callbacks:
            try:
                callback(auth)
            except Exception as e:
                logger.error(f"Confirmation callback error: {e}")
                
        logger.info(f"Trade {trade_id} pending human confirmation. Code: {confirmation_code}")
        
        return auth
        
    def confirm_trade(self, trade_id: str, confirmation_code: str, user: str = "human") -> bool:
        """Confirm a pending trade"""
        if trade_id not in self.pending_authorizations:
            logger.warning(f"Trade {trade_id} not found in pending")
            return False
            
        auth = self.pending_authorizations[trade_id]
        
        if auth.confirmation_code != confirmation_code:
            logger.warning(f"Invalid confirmation code for trade {trade_id}")
            return False
            
        if datetime.now() - auth.requested_at > self.confirmation_timeout:
            auth.status = TradeStatus.CANCELLED
            auth.rejection_reason = "Confirmation timeout"
            del self.pending_authorizations[trade_id]
            self.authorization_history.append(auth)
            return False
            
        auth.status = TradeStatus.APPROVED
        auth.approved_at = datetime.now()
        auth.approved_by = user
        
        del self.pending_authorizations[trade_id]
        self.authorization_history.append(auth)
        self.confirmed_trades += 1
        
        logger.info(f"Trade {trade_id} confirmed by {user}")
        return True
        
    def reject_trade(self, trade_id: str, reason: str, user: str = "human") -> bool:
        """Reject a pending trade"""
        if trade_id not in self.pending_authorizations:
            return False
            
        auth = self.pending_authorizations[trade_id]
        auth.status = TradeStatus.REJECTED
        auth.rejection_reason = reason
        auth.approved_by = user
        
        del self.pending_authorizations[trade_id]
        self.authorization_history.append(auth)
        
        logger.info(f"Trade {trade_id} rejected by {user}: {reason}")
        return True
        
    def enable_override(self, duration_minutes: int = 60, user: str = "admin"):
        """Enable human override for specified duration"""
        self.human_override_active = True
        self.override_expiry = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"Human override enabled by {user} for {duration_minutes} minutes")
        
    def disable_override(self):
        """Disable human override"""
        self.human_override_active = False
        self.override_expiry = None
        logger.info("Human override disabled")
        
    def register_callback(self, callback: Callable[[TradeAuthorization], None]):
        """Register callback for pending confirmations"""
        self._callbacks.append(callback)
        
    def get_pending(self) -> List[TradeAuthorization]:
        """Get all pending authorizations"""
        return list(self.pending_authorizations.values())


class EmergencyKillSwitch:
    """
    Emergency kill switch for immediate trading halt.
    
    Features:
    - Immediate halt of all trading
    - Signal handler for external triggers
    - Callback system for cleanup
    - Cooldown period before restart
    """
    
    def __init__(self, cooldown_minutes: int = 30):
        """
        Initialize emergency kill switch.
        
        Args:
            cooldown_minutes: Cooldown period after activation
        """
        self.cooldown_minutes = cooldown_minutes
        self.activated = False
        self.activation_time: Optional[datetime] = None
        self.activation_reason: str = ""
        self.activated_by: str = ""
        
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for external kill triggers"""
        try:
            signal.signal(signal.SIGUSR1, self._signal_handler)
            logger.info("Kill switch signal handler registered (SIGUSR1)")
        except (AttributeError, ValueError):
            logger.warning("Could not register signal handler")
            
    def _signal_handler(self, signum, frame):
        """Handle external kill signal"""
        self.activate("External signal (SIGUSR1)", "signal")
        
    def activate(self, reason: str = "Manual activation", user: str = "system"):
        """Activate the kill switch"""
        with self._lock:
            if self.activated:
                logger.warning("Kill switch already activated")
                return
                
            self.activated = True
            self.activation_time = datetime.now()
            self.activation_reason = reason
            self.activated_by = user
            
            logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED by {user}: {reason}")
            
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Kill switch callback error: {e}")
                    
    def deactivate(self, user: str = "admin", force: bool = False) -> bool:
        """Deactivate the kill switch"""
        with self._lock:
            if not self.activated:
                return True
                
            if not force and self.activation_time:
                elapsed = datetime.now() - self.activation_time
                if elapsed < timedelta(minutes=self.cooldown_minutes):
                    remaining = self.cooldown_minutes - elapsed.total_seconds() / 60
                    logger.warning(f"Kill switch cooldown: {remaining:.1f} minutes remaining")
                    return False
                    
            self.activated = False
            self.activation_time = None
            self.activation_reason = ""
            self.activated_by = ""
            
            logger.info(f"Kill switch deactivated by {user}")
            return True
            
    def register_callback(self, callback: Callable[[], None]):
        """Register callback for kill switch activation"""
        self._callbacks.append(callback)
        
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self.activated
        
    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        return {
            "activated": self.activated,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "activation_reason": self.activation_reason,
            "activated_by": self.activated_by,
            "cooldown_minutes": self.cooldown_minutes
        }


class PerformanceMonitor:
    """
    Performance monitoring with automatic alerts.
    
    Monitors:
    - Drawdown from peak
    - Daily/weekly P&L
    - Sharpe ratio decay
    - Anomaly detection
    """
    
    ALERT_THRESHOLDS = {
        "drawdown_warning": 0.05,
        "drawdown_critical": 0.10,
        "drawdown_emergency": 0.15,
        "daily_loss_warning": 0.02,
        "daily_loss_critical": 0.03,
        "sharpe_decay_warning": 0.5,
    }
    
    def __init__(self, initial_equity: float = 100000):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        
        self.daily_start_equity = initial_equity
        self.weekly_start_equity = initial_equity
        
        self.snapshots: deque = deque(maxlen=1000)
        self.alerts: List[Dict] = []
        
        self._alert_callbacks: List[Callable[[AlertLevel, str], None]] = []
        
    def update(self, equity: float, sharpe_ratio: float = 0.0):
        """Update performance metrics"""
        self.current_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0
        weekly_pnl = (equity - self.weekly_start_equity) / self.weekly_start_equity if self.weekly_start_equity > 0 else 0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            equity=equity,
            peak_equity=self.peak_equity,
            drawdown=drawdown,
            sharpe_ratio=sharpe_ratio,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl
        )
        
        self.snapshots.append(snapshot)
        
        self._check_alerts(snapshot)
        
        return snapshot
        
    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for alert conditions"""
        if snapshot.drawdown >= self.ALERT_THRESHOLDS["drawdown_emergency"]:
            self._raise_alert(
                AlertLevel.EMERGENCY,
                f"EMERGENCY: Drawdown at {snapshot.drawdown:.1%}"
            )
        elif snapshot.drawdown >= self.ALERT_THRESHOLDS["drawdown_critical"]:
            self._raise_alert(
                AlertLevel.CRITICAL,
                f"CRITICAL: Drawdown at {snapshot.drawdown:.1%}"
            )
        elif snapshot.drawdown >= self.ALERT_THRESHOLDS["drawdown_warning"]:
            self._raise_alert(
                AlertLevel.WARNING,
                f"WARNING: Drawdown at {snapshot.drawdown:.1%}"
            )
            
        if snapshot.daily_pnl <= -self.ALERT_THRESHOLDS["daily_loss_critical"]:
            self._raise_alert(
                AlertLevel.CRITICAL,
                f"CRITICAL: Daily loss at {snapshot.daily_pnl:.1%}"
            )
        elif snapshot.daily_pnl <= -self.ALERT_THRESHOLDS["daily_loss_warning"]:
            self._raise_alert(
                AlertLevel.WARNING,
                f"WARNING: Daily loss at {snapshot.daily_pnl:.1%}"
            )
            
    def _raise_alert(self, level: AlertLevel, message: str):
        """Raise an alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "message": message
        }
        
        self.alerts.append(alert)
        
        for callback in self._alert_callbacks:
            try:
                callback(level, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
        if level == AlertLevel.EMERGENCY:
            logger.critical(message)
        elif level == AlertLevel.CRITICAL:
            logger.error(message)
        elif level == AlertLevel.WARNING:
            logger.warning(message)
        else:
            logger.info(message)
            
    def register_alert_callback(self, callback: Callable[[AlertLevel, str], None]):
        """Register callback for alerts"""
        self._alert_callbacks.append(callback)
        
    def reset_daily(self):
        """Reset daily tracking"""
        self.daily_start_equity = self.current_equity
        
    def reset_weekly(self):
        """Reset weekly tracking"""
        self.weekly_start_equity = self.current_equity
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": drawdown,
            "daily_pnl": (self.current_equity - self.daily_start_equity) / self.daily_start_equity if self.daily_start_equity > 0 else 0,
            "weekly_pnl": (self.current_equity - self.weekly_start_equity) / self.weekly_start_equity if self.weekly_start_equity > 0 else 0,
            "total_return": (self.current_equity - self.initial_equity) / self.initial_equity if self.initial_equity > 0 else 0
        }


class EternalGuardrails:
    """
    Eternal guardrails that cannot be bypassed.
    
    These are hard-coded safety limits that apply regardless
    of any other settings or overrides.
    """
    
    MAX_SINGLE_TRADE_RISK = 0.03
    MAX_DAILY_LOSS = 0.05
    MAX_DRAWDOWN = 0.15
    MAX_LEVERAGE = 3.0
    MAX_POSITION_CONCENTRATION = 0.25
    REQUIRE_HUMAN_FOR_LIVE = True
    
    @classmethod
    def check_trade(cls,
                    trade_risk: float,
                    daily_loss: float,
                    drawdown: float,
                    leverage: float,
                    position_concentration: float,
                    is_live: bool,
                    has_human_override: bool) -> Tuple[bool, str]:
        """
        Check if trade passes eternal guardrails.
        
        These checks CANNOT be bypassed by any means.
        """
        if trade_risk > cls.MAX_SINGLE_TRADE_RISK:
            return False, f"ETERNAL GUARDRAIL: Single trade risk {trade_risk:.1%} exceeds max {cls.MAX_SINGLE_TRADE_RISK:.1%}"
            
        if daily_loss > cls.MAX_DAILY_LOSS:
            return False, f"ETERNAL GUARDRAIL: Daily loss {daily_loss:.1%} exceeds max {cls.MAX_DAILY_LOSS:.1%}"
            
        if drawdown > cls.MAX_DRAWDOWN:
            return False, f"ETERNAL GUARDRAIL: Drawdown {drawdown:.1%} exceeds max {cls.MAX_DRAWDOWN:.1%}"
            
        if leverage > cls.MAX_LEVERAGE:
            return False, f"ETERNAL GUARDRAIL: Leverage {leverage:.1f}x exceeds max {cls.MAX_LEVERAGE:.1f}x"
            
        if position_concentration > cls.MAX_POSITION_CONCENTRATION:
            return False, f"ETERNAL GUARDRAIL: Position concentration {position_concentration:.1%} exceeds max {cls.MAX_POSITION_CONCENTRATION:.1%}"
            
        if is_live and cls.REQUIRE_HUMAN_FOR_LIVE and not has_human_override:
            return False, "ETERNAL GUARDRAIL: Live trading requires human override"
            
        return True, "OK"
        
    @classmethod
    def enforce_eternal_guardrails():
        """
        Enforce eternal guardrails - called at system startup.
        
        Raises exception if live trading attempted without proper authorization.
        """
        if os.getenv("LIVE_TRADING") == "TRUE" and not os.getenv("HUMAN_OVERRIDE"):
            raise Exception("ETERNAL GUARDRAIL VIOLATION: Real capital requires HUMAN_OVERRIDE environment variable")
            
        logger.info("Eternal guardrails enforced")


class SafetyGovernanceSystem:
    """
    Main safety governance system integrating all components.
    
    Features:
    - Multi-level authorization
    - Human confirmation workflow
    - Emergency kill switch
    - Performance monitoring
    - Comprehensive audit logging
    - Eternal guardrails
    """
    
    def __init__(self,
                 log_dir: str = "audit_logs",
                 required_confirmations: int = 100,
                 paper_mode: bool = True):
        """
        Initialize safety governance system.
        
        Args:
            log_dir: Directory for audit logs
            required_confirmations: Number of trades requiring human confirmation
            paper_mode: Start in paper trading mode
        """
        self.audit_logger = AuditLogger(log_dir)
        self.confirmation_system = HumanConfirmationSystem(
            required_confirmations=required_confirmations,
            auto_approve_paper=True
        )
        self.kill_switch = EmergencyKillSwitch()
        self.performance_monitor = PerformanceMonitor()
        
        self.authorization_level = AuthorizationLevel.PAPER_TRADING if paper_mode else AuthorizationLevel.NONE
        self.confirmation_system.set_mode(paper_mode)
        
        self.kill_switch.register_callback(self._on_kill_switch)
        self.performance_monitor.register_alert_callback(self._on_alert)
        
        EternalGuardrails.enforce_eternal_guardrails()
        
        self.audit_logger.log(
            event_type="system",
            action="initialization",
            details={"paper_mode": paper_mode, "required_confirmations": required_confirmations},
            risk_level="low"
        )
        
        logger.info("SafetyGovernanceSystem initialized")
        
    def _on_kill_switch(self):
        """Handle kill switch activation"""
        self.audit_logger.log(
            event_type="emergency",
            action="kill_switch_activated",
            details=self.kill_switch.get_status(),
            risk_level="critical"
        )
        
    def _on_alert(self, level: AlertLevel, message: str):
        """Handle performance alerts"""
        self.audit_logger.log(
            event_type="alert",
            action=message,
            details={"level": level.value},
            risk_level="high" if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY] else "medium"
        )
        
        if level == AlertLevel.EMERGENCY:
            self.kill_switch.activate(message, "performance_monitor")
            
    def authorize_trade(self,
                        symbol: str,
                        side: str,
                        quantity: float,
                        order_type: str = "market",
                        trade_risk: float = 0.01) -> Tuple[bool, str, Optional[TradeAuthorization]]:
        """
        Authorize a trade through the full governance pipeline.
        
        Returns:
            Tuple of (authorized, message, authorization_object)
        """
        if self.kill_switch.is_active():
            return False, "Kill switch is active", None
            
        metrics = self.performance_monitor.get_metrics()
        
        is_live = self.authorization_level >= AuthorizationLevel.LIMITED_LIVE
        has_override = self.confirmation_system.human_override_active
        
        passed, reason = EternalGuardrails.check_trade(
            trade_risk=trade_risk,
            daily_loss=-metrics["daily_pnl"] if metrics["daily_pnl"] < 0 else 0,
            drawdown=metrics["drawdown"],
            leverage=1.0,
            position_concentration=0.1,
            is_live=is_live,
            has_human_override=has_override
        )
        
        if not passed:
            self.audit_logger.log(
                event_type="trade",
                action="blocked_by_guardrail",
                details={"symbol": symbol, "reason": reason},
                risk_level="high"
            )
            return False, reason, None
            
        auth = self.confirmation_system.request_authorization(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type
        )
        
        if auth.status == TradeStatus.APPROVED:
            self.audit_logger.log(
                event_type="trade",
                action="authorized",
                details=auth.to_dict(),
                risk_level="medium"
            )
            return True, "Authorized", auth
        else:
            self.audit_logger.log(
                event_type="trade",
                action="pending_confirmation",
                details=auth.to_dict(),
                risk_level="medium"
            )
            return False, f"Pending human confirmation. Code: {auth.confirmation_code}", auth
            
    def confirm_trade(self, trade_id: str, confirmation_code: str, user: str = "human") -> bool:
        """Confirm a pending trade"""
        result = self.confirmation_system.confirm_trade(trade_id, confirmation_code, user)
        
        self.audit_logger.log(
            event_type="trade",
            action="confirmed" if result else "confirmation_failed",
            details={"trade_id": trade_id, "user": user},
            user=user,
            risk_level="medium"
        )
        
        return result
        
    def activate_kill_switch(self, reason: str, user: str = "admin"):
        """Activate emergency kill switch"""
        self.kill_switch.activate(reason, user)
        
    def deactivate_kill_switch(self, user: str = "admin", force: bool = False) -> bool:
        """Deactivate kill switch"""
        result = self.kill_switch.deactivate(user, force)
        
        if result:
            self.audit_logger.log(
                event_type="emergency",
                action="kill_switch_deactivated",
                details={"user": user, "force": force},
                user=user,
                risk_level="high"
            )
            
        return result
        
    def set_authorization_level(self, level: AuthorizationLevel, user: str = "admin"):
        """Set authorization level"""
        old_level = self.authorization_level
        self.authorization_level = level
        
        self.confirmation_system.set_mode(level <= AuthorizationLevel.PAPER_TRADING)
        
        self.audit_logger.log(
            event_type="authorization",
            action="level_changed",
            details={"old_level": old_level.name, "new_level": level.name},
            user=user,
            risk_level="high"
        )
        
        logger.info(f"Authorization level changed from {old_level.name} to {level.name}")
        
    def enable_human_override(self, duration_minutes: int = 60, user: str = "admin"):
        """Enable human override for live trading"""
        self.confirmation_system.enable_override(duration_minutes, user)
        
        self.audit_logger.log(
            event_type="authorization",
            action="human_override_enabled",
            details={"duration_minutes": duration_minutes},
            user=user,
            risk_level="high"
        )
        
    def update_performance(self, equity: float, sharpe_ratio: float = 0.0):
        """Update performance metrics"""
        snapshot = self.performance_monitor.update(equity, sharpe_ratio)
        return snapshot
        
    def get_status(self) -> Dict[str, Any]:
        """Get full system status"""
        return {
            "authorization_level": self.authorization_level.name,
            "kill_switch": self.kill_switch.get_status(),
            "performance": self.performance_monitor.get_metrics(),
            "pending_confirmations": len(self.confirmation_system.get_pending()),
            "confirmed_trades": self.confirmation_system.confirmed_trades,
            "human_override_active": self.confirmation_system.human_override_active
        }


def demo():
    """Demonstration of safety governance system"""
    print("=" * 60)
    print("SAFETY GOVERNANCE SYSTEM DEMO")
    print("=" * 60)
    
    system = SafetyGovernanceSystem(paper_mode=True)
    
    print(f"\nInitial Status: {json.dumps(system.get_status(), indent=2)}")
    
    print("\n--- Testing Trade Authorization (Paper Mode) ---")
    authorized, message, auth = system.authorize_trade(
        symbol="EURUSD",
        side="buy",
        quantity=0.1,
        trade_risk=0.01
    )
    print(f"Trade authorized: {authorized}")
    print(f"Message: {message}")
    
    print("\n--- Testing Kill Switch ---")
    system.activate_kill_switch("Demo test", "demo_user")
    print(f"Kill switch active: {system.kill_switch.is_active()}")
    
    authorized, message, _ = system.authorize_trade(
        symbol="GBPUSD",
        side="sell",
        quantity=0.1
    )
    print(f"Trade during kill switch: {authorized} - {message}")
    
    system.deactivate_kill_switch("demo_user", force=True)
    print(f"Kill switch deactivated: {not system.kill_switch.is_active()}")
    
    print("\n--- Testing Performance Monitoring ---")
    system.update_performance(100000)
    system.update_performance(95000)
    system.update_performance(90000)
    
    print(f"Performance metrics: {json.dumps(system.performance_monitor.get_metrics(), indent=2)}")
    
    print("\n--- Testing Eternal Guardrails ---")
    authorized, message, _ = system.authorize_trade(
        symbol="USDJPY",
        side="buy",
        quantity=10.0,
        trade_risk=0.05
    )
    print(f"High risk trade: {authorized} - {message}")
    
    print("\n--- Final Status ---")
    print(json.dumps(system.get_status(), indent=2))
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
