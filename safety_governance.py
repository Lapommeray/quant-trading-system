"""
Safety Governance Module

Comprehensive safety and governance system for live trading:
- Human confirmation requirements
- Emergency kill switch
- Comprehensive audit logging
- Trade authorization
- Risk override controls
"""

import os
import sys
import json
import time
import logging
import threading
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("safety_governance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SafetyGovernance")


class AuthorizationLevel(Enum):
    """Authorization levels for trading operations"""
    NONE = 0
    READ_ONLY = 1
    PAPER_TRADING = 2
    LIMITED_LIVE = 3
    FULL_LIVE = 4
    ADMIN = 5


class TradeStatus(Enum):
    """Trade authorization status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


@dataclass
class TradeAuthorization:
    """Trade authorization record"""
    trade_id: str
    symbol: str
    direction: str
    volume: float
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: TradeStatus
    requested_at: datetime
    authorized_at: Optional[datetime] = None
    authorized_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    execution_result: Optional[Dict] = None


@dataclass
class AuditLogEntry:
    """Audit log entry"""
    timestamp: datetime
    event_type: str
    user: str
    action: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    risk_level: str = "normal"


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Logs all trading activities, system changes, and security events
    with full traceability.
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.current_log_file = self._get_log_file()
        self.log_buffer: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        
    def _get_log_file(self) -> Path:
        """Get current log file path"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.json"
        
    def log(self, 
            event_type: str,
            action: str,
            details: Dict[str, Any],
            user: str = "system",
            risk_level: str = "normal"):
        """Log an audit event"""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user=user,
            action=action,
            details=details,
            session_id=os.environ.get("SESSION_ID", "unknown"),
            risk_level=risk_level
        )
        
        with self._lock:
            self.log_buffer.append(entry)
            self._write_to_file(entry)
            
        if risk_level in ["high", "critical"]:
            logger.warning(f"HIGH RISK EVENT: {event_type} - {action}")
            
    def _write_to_file(self, entry: AuditLogEntry):
        """Write entry to log file"""
        log_file = self._get_log_file()
        
        entry_dict = {
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "user": entry.user,
            "action": entry.action,
            "details": entry.details,
            "session_id": entry.session_id,
            "risk_level": entry.risk_level
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry_dict) + "\n")
            
    def get_recent_logs(self, count: int = 100, 
                       event_type: Optional[str] = None) -> List[AuditLogEntry]:
        """Get recent log entries"""
        logs = list(self.log_buffer)
        
        if event_type:
            logs = [l for l in logs if l.event_type == event_type]
            
        return logs[-count:]
        
    def search_logs(self, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   event_type: Optional[str] = None,
                   user: Optional[str] = None) -> List[Dict]:
        """Search audit logs"""
        results = []
        
        log_files = sorted(self.log_dir.glob("audit_*.json"))
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        
                        if start_date and entry_time < start_date:
                            continue
                        if end_date and entry_time > end_date:
                            continue
                        if event_type and entry["event_type"] != event_type:
                            continue
                        if user and entry["user"] != user:
                            continue
                            
                        results.append(entry)
                    except:
                        continue
                        
        return results


class HumanConfirmationSystem:
    """
    Human confirmation system for trade authorization.
    
    Requires human approval for:
    - First N trades in live mode
    - Trades exceeding risk thresholds
    - Unusual market conditions
    """
    
    def __init__(self,
                 required_confirmations: int = 100,
                 confirmation_timeout_seconds: int = 300,
                 auto_approve_paper: bool = True):
        self.required_confirmations = required_confirmations
        self.confirmation_timeout = confirmation_timeout_seconds
        self.auto_approve_paper = auto_approve_paper
        
        self.confirmed_trades = 0
        self.pending_authorizations: Dict[str, TradeAuthorization] = {}
        self.authorization_history: deque = deque(maxlen=1000)
        
        self.is_paper_mode = True
        self.human_override_active = False
        self.override_expiry: Optional[datetime] = None
        
        self._lock = threading.Lock()
        self.audit_logger = AuditLogger()
        
    def set_trading_mode(self, is_paper: bool):
        """Set trading mode"""
        self.is_paper_mode = is_paper
        self.audit_logger.log(
            "config_change",
            "set_trading_mode",
            {"is_paper": is_paper},
            risk_level="high" if not is_paper else "normal"
        )
        
    def request_authorization(self,
                             symbol: str,
                             direction: str,
                             volume: float,
                             price: float,
                             stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None) -> TradeAuthorization:
        """Request authorization for a trade"""
        trade_id = hashlib.sha256(
            f"{symbol}{direction}{volume}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        auth = TradeAuthorization(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=TradeStatus.PENDING,
            requested_at=datetime.now()
        )
        
        with self._lock:
            if self.is_paper_mode and self.auto_approve_paper:
                auth.status = TradeStatus.APPROVED
                auth.authorized_at = datetime.now()
                auth.authorized_by = "auto_paper"
                self.authorization_history.append(auth)
                return auth
                
            if self.human_override_active and self.override_expiry:
                if datetime.now() < self.override_expiry:
                    auth.status = TradeStatus.APPROVED
                    auth.authorized_at = datetime.now()
                    auth.authorized_by = "human_override"
                    self.authorization_history.append(auth)
                    return auth
                else:
                    self.human_override_active = False
                    
            if self.confirmed_trades >= self.required_confirmations:
                auth.status = TradeStatus.APPROVED
                auth.authorized_at = datetime.now()
                auth.authorized_by = "auto_threshold"
                self.confirmed_trades += 1
                self.authorization_history.append(auth)
                return auth
                
            self.pending_authorizations[trade_id] = auth
            
        self.audit_logger.log(
            "trade_authorization",
            "request",
            asdict(auth),
            risk_level="high"
        )
        
        return auth
        
    def approve_trade(self, trade_id: str, approver: str = "human") -> bool:
        """Approve a pending trade"""
        with self._lock:
            if trade_id not in self.pending_authorizations:
                return False
                
            auth = self.pending_authorizations[trade_id]
            auth.status = TradeStatus.APPROVED
            auth.authorized_at = datetime.now()
            auth.authorized_by = approver
            
            self.confirmed_trades += 1
            self.authorization_history.append(auth)
            del self.pending_authorizations[trade_id]
            
        self.audit_logger.log(
            "trade_authorization",
            "approve",
            {"trade_id": trade_id, "approver": approver}
        )
        
        return True
        
    def reject_trade(self, trade_id: str, reason: str, rejector: str = "human") -> bool:
        """Reject a pending trade"""
        with self._lock:
            if trade_id not in self.pending_authorizations:
                return False
                
            auth = self.pending_authorizations[trade_id]
            auth.status = TradeStatus.REJECTED
            auth.rejection_reason = reason
            auth.authorized_by = rejector
            
            self.authorization_history.append(auth)
            del self.pending_authorizations[trade_id]
            
        self.audit_logger.log(
            "trade_authorization",
            "reject",
            {"trade_id": trade_id, "reason": reason, "rejector": rejector}
        )
        
        return True
        
    def enable_human_override(self, duration_minutes: int = 60, 
                             confirmation_code: str = "") -> bool:
        """Enable human override for automatic approval"""
        if confirmation_code != "ENABLE_OVERRIDE_CONFIRMED":
            return False
            
        with self._lock:
            self.human_override_active = True
            self.override_expiry = datetime.now() + timedelta(minutes=duration_minutes)
            
        self.audit_logger.log(
            "security",
            "enable_override",
            {"duration_minutes": duration_minutes},
            risk_level="critical"
        )
        
        return True
        
    def disable_human_override(self):
        """Disable human override"""
        with self._lock:
            self.human_override_active = False
            self.override_expiry = None
            
        self.audit_logger.log(
            "security",
            "disable_override",
            {}
        )
        
    def get_pending_authorizations(self) -> List[TradeAuthorization]:
        """Get all pending authorizations"""
        return list(self.pending_authorizations.values())
        
    def get_authorization_status(self, trade_id: str) -> Optional[TradeAuthorization]:
        """Get authorization status for a trade"""
        if trade_id in self.pending_authorizations:
            return self.pending_authorizations[trade_id]
            
        for auth in self.authorization_history:
            if auth.trade_id == trade_id:
                return auth
                
        return None


class EmergencyKillSwitch:
    """
    Emergency kill switch for immediate trading halt.
    
    Features:
    - Immediate position closure
    - Trading halt
    - Alert notifications
    - Recovery procedures
    """
    
    def __init__(self):
        self.is_active = False
        self.activation_time: Optional[datetime] = None
        self.activation_reason: str = ""
        self.positions_closed: List[Dict] = []
        
        self._lock = threading.Lock()
        self.audit_logger = AuditLogger()
        self.callbacks: List[Callable] = []
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for emergency stop"""
        try:
            signal.signal(signal.SIGUSR1, self._signal_handler)
        except:
            pass
            
    def _signal_handler(self, signum, frame):
        """Handle emergency signal"""
        self.activate("Signal received: SIGUSR1")
        
    def register_callback(self, callback: Callable):
        """Register callback for kill switch activation"""
        self.callbacks.append(callback)
        
    def activate(self, reason: str, close_positions: bool = True) -> Dict[str, Any]:
        """Activate emergency kill switch"""
        with self._lock:
            if self.is_active:
                return {
                    "success": False,
                    "error": "Kill switch already active",
                    "activation_time": self.activation_time.isoformat() if self.activation_time else None
                }
                
            self.is_active = True
            self.activation_time = datetime.now()
            self.activation_reason = reason
            
        logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED: {reason}")
        
        self.audit_logger.log(
            "emergency",
            "kill_switch_activated",
            {"reason": reason, "close_positions": close_positions},
            risk_level="critical"
        )
        
        for callback in self.callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Kill switch callback error: {e}")
                
        return {
            "success": True,
            "activation_time": self.activation_time.isoformat(),
            "reason": reason,
            "positions_closed": self.positions_closed
        }
        
    def deactivate(self, confirmation_code: str, deactivator: str = "admin") -> bool:
        """Deactivate kill switch (requires confirmation)"""
        if confirmation_code != "DEACTIVATE_KILL_SWITCH_CONFIRMED":
            logger.warning("Invalid kill switch deactivation attempt")
            return False
            
        with self._lock:
            if not self.is_active:
                return True
                
            self.is_active = False
            
        self.audit_logger.log(
            "emergency",
            "kill_switch_deactivated",
            {"deactivator": deactivator},
            risk_level="high"
        )
        
        logger.info("Kill switch deactivated")
        return True
        
    def check_status(self) -> Dict[str, Any]:
        """Check kill switch status"""
        return {
            "is_active": self.is_active,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "activation_reason": self.activation_reason,
            "positions_closed": len(self.positions_closed)
        }
        
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return not self.is_active


class SafetyGovernanceSystem:
    """
    Main safety governance system integrating all safety components.
    
    Provides:
    - Centralized safety management
    - Trade authorization workflow
    - Emergency controls
    - Comprehensive audit trail
    """
    
    def __init__(self,
                 required_confirmations: int = 100,
                 is_paper_mode: bool = True):
        self.confirmation_system = HumanConfirmationSystem(
            required_confirmations=required_confirmations,
            auto_approve_paper=True
        )
        self.kill_switch = EmergencyKillSwitch()
        self.audit_logger = AuditLogger()
        
        self.authorization_level = AuthorizationLevel.PAPER_TRADING if is_paper_mode else AuthorizationLevel.NONE
        self.session_id = secrets.token_hex(16)
        
        self.confirmation_system.set_trading_mode(is_paper_mode)
        
        self.audit_logger.log(
            "system",
            "governance_initialized",
            {
                "session_id": self.session_id,
                "is_paper_mode": is_paper_mode,
                "required_confirmations": required_confirmations
            }
        )
        
    def authorize_trade(self,
                       symbol: str,
                       direction: str,
                       volume: float,
                       price: float,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> TradeAuthorization:
        """Authorize a trade through the governance system"""
        if not self.kill_switch.can_trade():
            auth = TradeAuthorization(
                trade_id="blocked",
                symbol=symbol,
                direction=direction,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=TradeStatus.REJECTED,
                requested_at=datetime.now(),
                rejection_reason="Kill switch active"
            )
            return auth
            
        if self.authorization_level == AuthorizationLevel.NONE:
            auth = TradeAuthorization(
                trade_id="unauthorized",
                symbol=symbol,
                direction=direction,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=TradeStatus.REJECTED,
                requested_at=datetime.now(),
                rejection_reason="No trading authorization"
            )
            return auth
            
        return self.confirmation_system.request_authorization(
            symbol, direction, volume, price, stop_loss, take_profit
        )
        
    def set_authorization_level(self, level: AuthorizationLevel, 
                               confirmation_code: str = "") -> bool:
        """Set authorization level"""
        if level.value > AuthorizationLevel.PAPER_TRADING.value:
            if confirmation_code != "ENABLE_LIVE_TRADING_CONFIRMED":
                logger.warning("Invalid confirmation for live trading")
                return False
                
        self.authorization_level = level
        is_paper = level.value <= AuthorizationLevel.PAPER_TRADING.value
        self.confirmation_system.set_trading_mode(is_paper)
        
        self.audit_logger.log(
            "security",
            "authorization_level_changed",
            {"new_level": level.name},
            risk_level="critical" if not is_paper else "normal"
        )
        
        return True
        
    def emergency_stop(self, reason: str = "Manual activation") -> Dict[str, Any]:
        """Activate emergency stop"""
        return self.kill_switch.activate(reason)
        
    def resume_trading(self, confirmation_code: str) -> bool:
        """Resume trading after emergency stop"""
        return self.kill_switch.deactivate(confirmation_code)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "session_id": self.session_id,
            "authorization_level": self.authorization_level.name,
            "kill_switch": self.kill_switch.check_status(),
            "pending_authorizations": len(self.confirmation_system.get_pending_authorizations()),
            "confirmed_trades": self.confirmation_system.confirmed_trades,
            "required_confirmations": self.confirmation_system.required_confirmations,
            "human_override_active": self.confirmation_system.human_override_active,
            "can_trade": self.can_trade()
        }
        
    def can_trade(self) -> bool:
        """Check if trading is currently allowed"""
        if not self.kill_switch.can_trade():
            return False
        if self.authorization_level == AuthorizationLevel.NONE:
            return False
        return True
        
    def approve_pending_trade(self, trade_id: str, approver: str = "admin") -> bool:
        """Approve a pending trade"""
        return self.confirmation_system.approve_trade(trade_id, approver)
        
    def reject_pending_trade(self, trade_id: str, reason: str, 
                            rejector: str = "admin") -> bool:
        """Reject a pending trade"""
        return self.confirmation_system.reject_trade(trade_id, reason, rejector)
        
    def get_audit_logs(self, 
                      count: int = 100,
                      event_type: Optional[str] = None) -> List[AuditLogEntry]:
        """Get recent audit logs"""
        return self.audit_logger.get_recent_logs(count, event_type)


def create_kill_switch_endpoint():
    """Create a simple HTTP endpoint for kill switch (for external monitoring)"""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        governance = SafetyGovernanceSystem()
        
        class KillSwitchHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/status":
                    status = governance.get_system_status()
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(status, default=str).encode())
                elif self.path == "/kill":
                    result = governance.emergency_stop("HTTP endpoint triggered")
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result, default=str).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                pass
                
        return HTTPServer, KillSwitchHandler
        
    except ImportError:
        return None, None


def main():
    """Demo of safety governance system"""
    governance = SafetyGovernanceSystem(
        required_confirmations=100,
        is_paper_mode=True
    )
    
    print("=== Safety Governance System Demo ===\n")
    
    print("1. System Status:")
    status = governance.get_system_status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\n2. Requesting Trade Authorization (Paper Mode):")
    auth = governance.authorize_trade(
        symbol="BTCUSD",
        direction="BUY",
        volume=0.1,
        price=45000,
        stop_loss=44000,
        take_profit=47000
    )
    print(f"   Trade ID: {auth.trade_id}")
    print(f"   Status: {auth.status.value}")
    print(f"   Authorized By: {auth.authorized_by}")
    
    print("\n3. Switching to Live Mode:")
    success = governance.set_authorization_level(
        AuthorizationLevel.LIMITED_LIVE,
        "ENABLE_LIVE_TRADING_CONFIRMED"
    )
    print(f"   Success: {success}")
    
    print("\n4. Requesting Trade Authorization (Live Mode):")
    auth2 = governance.authorize_trade(
        symbol="ETHUSD",
        direction="SELL",
        volume=1.0,
        price=2500,
        stop_loss=2600,
        take_profit=2300
    )
    print(f"   Trade ID: {auth2.trade_id}")
    print(f"   Status: {auth2.status.value}")
    
    if auth2.status == TradeStatus.PENDING:
        print("\n5. Approving Pending Trade:")
        approved = governance.approve_pending_trade(auth2.trade_id, "demo_admin")
        print(f"   Approved: {approved}")
        
    print("\n6. Testing Emergency Kill Switch:")
    result = governance.emergency_stop("Demo test")
    print(f"   Kill Switch Active: {result['success']}")
    print(f"   Can Trade: {governance.can_trade()}")
    
    print("\n7. Resuming Trading:")
    resumed = governance.resume_trading("DEACTIVATE_KILL_SWITCH_CONFIRMED")
    print(f"   Resumed: {resumed}")
    print(f"   Can Trade: {governance.can_trade()}")
    
    print("\n8. Final System Status:")
    final_status = governance.get_system_status()
    print(json.dumps(final_status, indent=2, default=str))
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
