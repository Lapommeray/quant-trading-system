"""
Audit Trail Module

This module implements quantum-safe cryptographic signatures for compliance
and audit trail logging in the Quantum Trading System.

Dependencies:
- pqcrypto (for post-quantum cryptography)
- numpy
- pandas
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import json
import hashlib
import base64
import time
import threading
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audit_trail.log')
    ]
)

logger = logging.getLogger("AuditTrail")

try:
    import pqcrypto
    from pqcrypto.sign import falcon512
    PQCRYPTO_AVAILABLE = True
    logger.info("PQCrypto loaded successfully")
except ImportError:
    logger.warning("PQCrypto not available. Install with: pip install pqcrypto")
    PQCRYPTO_AVAILABLE = False

class AuditTrail:
    """
    Quantum-safe audit trail system for the Quantum Trading System.
    Provides cryptographic signatures and tamper-proof logging.
    """
    
    def __init__(
        self,
        storage_dir: str = "audit_logs",
        signature_algorithm: str = "falcon512",
        retention_days: int = 365,
        auto_verify: bool = True
    ):
        """
        Initialize the audit trail system.
        
        Parameters:
        - storage_dir: Directory to store audit logs
        - signature_algorithm: Post-quantum signature algorithm to use
        - retention_days: Number of days to retain audit logs
        - auto_verify: Whether to automatically verify logs on read
        """
        self.storage_dir = storage_dir
        self.signature_algorithm = signature_algorithm
        self.retention_days = retention_days
        self.auto_verify = auto_verify
        
        os.makedirs(storage_dir, exist_ok=True)
        
        self.public_key = None
        self.private_key = None
        
        if PQCRYPTO_AVAILABLE:
            self._initialize_keys()
        
        self.log_cache = []
        self.max_cache_size = 1000
        
        self.flush_thread = None
        self.flush_interval = 60  # seconds
        self.flushing = False
        
        self.total_logs = 0
        self.verified_logs = 0
        self.failed_verifications = 0
        
        logger.info("AuditTrail initialized")
        
    def _initialize_keys(self) -> None:
        """Initialize cryptographic keys"""
        try:
            key_path = os.path.join(self.storage_dir, "keys")
            os.makedirs(key_path, exist_ok=True)
            
            private_key_path = os.path.join(key_path, "private_key.bin")
            public_key_path = os.path.join(key_path, "public_key.bin")
            
            if os.path.exists(private_key_path) and os.path.exists(public_key_path):
                with open(private_key_path, "rb") as f:
                    self.private_key = f.read()
                    
                with open(public_key_path, "rb") as f:
                    self.public_key = f.read()
                    
                logger.info("Loaded existing cryptographic keys")
            else:
                if self.signature_algorithm == "falcon512":
                    self.public_key, self.private_key = falcon512.generate_keypair()
                else:
                    raise ValueError(f"Unsupported signature algorithm: {self.signature_algorithm}")
                    
                with open(private_key_path, "wb") as f:
                    f.write(self.private_key)
                    
                with open(public_key_path, "wb") as f:
                    f.write(self.public_key)
                    
                logger.info("Generated new cryptographic keys")
        except Exception as e:
            logger.error(f"Error initializing keys: {str(e)}")
            
    def start_flush_thread(self) -> bool:
        """
        Start background thread for log flushing.
        
        Returns:
        - Success status
        """
        if self.flushing:
            logger.warning("Flush thread already running")
            return False
            
        try:
            self.flushing = True
            
            self.flush_thread = threading.Thread(
                target=self._flush_loop,
                daemon=True
            )
            self.flush_thread.start()
            
            logger.info("Started log flush thread")
            return True
        except Exception as e:
            logger.error(f"Error starting flush thread: {str(e)}")
            self.flushing = False
            return False
            
    def stop_flush_thread(self) -> bool:
        """
        Stop background thread for log flushing.
        
        Returns:
        - Success status
        """
        if not self.flushing:
            logger.warning("Flush thread not running")
            return False
            
        try:
            self.flushing = False
            
            if self.flush_thread and self.flush_thread.is_alive():
                self.flush_thread.join(timeout=5)
                
            self._flush_logs()
            
            logger.info("Stopped log flush thread")
            return True
        except Exception as e:
            logger.error(f"Error stopping flush thread: {str(e)}")
            return False
            
    def _flush_loop(self) -> None:
        """Background thread for log flushing"""
        while self.flushing:
            try:
                time.sleep(self.flush_interval)
                
                self._flush_logs()
            except Exception as e:
                logger.error(f"Error in flush loop: {str(e)}")
                
    def _flush_logs(self) -> None:
        """Flush logs from cache to disk"""
        if not self.log_cache:
            return
            
        try:
            logs_to_flush = self.log_cache.copy()
            
            self.log_cache = []
            
            logs_by_date = {}
            
            for log in logs_to_flush:
                log_date = log["timestamp"].split("T")[0]  # YYYY-MM-DD
                
                if log_date not in logs_by_date:
                    logs_by_date[log_date] = []
                    
                logs_by_date[log_date].append(log)
                
            for log_date, logs in logs_by_date.items():
                date_dir = os.path.join(self.storage_dir, log_date)
                os.makedirs(date_dir, exist_ok=True)
                
                file_path = os.path.join(date_dir, f"audit_{log_date}_{uuid.uuid4().hex[:8]}.json")
                
                with open(file_path, "w") as f:
                    json.dump(logs, f, indent=2)
                    
                logger.debug(f"Flushed {len(logs)} logs to {file_path}")
                
            logger.info(f"Flushed {len(logs_to_flush)} logs to disk")
        except Exception as e:
            logger.error(f"Error flushing logs: {str(e)}")
            
            self.log_cache.extend(logs_to_flush)
            
    def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "system",
        severity: str = "info",
        sign: bool = True
    ) -> Dict[str, Any]:
        """
        Log an event to the audit trail.
        
        Parameters:
        - event_type: Type of event
        - data: Event data
        - source: Source of the event
        - severity: Severity of the event
        - sign: Whether to sign the event
        
        Returns:
        - Audit log entry
        """
        try:
            timestamp = datetime.now().isoformat()
            log_id = str(uuid.uuid4())
            
            log_entry = {
                "id": log_id,
                "timestamp": timestamp,
                "event_type": event_type,
                "source": source,
                "severity": severity,
                "data": data
            }
            
            log_entry["hash"] = self._hash_log(log_entry)
            
            if sign and PQCRYPTO_AVAILABLE and self.private_key:
                log_entry["signature"] = self._sign_log(log_entry)
                
            self.log_cache.append(log_entry)
            
            if len(self.log_cache) >= self.max_cache_size:
                self._flush_logs()
                
            self.total_logs += 1
            
            return log_entry
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
            
            error_log = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "event_type": "audit_error",
                "source": "audit_trail",
                "severity": "error",
                "data": {
                    "error": str(e),
                    "original_event_type": event_type,
                    "original_source": source
                }
            }
            
            self.log_cache.append(error_log)
            
            return error_log
            
    def _hash_log(self, log_entry: Dict[str, Any]) -> str:
        """
        Calculate hash of log entry.
        
        Parameters:
        - log_entry: Log entry to hash
        
        Returns:
        - Hash of log entry
        """
        log_copy = log_entry.copy()
        log_copy.pop("hash", None)
        log_copy.pop("signature", None)
        
        log_json = json.dumps(log_copy, sort_keys=True)
        
        log_hash = hashlib.sha256(log_json.encode()).hexdigest()
        
        return log_hash
        
    def _sign_log(self, log_entry: Dict[str, Any]) -> str:
        """
        Sign log entry.
        
        Parameters:
        - log_entry: Log entry to sign
        
        Returns:
        - Base64-encoded signature
        """
        if not PQCRYPTO_AVAILABLE or not self.private_key:
            return None
            
        try:
            log_hash = log_entry["hash"]
            
            if self.signature_algorithm == "falcon512":
                signature = falcon512.sign(self.private_key, log_hash.encode())
            else:
                raise ValueError(f"Unsupported signature algorithm: {self.signature_algorithm}")
                
            signature_b64 = base64.b64encode(signature).decode()
            
            return signature_b64
        except Exception as e:
            logger.error(f"Error signing log: {str(e)}")
            return None
            
    def verify_log(self, log_entry: Dict[str, Any]) -> bool:
        """
        Verify log entry.
        
        Parameters:
        - log_entry: Log entry to verify
        
        Returns:
        - Verification status
        """
        if not PQCRYPTO_AVAILABLE or not self.public_key:
            logger.warning("PQCrypto not available, cannot verify log")
            return False
            
        try:
            if "hash" not in log_entry or "signature" not in log_entry:
                logger.warning("Log entry missing hash or signature")
                return False
                
            calculated_hash = self._hash_log(log_entry)
            
            if calculated_hash != log_entry["hash"]:
                logger.warning("Log hash verification failed")
                self.failed_verifications += 1
                return False
                
            signature = base64.b64decode(log_entry["signature"])
            
            if self.signature_algorithm == "falcon512":
                try:
                    falcon512.verify(self.public_key, log_entry["hash"].encode(), signature)
                    self.verified_logs += 1
                    return True
                except Exception:
                    logger.warning("Log signature verification failed")
                    self.failed_verifications += 1
                    return False
            else:
                raise ValueError(f"Unsupported signature algorithm: {self.signature_algorithm}")
        except Exception as e:
            logger.error(f"Error verifying log: {str(e)}")
            self.failed_verifications += 1
            return False
            
    def get_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        verify: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get logs from the audit trail.
        
        Parameters:
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - event_types: List of event types to include
        - sources: List of sources to include
        - severities: List of severities to include
        - verify: Whether to verify logs (defaults to self.auto_verify)
        
        Returns:
        - List of log entries
        """
        try:
            if start_date is None:
                start_date = (datetime.now().date() - pd.Timedelta(days=7)).isoformat()
                
            if end_date is None:
                end_date = datetime.now().date().isoformat()
                
            start_date_obj = pd.Timestamp(start_date).date()
            end_date_obj = pd.Timestamp(end_date).date()
            
            date_range = pd.date_range(start=start_date_obj, end=end_date_obj)
            
            logs = []
            
            for date in date_range:
                date_str = date.strftime("%Y-%m-%d")
                date_dir = os.path.join(self.storage_dir, date_str)
                
                if not os.path.exists(date_dir):
                    continue
                    
                log_files = [f for f in os.listdir(date_dir) if f.startswith("audit_") and f.endswith(".json")]
                
                for log_file in log_files:
                    file_path = os.path.join(date_dir, log_file)
                    
                    with open(file_path, "r") as f:
                        file_logs = json.load(f)
                        
                    logs.extend(file_logs)
                    
            for log in self.log_cache:
                log_date = log["timestamp"].split("T")[0]
                
                if start_date <= log_date <= end_date:
                    logs.append(log)
                    
            filtered_logs = []
            
            for log in logs:
                if event_types and log["event_type"] not in event_types:
                    continue
                    
                if sources and log["source"] not in sources:
                    continue
                    
                if severities and log["severity"] not in severities:
                    continue
                    
                if verify is None:
                    verify = self.auto_verify
                    
                if verify and "signature" in log:
                    if not self.verify_log(log):
                        logger.warning(f"Log verification failed: {log['id']}")
                        continue
                        
                filtered_logs.append(log)
                
            return filtered_logs
        except Exception as e:
            logger.error(f"Error getting logs: {str(e)}")
            return []
            
    def get_log_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of logs.
        
        Parameters:
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        
        Returns:
        - Summary of logs
        """
        try:
            logs = self.get_logs(start_date=start_date, end_date=end_date, verify=False)
            
            summary = {
                "total_logs": len(logs),
                "event_types": {},
                "sources": {},
                "severities": {},
                "start_date": start_date,
                "end_date": end_date,
                "generated_at": datetime.now().isoformat()
            }
            
            for log in logs:
                event_type = log["event_type"]
                source = log["source"]
                severity = log["severity"]
                
                if event_type not in summary["event_types"]:
                    summary["event_types"][event_type] = 0
                    
                summary["event_types"][event_type] += 1
                
                if source not in summary["sources"]:
                    summary["sources"][source] = 0
                    
                summary["sources"][source] += 1
                
                if severity not in summary["severities"]:
                    summary["severities"][severity] = 0
                    
                summary["severities"][severity] += 1
                
            return summary
        except Exception as e:
            logger.error(f"Error getting log summary: {str(e)}")
            return {
                "error": str(e),
                "total_logs": 0,
                "generated_at": datetime.now().isoformat()
            }
            
    def export_logs(
        self,
        output_file: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        format: str = "json"
    ) -> bool:
        """
        Export logs to a file.
        
        Parameters:
        - output_file: Output file path
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - event_types: List of event types to include
        - sources: List of sources to include
        - severities: List of severities to include
        - format: Output format ('json' or 'csv')
        
        Returns:
        - Success status
        """
        try:
            logs = self.get_logs(
                start_date=start_date,
                end_date=end_date,
                event_types=event_types,
                sources=sources,
                severities=severities,
                verify=False
            )
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format.lower() == "json":
                with open(output_file, "w") as f:
                    json.dump(logs, f, indent=2)
            elif format.lower() == "csv":
                flattened_logs = []
                
                for log in logs:
                    flat_log = log.copy()
                    
                    flat_log["data"] = json.dumps(flat_log["data"])
                    
                    flattened_logs.append(flat_log)
                    
                df = pd.DataFrame(flattened_logs)
                
                df.to_csv(output_file, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported {len(logs)} logs to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting logs: {str(e)}")
            return False
            
    def cleanup_old_logs(self) -> int:
        """
        Clean up old logs.
        
        Returns:
        - Number of logs deleted
        """
        try:
            cutoff_date = (datetime.now().date() - pd.Timedelta(days=self.retention_days)).isoformat()
            
            date_dirs = [d for d in os.listdir(self.storage_dir) if os.path.isdir(os.path.join(self.storage_dir, d))]
            
            old_dirs = [d for d in date_dirs if d < cutoff_date and d != "keys"]
            
            deleted_count = 0
            
            for old_dir in old_dirs:
                dir_path = os.path.join(self.storage_dir, old_dir)
                
                log_files = [f for f in os.listdir(dir_path) if f.startswith("audit_") and f.endswith(".json")]
                deleted_count += len(log_files)
                
                for file in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file))
                    
                os.rmdir(dir_path)
                
            logger.info(f"Cleaned up {deleted_count} old logs")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {str(e)}")
            return 0
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get audit trail metrics.
        
        Returns:
        - Dictionary with metrics
        """
        return {
            "total_logs": self.total_logs,
            "verified_logs": self.verified_logs,
            "failed_verifications": self.failed_verifications,
            "cache_size": len(self.log_cache),
            "retention_days": self.retention_days,
            "pqcrypto_available": PQCRYPTO_AVAILABLE,
            "signature_algorithm": self.signature_algorithm,
            "last_updated": datetime.now().isoformat()
        }

class ComplianceLogger:
    """
    Compliance logger for the Quantum Trading System.
    Provides specialized logging for compliance events.
    """
    
    def __init__(
        self,
        audit_trail: Optional[AuditTrail] = None,
        compliance_dir: str = "compliance_logs"
    ):
        """
        Initialize the compliance logger.
        
        Parameters:
        - audit_trail: AuditTrail instance
        - compliance_dir: Directory to store compliance logs
        """
        self.audit_trail = audit_trail or AuditTrail(storage_dir="audit_logs")
        self.compliance_dir = compliance_dir
        
        os.makedirs(compliance_dir, exist_ok=True)
        
        logger.info("ComplianceLogger initialized")
        
    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        timestamp: Optional[str] = None,
        exchange: str = "unknown",
        account_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a trade event.
        
        Parameters:
        - trade_id: Trade ID
        - symbol: Trading symbol
        - direction: Trade direction ('buy' or 'sell')
        - quantity: Trade quantity
        - price: Trade price
        - timestamp: Trade timestamp (defaults to current time)
        - exchange: Exchange name
        - account_id: Account ID
        - metadata: Additional metadata
        
        Returns:
        - Audit log entry
        """
        trade_data = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp or datetime.now().isoformat(),
            "exchange": exchange,
            "account_id": account_id,
            "metadata": metadata or {}
        }
        
        log_entry = self.audit_trail.log_event(
            event_type="trade",
            data=trade_data,
            source="trading_system",
            severity="info"
        )
        
        return log_entry
        
    def log_order(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        status: str = "new",
        timestamp: Optional[str] = None,
        exchange: str = "unknown",
        account_id: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an order event.
        
        Parameters:
        - order_id: Order ID
        - symbol: Trading symbol
        - direction: Order direction ('buy' or 'sell')
        - quantity: Order quantity
        - price: Order price (for limit orders)
        - order_type: Order type ('market', 'limit', etc.)
        - status: Order status ('new', 'filled', 'cancelled', etc.)
        - timestamp: Order timestamp (defaults to current time)
        - exchange: Exchange name
        - account_id: Account ID
        - metadata: Additional metadata
        
        Returns:
        - Audit log entry
        """
        order_data = {
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "status": status,
            "timestamp": timestamp or datetime.now().isoformat(),
            "exchange": exchange,
            "account_id": account_id,
            "metadata": metadata or {}
        }
        
        log_entry = self.audit_trail.log_event(
            event_type="order",
            data=order_data,
            source="trading_system",
            severity="info"
        )
        
        return log_entry
        
    def log_strategy_decision(
        self,
        strategy_id: str,
        decision: str,
        symbols: List[str],
        confidence: float,
        timestamp: Optional[str] = None,
        signals: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a strategy decision event.
        
        Parameters:
        - strategy_id: Strategy ID
        - decision: Decision ('buy', 'sell', 'hold', etc.)
        - symbols: List of symbols
        - confidence: Decision confidence
        - timestamp: Decision timestamp (defaults to current time)
        - signals: Signal data
        - metadata: Additional metadata
        
        Returns:
        - Audit log entry
        """
        decision_data = {
            "strategy_id": strategy_id,
            "decision": decision,
            "symbols": symbols,
            "confidence": confidence,
            "timestamp": timestamp or datetime.now().isoformat(),
            "signals": signals or {},
            "metadata": metadata or {}
        }
        
        log_entry = self.audit_trail.log_event(
            event_type="strategy_decision",
            data=decision_data,
            source="strategy_engine",
            severity="info"
        )
        
        return log_entry
        
    def log_risk_event(
        self,
        event_type: str,
        risk_level: str,
        description: str,
        affected_symbols: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a risk event.
        
        Parameters:
        - event_type: Risk event type
        - risk_level: Risk level ('low', 'medium', 'high', 'critical')
        - description: Event description
        - affected_symbols: List of affected symbols
        - timestamp: Event timestamp (defaults to current time)
        - metadata: Additional metadata
        
        Returns:
        - Audit log entry
        """
        risk_data = {
            "event_type": event_type,
            "risk_level": risk_level,
            "description": description,
            "affected_symbols": affected_symbols or [],
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if risk_level == "critical":
            severity = "critical"
        elif risk_level == "high":
            severity = "error"
        elif risk_level == "medium":
            severity = "warning"
        else:
            severity = "info"
            
        log_entry = self.audit_trail.log_event(
            event_type="risk_event",
            data=risk_data,
            source="risk_system",
            severity=severity
        )
        
        return log_entry
        
    def log_compliance_event(
        self,
        event_type: str,
        description: str,
        compliance_level: str = "info",
        affected_accounts: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a compliance event.
        
        Parameters:
        - event_type: Compliance event type
        - description: Event description
        - compliance_level: Compliance level ('info', 'warning', 'violation')
        - affected_accounts: List of affected accounts
        - timestamp: Event timestamp (defaults to current time)
        - metadata: Additional metadata
        
        Returns:
        - Audit log entry
        """
        compliance_data = {
            "event_type": event_type,
            "description": description,
            "compliance_level": compliance_level,
            "affected_accounts": affected_accounts or [],
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if compliance_level == "violation":
            severity = "critical"
        elif compliance_level == "warning":
            severity = "warning"
        else:
            severity = "info"
            
        log_entry = self.audit_trail.log_event(
            event_type="compliance_event",
            data=compliance_data,
            source="compliance_system",
            severity=severity
        )
        
        try:
            log_date = datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.compliance_dir, log_date)
            os.makedirs(date_dir, exist_ok=True)
            
            file_path = os.path.join(date_dir, f"compliance_{log_date}.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
                
            logs.append(log_entry)
            
            with open(file_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to compliance log file: {str(e)}")
            
        return log_entry
        
    def get_compliance_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compliance_levels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get compliance logs.
        
        Parameters:
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - compliance_levels: List of compliance levels to include
        
        Returns:
        - List of compliance logs
        """
        return self.audit_trail.get_logs(
            start_date=start_date,
            end_date=end_date,
            event_types=["compliance_event"],
            sources=["compliance_system"],
            verify=True
        )
        
    def export_compliance_report(
        self,
        output_file: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = "json"
    ) -> bool:
        """
        Export compliance report.
        
        Parameters:
        - output_file: Output file path
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        - format: Output format ('json' or 'csv')
        
        Returns:
        - Success status
        """
        return self.audit_trail.export_logs(
            output_file=output_file,
            start_date=start_date,
            end_date=end_date,
            event_types=["compliance_event"],
            sources=["compliance_system"],
            format=format
        )

if __name__ == "__main__":
    audit_trail = AuditTrail(storage_dir="audit_logs")
    
    audit_trail.start_flush_thread()
    
    try:
        audit_trail.log_event(
            event_type="system_start",
            data={"version": "1.0.0"},
            source="system",
            severity="info"
        )
        
        compliance_logger = ComplianceLogger(audit_trail=audit_trail)
        
        compliance_logger.log_trade(
            trade_id="T12345",
            symbol="BTCUSD",
            direction="buy",
            quantity=1.0,
            price=50000.0,
            exchange="binance"
        )
        
        compliance_logger.log_order(
            order_id="O12345",
            symbol="BTCUSD",
            direction="buy",
            quantity=1.0,
            price=50000.0,
            order_type="limit",
            status="new",
            exchange="binance"
        )
        
        compliance_logger.log_strategy_decision(
            strategy_id="S12345",
            decision="buy",
            symbols=["BTCUSD"],
            confidence=0.8,
            signals={"rsi": 30, "macd": "bullish"}
        )
        
        compliance_logger.log_risk_event(
            event_type="yield_curve_inversion",
            risk_level="high",
            description="2s10s yield curve inverted for 5 days",
            affected_symbols=["SPY", "QQQ"]
        )
        
        compliance_logger.log_compliance_event(
            event_type="position_limit",
            description="Position limit exceeded for BTCUSD",
            compliance_level="warning",
            affected_accounts=["A12345"]
        )
        
        logs = audit_trail.get_logs()
        print(f"Total logs: {len(logs)}")
        
        compliance_logs = compliance_logger.get_compliance_logs()
        print(f"Compliance logs: {len(compliance_logs)}")
        
        compliance_logger.export_compliance_report("compliance_report.json")
        
        metrics = audit_trail.get_metrics()
        print(f"Metrics: {metrics}")
    finally:
        audit_trail.stop_flush_thread()
