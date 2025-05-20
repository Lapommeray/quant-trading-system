"""
audit_logger.py

Audit Logger for QMP Overrider

Provides comprehensive logging and auditing capabilities for the QMP Overrider system.
"""

import os
import json
import logging
import csv
from datetime import datetime
import traceback

class AuditLogger:
    """
    Audit Logger for QMP Overrider
    
    Provides comprehensive logging and auditing capabilities for the QMP Overrider system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Audit Logger
        
        Parameters:
        - config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        self.config = self._load_config()
        self.loggers = {}
        self.csv_writers = {}
        self.initialized = False
    
    def _load_config(self):
        """
        Load configuration from file
        
        Returns:
        - Dictionary with configuration
        """
        config = {
            "log_level": "INFO",
            "log_file": "qmp_overrider.log",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_date_format": "%Y-%m-%d %H:%M:%S",
            "log_to_console": True,
            "log_to_file": True,
            "log_to_csv": True,
            "log_dir": "logs",
            "csv_dir": "logs/csv",
            "max_log_size": 10485760,  # 10 MB
            "max_log_backups": 5,
            "log_types": {
                "system": True,
                "signal": True,
                "trade": True,
                "performance": True,
                "error": True,
                "audit": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    
                    if "logging" in loaded_config:
                        for key, value in loaded_config["logging"].items():
                            config[key] = value
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return config
    
    def initialize(self):
        """
        Initialize the Audit Logger
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        self._create_log_directories()
        
        self._initialize_loggers()
        
        self._initialize_csv_writers()
        
        self.initialized = True
        
        self.log_system("Audit Logger initialized")
        
        return True
    
    def _create_log_directories(self):
        """Create log directories"""
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.config["log_dir"]
        )
        os.makedirs(log_dir, exist_ok=True)
        
        csv_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.config["csv_dir"]
        )
        os.makedirs(csv_dir, exist_ok=True)
    
    def _initialize_loggers(self):
        """Initialize loggers"""
        log_level = getattr(logging, self.config["log_level"].upper(), logging.INFO)
        
        formatter = logging.Formatter(
            self.config["log_format"],
            self.config["log_date_format"]
        )
        
        for log_type in self.config["log_types"]:
            if not self.config["log_types"][log_type]:
                continue
            
            logger = logging.getLogger(f"qmp_overrider.{log_type}")
            logger.setLevel(log_level)
            
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            if self.config["log_to_console"]:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            if self.config["log_to_file"]:
                log_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    self.config["log_dir"],
                    f"{log_type}.log"
                )
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.config["max_log_size"],
                    backupCount=self.config["max_log_backups"]
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            self.loggers[log_type] = logger
    
    def _initialize_csv_writers(self):
        """Initialize CSV writers"""
        if not self.config["log_to_csv"]:
            return
        
        for log_type in self.config["log_types"]:
            if not self.config["log_types"][log_type]:
                continue
            
            csv_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.config["csv_dir"],
                f"{log_type}.csv"
            )
            
            try:
                file_exists = os.path.exists(csv_file)
                
                f = open(csv_file, "a", newline="")
                
                writer = csv.writer(f)
                
                if not file_exists:
                    if log_type == "system":
                        writer.writerow(["timestamp", "level", "message", "details"])
                    elif log_type == "signal":
                        writer.writerow(["timestamp", "symbol", "direction", "confidence", "modules", "details"])
                    elif log_type == "trade":
                        writer.writerow(["timestamp", "symbol", "direction", "price", "quantity", "position_size", "details"])
                    elif log_type == "performance":
                        writer.writerow(["timestamp", "metric", "value", "details"])
                    elif log_type == "error":
                        writer.writerow(["timestamp", "level", "message", "traceback", "details"])
                    elif log_type == "audit":
                        writer.writerow(["timestamp", "user", "action", "resource", "details"])
                
                self.csv_writers[log_type] = {
                    "writer": writer,
                    "file": f
                }
            except Exception as e:
                print(f"Error initializing CSV writer for {log_type}: {e}")
    
    def log_system(self, message, level="INFO", details=None):
        """
        Log system message
        
        Parameters:
        - message: Message to log
        - level: Log level (optional)
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("system", False):
            return
        
        logger = self.loggers.get("system")
        
        if not logger:
            return
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        logger.log(log_level, message)
        
        if self.config["log_to_csv"] and "system" in self.csv_writers:
            try:
                writer = self.csv_writers["system"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    level,
                    message,
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["system"]["file"].flush()
            except Exception as e:
                print(f"Error writing to system CSV: {e}")
    
    def log_signal(self, symbol, direction, confidence, modules=None, details=None):
        """
        Log signal
        
        Parameters:
        - symbol: Symbol
        - direction: Signal direction
        - confidence: Signal confidence
        - modules: Active modules (optional)
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("signal", False):
            return
        
        logger = self.loggers.get("signal")
        
        if not logger:
            return
        
        message = f"{symbol} {direction} | Confidence: {confidence:.2f}"
        
        if modules:
            message += f" | Modules: {', '.join(modules)}"
        
        logger.info(message)
        
        if self.config["log_to_csv"] and "signal" in self.csv_writers:
            try:
                writer = self.csv_writers["signal"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol,
                    direction,
                    confidence,
                    json.dumps(modules) if modules else "",
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["signal"]["file"].flush()
            except Exception as e:
                print(f"Error writing to signal CSV: {e}")
    
    def log_trade(self, symbol, direction, price, quantity, position_size, details=None):
        """
        Log trade
        
        Parameters:
        - symbol: Symbol
        - direction: Trade direction
        - price: Trade price
        - quantity: Trade quantity
        - position_size: Position size
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("trade", False):
            return
        
        logger = self.loggers.get("trade")
        
        if not logger:
            return
        
        message = f"{symbol} {direction} | Price: {price:.2f} | Quantity: {quantity:.2f} | Position Size: {position_size:.2f}"
        
        logger.info(message)
        
        if self.config["log_to_csv"] and "trade" in self.csv_writers:
            try:
                writer = self.csv_writers["trade"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    symbol,
                    direction,
                    price,
                    quantity,
                    position_size,
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["trade"]["file"].flush()
            except Exception as e:
                print(f"Error writing to trade CSV: {e}")
    
    def log_performance(self, metric, value, details=None):
        """
        Log performance metric
        
        Parameters:
        - metric: Performance metric
        - value: Metric value
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("performance", False):
            return
        
        logger = self.loggers.get("performance")
        
        if not logger:
            return
        
        message = f"{metric}: {value}"
        
        logger.info(message)
        
        if self.config["log_to_csv"] and "performance" in self.csv_writers:
            try:
                writer = self.csv_writers["performance"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    metric,
                    value,
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["performance"]["file"].flush()
            except Exception as e:
                print(f"Error writing to performance CSV: {e}")
    
    def log_error(self, message, level="ERROR", exc_info=None, details=None):
        """
        Log error
        
        Parameters:
        - message: Error message
        - level: Log level (optional)
        - exc_info: Exception info (optional)
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("error", False):
            return
        
        logger = self.loggers.get("error")
        
        if not logger:
            return
        
        log_level = getattr(logging, level.upper(), logging.ERROR)
        
        logger.log(log_level, message, exc_info=exc_info)
        
        tb = ""
        
        if exc_info:
            tb = "".join(traceback.format_exception(*exc_info))
        
        if self.config["log_to_csv"] and "error" in self.csv_writers:
            try:
                writer = self.csv_writers["error"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    level,
                    message,
                    tb,
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["error"]["file"].flush()
            except Exception as e:
                print(f"Error writing to error CSV: {e}")
    
    def log_audit(self, user, action, resource, details=None):
        """
        Log audit event
        
        Parameters:
        - user: User
        - action: Action
        - resource: Resource
        - details: Additional details (optional)
        """
        if not self.initialized:
            self.initialize()
        
        if not self.config["log_types"].get("audit", False):
            return
        
        logger = self.loggers.get("audit")
        
        if not logger:
            return
        
        message = f"{user} {action} {resource}"
        
        logger.info(message)
        
        if self.config["log_to_csv"] and "audit" in self.csv_writers:
            try:
                writer = self.csv_writers["audit"]["writer"]
                
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user,
                    action,
                    resource,
                    json.dumps(details) if details else ""
                ])
                
                self.csv_writers["audit"]["file"].flush()
            except Exception as e:
                print(f"Error writing to audit CSV: {e}")
    
    def close(self):
        """Close loggers and CSV writers"""
        for log_type, writer_info in self.csv_writers.items():
            try:
                writer_info["file"].close()
            except Exception as e:
                print(f"Error closing CSV writer for {log_type}: {e}")
        
        self.csv_writers = {}
        
        self.loggers = {}
        
        self.initialized = False
    
    def get_status(self):
        """
        Get Audit Logger status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "log_types": list(self.loggers.keys()),
            "csv_writers": list(self.csv_writers.keys())
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QMP Overrider Audit Logger")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log", type=str, help="Log message")
    parser.add_argument("--level", type=str, default="INFO", help="Log level")
    parser.add_argument("--type", type=str, default="system", help="Log type")
    
    args = parser.parse_args()
    
    audit_logger = AuditLogger(args.config)
    
    audit_logger.initialize()
    
    if args.log:
        if args.type == "system":
            audit_logger.log_system(args.log, args.level)
        elif args.type == "error":
            audit_logger.log_error(args.log, args.level)
        else:
            print(f"Unsupported log type: {args.type}")
    
    audit_logger.close()

if __name__ == "__main__":
    main()
