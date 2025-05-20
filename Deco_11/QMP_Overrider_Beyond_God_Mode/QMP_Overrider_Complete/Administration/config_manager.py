"""
config_manager.py

Configuration Manager for QMP Overrider

Manages configuration settings for the QMP Overrider system.
"""

import os
import json
import argparse
from datetime import datetime

class ConfigManager:
    """
    Configuration Manager for QMP Overrider
    
    Manages configuration settings for the QMP Overrider system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Configuration Manager
        
        Parameters:
        - config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        self.config = self._load_config()
        self.last_update_time = None
        self.update_history = []
        self.max_history = 100
    
    def _load_config(self):
        """
        Load configuration from file
        
        Returns:
        - Dictionary with configuration
        """
        config = {
            "version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbols": ["BTCUSD", "ETHUSD", "XAUUSD", "DIA", "QQQ"],
            "markets": {
                "BTCUSD": "crypto",
                "ETHUSD": "crypto",
                "XAUUSD": "forex",
                "DIA": "equity",
                "QQQ": "equity"
            },
            "timeframes": ["1m", "5m", "10m", "15m", "20m", "25m"],
            "modules": {
                "phoenix": True,
                "aurora": True,
                "truth": True,
                "ritual": True,
                "darwin": True,
                "consciousness": True,
                "event_probability": True
            },
            "thresholds": {
                "alien": 0.85,
                "cosmic": 0.9,
                "quantum": 0.92,
                "emotion": 0.7,
                "angelic": 0.5,
                "divine_timing": 0.5,
                "sacred_date": 0.0,
                "big_move": 0.5,
                "timeline": 0.5,
                "macro": 0.5
            },
            "trading": {
                "position_size": 0.1,
                "max_positions": 5,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "max_drawdown": 0.2,
                "max_leverage": 1.0
            },
            "logging": {
                "log_signals": True,
                "log_trades": True,
                "log_level": "INFO",
                "log_file": "qmp_overrider.log"
            },
            "paths": {
                "data": "data",
                "logs": "logs",
                "results": "results",
                "models": "models"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    
                    for key, value in loaded_config.items():
                        config[key] = value
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return config
    
    def save_config(self):
        """
        Save configuration to file
        
        Returns:
        - True if successful, False otherwise
        """
        self.config["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            
            self.last_update_time = datetime.now()
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_config(self):
        """
        Get configuration
        
        Returns:
        - Dictionary with configuration
        """
        return self.config
    
    def update_config(self, updates):
        """
        Update configuration
        
        Parameters:
        - updates: Dictionary with configuration updates
        
        Returns:
        - True if successful, False otherwise
        """
        update_record = {
            "timestamp": datetime.now(),
            "updates": updates
        }
        
        for key, value in updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey in self.config[key]:
                            update_record[f"{key}.{subkey}"] = {
                                "old": self.config[key][subkey],
                                "new": subvalue
                            }
                            self.config[key][subkey] = subvalue
                else:
                    update_record[key] = {
                        "old": self.config[key],
                        "new": value
                    }
                    self.config[key] = value
        
        self.update_history.append(update_record)
        
        if len(self.update_history) > self.max_history:
            self.update_history = self.update_history[-self.max_history:]
        
        return self.save_config()
    
    def reset_config(self):
        """
        Reset configuration to defaults
        
        Returns:
        - True if successful, False otherwise
        """
        self.config = self._load_config()
        
        return self.save_config()
    
    def get_update_history(self, max_count=None):
        """
        Get update history
        
        Parameters:
        - max_count: Maximum number of records to return (optional)
        
        Returns:
        - List of update records
        """
        if max_count:
            return self.update_history[-max_count:]
        
        return self.update_history
    
    def setup(self):
        """
        Setup configuration
        
        Returns:
        - True if successful, False otherwise
        """
        for path_name, path in self.config["paths"].items():
            os.makedirs(path, exist_ok=True)
        
        return self.save_config()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="QMP Overrider Configuration Manager")
    parser.add_argument("--setup", action="store_true", help="Setup configuration")
    parser.add_argument("--reset", action="store_true", help="Reset configuration to defaults")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--update", type=str, help="Update configuration (JSON string)")
    parser.add_argument("--get", type=str, help="Get configuration value")
    
    args = parser.parse_args()
    
    config_manager = ConfigManager(args.config)
    
    if args.setup:
        if config_manager.setup():
            print("Configuration setup successful")
        else:
            print("Configuration setup failed")
    
    elif args.reset:
        if config_manager.reset_config():
            print("Configuration reset successful")
        else:
            print("Configuration reset failed")
    
    elif args.update:
        try:
            updates = json.loads(args.update)
            
            if config_manager.update_config(updates):
                print("Configuration update successful")
            else:
                print("Configuration update failed")
        except json.JSONDecodeError:
            print("Invalid JSON string")
    
    elif args.get:
        config = config_manager.get_config()
        
        if args.get in config:
            print(json.dumps(config[args.get], indent=4))
        else:
            print(f"Configuration key '{args.get}' not found")
    
    else:
        config = config_manager.get_config()
        print(json.dumps(config, indent=4))

if __name__ == "__main__":
    main()
