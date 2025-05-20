"""
priority_engine.py

Priority Engine for OversoulDirector

Manages the priority matrix for module activation and signal routing.
"""

import numpy as np
from datetime import datetime
import json
import os

class PriorityEngine:
    """
    Priority Engine for OversoulDirector
    
    Manages the priority matrix for module activation and signal routing.
    """
    
    def __init__(self, oversoul_director=None):
        """
        Initialize the Priority Engine
        
        Parameters:
        - oversoul_director: OversoulDirector instance (optional)
        """
        self.oversoul_director = oversoul_director
        self.priority_matrix = self._load_default_priority_matrix()
        self.priority_history = []
        self.last_update_time = None
    
    def _load_default_priority_matrix(self):
        """
        Load the default priority matrix
        
        Returns:
        - Dictionary with module priorities
        """
        return {
            "phoenix": 1.0,
            "aurora": 0.9,
            "qmp": 0.8,
            "truth": 1.0,
            "ritual": 0.7,
            "darwin": 0.8,
            "consciousness": 0.6,
            "event_probability": 0.7
        }
    
    def load_priority_matrix(self, file_path=None):
        """
        Load a priority matrix from a file
        
        Parameters:
        - file_path: Path to priority matrix file (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "priority_matrix.json"
            )
        
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, "r") as f:
                self.priority_matrix = json.load(f)
            
            self.last_update_time = datetime.now()
            return True
        except Exception as e:
            print(f"Error loading priority matrix: {e}")
            return False
    
    def save_priority_matrix(self, file_path=None):
        """
        Save the priority matrix to a file
        
        Parameters:
        - file_path: Path to priority matrix file (optional)
        
        Returns:
        - True if successful, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "priority_matrix.json"
            )
        
        try:
            with open(file_path, "w") as f:
                json.dump(self.priority_matrix, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving priority matrix: {e}")
            return False
    
    def update_priority(self, module_name, priority):
        """
        Update the priority of a module
        
        Parameters:
        - module_name: Module name
        - priority: New priority (0.0 to 1.0)
        
        Returns:
        - True if successful, False otherwise
        """
        if module_name not in self.priority_matrix:
            return False
        
        priority = max(0.0, min(1.0, priority))
        
        old_priority = self.priority_matrix[module_name]
        
        self.priority_matrix[module_name] = priority
        
        update_record = {
            "timestamp": datetime.now(),
            "module": module_name,
            "old_priority": old_priority,
            "new_priority": priority
        }
        
        self.priority_history.append(update_record)
        self.last_update_time = update_record["timestamp"]
        
        if len(self.priority_history) > 100:
            self.priority_history = self.priority_history[-100:]
        
        return True
    
    def get_priority(self, module_name):
        """
        Get the priority of a module
        
        Parameters:
        - module_name: Module name
        
        Returns:
        - Module priority (0.0 to 1.0)
        """
        return self.priority_matrix.get(module_name, 0.0)
    
    def get_priority_matrix(self):
        """
        Get the priority matrix
        
        Returns:
        - Dictionary with module priorities
        """
        return self.priority_matrix
    
    def get_priority_history(self):
        """
        Get the priority update history
        
        Returns:
        - List of priority update records
        """
        return self.priority_history
    
    def adapt_priorities(self, market_state=None, performance_metrics=None):
        """
        Adapt module priorities based on market state and performance metrics
        
        Parameters:
        - market_state: Dictionary with market state information (optional)
        - performance_metrics: Dictionary with performance metrics (optional)
        
        Returns:
        - Dictionary with updated module priorities
        """
        if not market_state and not performance_metrics:
            return self.priority_matrix
        
        updated_priorities = self.priority_matrix.copy()
        
        if market_state:
            if "volatility" in market_state and market_state["volatility"] > 25:
                updated_priorities["phoenix"] = min(1.0, updated_priorities["phoenix"] * 1.2)
            
            if "unusual_conditions" in market_state and market_state["unusual_conditions"]:
                updated_priorities["aurora"] = min(1.0, updated_priorities["aurora"] * 1.2)
            
            if "time_sensitive" in market_state and market_state["time_sensitive"]:
                updated_priorities["ritual"] = min(1.0, updated_priorities["ritual"] * 1.2)
            
            if "event_risk" in market_state and market_state["event_risk"] > 0.5:
                updated_priorities["event_probability"] = min(1.0, updated_priorities["event_probability"] * 1.2)
        
        if performance_metrics:
            if "module_performance" in performance_metrics:
                for module, performance in performance_metrics["module_performance"].items():
                    if module in updated_priorities:
                        if performance > 0.7:
                            updated_priorities[module] = min(1.0, updated_priorities[module] * 1.1)
                        elif performance < 0.3:
                            updated_priorities[module] = max(0.1, updated_priorities[module] * 0.9)
        
        update_record = {
            "timestamp": datetime.now(),
            "old_priorities": self.priority_matrix.copy(),
            "new_priorities": updated_priorities,
            "market_state": market_state,
            "performance_metrics": performance_metrics
        }
        
        self.priority_history.append(update_record)
        self.last_update_time = update_record["timestamp"]
        
        self.priority_matrix = updated_priorities
        
        if len(self.priority_history) > 100:
            self.priority_history = self.priority_history[-100:]
        
        return updated_priorities
