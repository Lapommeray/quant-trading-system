"""
module_activator.py

Module Activator for OversoulDirector

Activates and deactivates modules based on market conditions and priority matrix.
"""

import numpy as np
from datetime import datetime

class ModuleActivator:
    """
    Module Activator for OversoulDirector
    
    Activates and deactivates modules based on market conditions and priority matrix.
    """
    
    def __init__(self, oversoul_director=None):
        """
        Initialize the Module Activator
        
        Parameters:
        - oversoul_director: OversoulDirector instance (optional)
        """
        self.oversoul_director = oversoul_director
        self.active_modules = {}
        self.activation_history = []
        self.last_activation_time = None
        
    def determine_active_modules(self, market_state):
        """
        Determine which modules should be active based on market state
        
        Parameters:
        - market_state: Dictionary with market state information
        
        Returns:
        - Dictionary with active modules and their activation levels
        """
        now = datetime.now()
        
        active_modules = {}
        
        core_modules = ["phoenix", "aurora", "qmp", "truth"]
        for module in core_modules:
            active_modules[module] = 1.0
        
        if market_state:
            if "time_sensitive" in market_state and market_state["time_sensitive"]:
                active_modules["ritual"] = 1.0
            else:
                active_modules["ritual"] = 0.5
            
            if "volatility" in market_state:
                volatility = market_state["volatility"]
                if volatility > 30:
                    active_modules["darwin"] = 1.0
                elif volatility > 20:
                    active_modules["darwin"] = 0.7
                elif volatility > 10:
                    active_modules["darwin"] = 0.5
                else:
                    active_modules["darwin"] = 0.3
            else:
                active_modules["darwin"] = 0.5
            
            if "complexity" in market_state:
                complexity = market_state["complexity"]
                if complexity > 0.8:
                    active_modules["consciousness"] = 1.0
                elif complexity > 0.5:
                    active_modules["consciousness"] = 0.7
                else:
                    active_modules["consciousness"] = 0.5
            else:
                active_modules["consciousness"] = 0.5
            
            if "event_risk" in market_state:
                event_risk = market_state["event_risk"]
                if event_risk > 0.7:
                    active_modules["event_probability"] = 1.0
                elif event_risk > 0.4:
                    active_modules["event_probability"] = 0.7
                else:
                    active_modules["event_probability"] = 0.3
            else:
                active_modules["event_probability"] = 0.5
        else:
            active_modules["ritual"] = 0.5
            active_modules["darwin"] = 0.5
            active_modules["consciousness"] = 0.5
            active_modules["event_probability"] = 0.5
        
        if self.oversoul_director and hasattr(self.oversoul_director, "priority_matrix"):
            priority_matrix = self.oversoul_director.priority_matrix
            for module, activation in active_modules.items():
                if module in priority_matrix:
                    active_modules[module] = activation * priority_matrix[module]
        
        activation_record = {
            "timestamp": now,
            "market_state": market_state,
            "active_modules": active_modules
        }
        
        self.activation_history.append(activation_record)
        self.last_activation_time = now
        self.active_modules = active_modules
        
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]
        
        return active_modules
    
    def activate_module(self, module_name, activation_level=1.0):
        """
        Activate a specific module
        
        Parameters:
        - module_name: Module name
        - activation_level: Activation level (0.0 to 1.0)
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.oversoul_director or not hasattr(self.oversoul_director, "modules"):
            self.active_modules[module_name] = activation_level
            return False
        
        if module_name not in self.oversoul_director.modules:
            return False
        
        self.active_modules[module_name] = activation_level
        
        module = self.oversoul_director.modules[module_name]
        if hasattr(module, "activate"):
            module.activate(activation_level)
        
        return True
    
    def deactivate_module(self, module_name):
        """
        Deactivate a specific module
        
        Parameters:
        - module_name: Module name
        
        Returns:
        - True if successful, False otherwise
        """
        if module_name in self.active_modules:
            self.active_modules[module_name] = 0.0
        
        if not self.oversoul_director or not hasattr(self.oversoul_director, "modules"):
            return False
        
        if module_name not in self.oversoul_director.modules:
            return False
        
        module = self.oversoul_director.modules[module_name]
        if hasattr(module, "deactivate"):
            module.deactivate()
        
        return True
    
    def get_active_modules(self):
        """
        Get active modules
        
        Returns:
        - Dictionary with active modules and their activation levels
        """
        return self.active_modules
    
    def get_activation_history(self):
        """
        Get activation history
        
        Returns:
        - List of activation records
        """
        return self.activation_history
