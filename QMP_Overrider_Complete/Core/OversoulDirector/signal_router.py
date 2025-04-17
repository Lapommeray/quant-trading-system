"""
signal_router.py

Signal Router for OversoulDirector

Routes signals between modules based on market conditions and priority matrix.
"""

import numpy as np
from datetime import datetime

class SignalRouter:
    """
    Signal Router for OversoulDirector
    
    Routes signals between modules based on market conditions and priority matrix.
    """
    
    def __init__(self, oversoul_director=None):
        """
        Initialize the Signal Router
        
        Parameters:
        - oversoul_director: OversoulDirector instance (optional)
        """
        self.oversoul_director = oversoul_director
        self.routing_table = {}
        self.active_routes = []
        self.last_route_time = None
        
    def route_signal(self, signal, source_module, target_modules=None):
        """
        Route a signal from source module to target modules
        
        Parameters:
        - signal: Signal to route
        - source_module: Source module name
        - target_modules: List of target module names (optional)
        
        Returns:
        - Dictionary with routing results
        """
        now = datetime.now()
        
        if target_modules is None:
            target_modules = self._get_default_targets(source_module)
        
        route = {
            "signal": signal,
            "source": source_module,
            "targets": target_modules,
            "timestamp": now,
            "results": {}
        }
        
        for target in target_modules:
            try:
                if self.oversoul_director and hasattr(self.oversoul_director, "modules"):
                    if target in self.oversoul_director.modules:
                        result = self._route_to_module(signal, target)
                        route["results"][target] = {
                            "status": "success",
                            "result": result
                        }
                    else:
                        route["results"][target] = {
                            "status": "error",
                            "message": f"Module {target} not found in oversoul"
                        }
                else:
                    route["results"][target] = {
                        "status": "simulated",
                        "message": "Simulated routing (no oversoul)"
                    }
            except Exception as e:
                route["results"][target] = {
                    "status": "error",
                    "message": str(e)
                }
        
        self.active_routes.append(route)
        self.last_route_time = now
        
        if len(self.active_routes) > 100:
            self.active_routes = self.active_routes[-100:]
        
        return route
    
    def _get_default_targets(self, source_module):
        """
        Get default target modules for a source module
        
        Parameters:
        - source_module: Source module name
        
        Returns:
        - List of default target module names
        """
        default_routing = {
            "phoenix": ["truth", "oversoul"],
            "aurora": ["truth", "oversoul"],
            "qmp": ["truth", "oversoul"],
            "truth": ["ritual", "darwin", "consciousness"],
            "ritual": ["oversoul"],
            "darwin": ["oversoul"],
            "consciousness": ["oversoul"]
        }
        
        return default_routing.get(source_module, ["oversoul"])
    
    def _route_to_module(self, signal, target_module):
        """
        Route a signal to a specific module
        
        Parameters:
        - signal: Signal to route
        - target_module: Target module name
        
        Returns:
        - Module response
        """
        if not self.oversoul_director or not hasattr(self.oversoul_director, "modules"):
            return None
        
        module = self.oversoul_director.modules.get(target_module)
        if not module:
            return None
        
        if target_module == "truth":
            if hasattr(module, "add_signal"):
                module.add_signal("router", signal["direction"], signal["confidence"])
                return {"status": "signal_added"}
        elif target_module == "ritual":
            if hasattr(module, "is_aligned"):
                is_aligned = module.is_aligned(signal["direction"])
                return {"status": "checked", "aligned": is_aligned}
        elif target_module == "darwin":
            if hasattr(module, "optimize"):
                optimized = module.optimize(signal)
                return {"status": "optimized", "result": optimized}
        elif target_module == "consciousness":
            if hasattr(module, "explain"):
                explanation = module.explain(
                    signal["direction"],
                    signal.get("gate_scores"),
                    signal.get("market_data")
                )
                return {"status": "explained", "explanation": explanation}
        
        if hasattr(module, "process"):
            return module.process(signal)
        
        return None
    
    def get_active_routes(self):
        """
        Get active routes
        
        Returns:
        - List of active routes
        """
        return self.active_routes
    
    def clear_routes(self):
        """
        Clear all active routes
        """
        self.active_routes = []
        self.last_route_time = None
