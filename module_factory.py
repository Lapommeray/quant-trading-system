"""
Module Factory for Advanced Trading System
"""
import importlib
import importlib.util
import os
import sys
from typing import Dict, Any, List, Optional

class AdvancedModuleFactory:
    """Factory for creating and managing advanced trading modules"""
    
    def __init__(self):
        self.module_registry = {}
        self.module_categories = [
            "quantum_error_correction",
            "market_reality_anchors", 
            "cern_safeguards",
            "temporal_stability",
            "elon_discovery",
            "cern_data",
            "hardware_adaptation",
            "ai_only_trades"
        ]
        self._discover_modules()
        
    def _discover_modules(self):
        """Automatically discover all available modules"""
        base_path = os.path.join(os.path.dirname(__file__), "advanced_modules")
        
        for category in self.module_categories:
            category_path = os.path.join(base_path, category)
            if os.path.exists(category_path):
                self.module_registry[category] = []
                
                for file in os.listdir(category_path):
                    if file.endswith(".py") and not file.startswith("__"):
                        module_name = file[:-3]
                        try:
                            current_dir = os.path.dirname(__file__)
                            if current_dir not in sys.path:
                                sys.path.insert(0, current_dir)
                            
                            file_path = os.path.join(category_path, f"{module_name}.py")
                            spec = importlib.util.spec_from_file_location(
                                f"advanced_modules.{category}.{module_name}", 
                                file_path
                            )
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                module_path = f"advanced_modules.{category}.{module_name}"
                            else:
                                continue
                            
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (isinstance(attr, type) and 
                                    hasattr(attr, '__init__') and 
                                    hasattr(attr, 'initialize') and
                                    attr_name != 'AdvancedModuleInterface'):
                                    self.module_registry[category].append({
                                        "name": module_name,
                                        "class": attr,
                                        "path": module_path
                                    })
                                    break
                                    
                        except (ImportError, AttributeError, Exception) as e:
                            print(f"Could not import {module_name}: {e}")
                            continue
                            
    def create_module(self, category: str, module_name: str, config: Optional[Dict[str, Any]] = None):
        """Create a specific module instance"""
        if category not in self.module_registry:
            raise ValueError(f"Unknown category: {category}")
            
        for module_info in self.module_registry[category]:
            if module_info["name"] == module_name:
                return module_info["class"](config)
                
        raise ValueError(f"Module {module_name} not found in category {category}")
        
    def create_all_modules(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
        """Create all available modules"""
        all_modules = {}
        
        for category, modules in self.module_registry.items():
            all_modules[category] = []
            for module_info in modules:
                try:
                    instance = module_info["class"](config)
                    if instance.initialize():
                        all_modules[category].append(instance)
                except Exception as e:
                    print(f"Failed to create {module_info['name']}: {e}")
                    
        return all_modules
        
    def get_module_count(self) -> int:
        """Get total number of available modules"""
        return sum(len(modules) for modules in self.module_registry.values())
        
    def get_category_modules(self, category: str) -> List[str]:
        """Get list of module names in a category"""
        if category not in self.module_registry:
            return []
        return [m["name"] for m in self.module_registry[category]]
        
    def list_all_modules(self) -> Dict[str, List[str]]:
        """List all available modules by category"""
        return {cat: self.get_category_modules(cat) for cat in self.module_categories}

def get_module_factory() -> AdvancedModuleFactory:
    """Get singleton module factory instance"""
    if not hasattr(get_module_factory, '_instance'):
        get_module_factory._instance = AdvancedModuleFactory()
    return get_module_factory._instance
