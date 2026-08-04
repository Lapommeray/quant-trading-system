"""
Enhanced automatic module registry for organism wiring.

Scans advanced_modules for both new BaseTradingModule subclasses and legacy
modules, auto-registers them, and exposes Organism wiring API.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

log = logging.getLogger(__name__)

try:
    from core.base_module import BaseTradingModule, register_module
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    BaseTradingModule = object  # type: ignore


class AutoModuleRegistry:
    """Discovers and registers all quant modules automatically."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.advanced_path = self.project_root / "advanced_modules"
        self.core_path = self.project_root / "core"
        self.discovered: Dict[str, Type] = {}
        self.legacy: Dict[str, Type] = {}
        self.load_errors: Dict[str, str] = {}

    def discover(self) -> Dict[str, Any]:
        """Perform full discovery."""
        self.discovered.clear()
        self.legacy.clear()
        self.load_errors.clear()

        # 1. Scan advanced_modules for decorated/new base classes
        self._scan_advanced_modules()
        # 2. Scan core for organism modules
        self._scan_core_modules()
        # 3. Import package to pick up lazy __getattr__ modules that are classes elsewhere
        self._scan_via_importlib()

        return {
            "new_style": list(self.discovered.keys()),
            "legacy": list(self.legacy.keys()),
            "errors": self.load_errors,
            "total": len(self.discovered) + len(self.legacy),
        }

    def _scan_advanced_modules(self):
        if not self.advanced_path.exists():
            return
        for py_file in self.advanced_path.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ("module_interface.py", "module_registry.py"):
                continue
            mod_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    f"advanced_modules.{mod_name}",
                    str(py_file),
                )
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                # Add to sys.modules temporarily to resolve relative imports
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module.__name__:
                        continue
                    # New style: inherits BaseTradingModule
                    if BASE_AVAILABLE and issubclass(obj, BaseTradingModule) and obj is not BaseTradingModule:
                        key = getattr(obj, "module_name", obj.__name__)
                        self.discovered[key] = obj
                        log.debug("Discovered new-style module %s -> %s", key, obj)
                    # Legacy: has initialize and analyze/detect
                    elif hasattr(obj, "initialize") and (hasattr(obj, "analyze") or hasattr(obj, "detect") or hasattr(obj, "decode") or hasattr(obj, "predict")):
                        self.legacy[mod_name] = obj
            except Exception as exc:
                self.load_errors[mod_name] = str(exc)
                log.debug("Discovery error for %s: %s", mod_name, exc)

    def _scan_core_modules(self):
        if not self.core_path.exists():
            return
        for py_file in self.core_path.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ("base_module.py", "event_bus.py", "organism.py", "indicators.py", "qmp_engine_v3.py", "oversoul_director.py", "hyper_evolution.py", "chrono_execution.py"):
                continue
            mod_name = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(f"core.{mod_name}", str(py_file))
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module.__name__:
                        continue
                    if BASE_AVAILABLE and issubclass(obj, BaseTradingModule) and obj is not BaseTradingModule:
                        key = getattr(obj, "module_name", obj.__name__)
                        self.discovered[key] = obj
            except Exception as exc:
                self.load_errors[f"core.{mod_name}"] = str(exc)

    def _scan_via_importlib(self):
        # Attempt to import advanced_modules package __getattr__ lazy modules for completeness
        # The __init__.py lazy map may hold classes like HumanLagExploit etc but not necessarily BaseTradingModule.
        # We already scanned files directly, so this is supplemental.
        try:
            import advanced_modules as adv_pkg  # type: ignore
            # To trigger lazy loads, iterate over known lazy keys manually by reading __all__
            all_keys = getattr(adv_pkg, "_LAZY_IMPORTS", {})
            for key in list(all_keys.keys())[:50]:  # limit
                try:
                    _ = getattr(adv_pkg, key)
                except Exception:
                    continue
        except Exception:
            pass

    def instantiate_all(self, config: Optional[Dict[str, Any]] = None, event_bus: Any = None):
        """Instantiate all discovered new-style modules."""
        instances: Dict[str, Any] = {}
        for name, cls in self.discovered.items():
            try:
                inst = cls(config=config or {}, event_bus=event_bus)
                if hasattr(inst, "initialize"):
                    ok = inst.initialize()
                    if not ok:
                        log.warning("Module %s initialize() returned False", name)
                        continue
                instances[name] = inst
            except Exception as exc:
                log.warning("Failed to instantiate %s: %s", name, exc)
                self.load_errors[name] = str(exc)
        return instances

    def get_module_info(self) -> Dict[str, Dict[str, Any]]:
        info: Dict[str, Dict[str, Any]] = {}
        for name, cls in {**self.discovered, **self.legacy}.items():
            try:
                info[name] = {
                    "module_name": getattr(cls, "module_name", name),
                    "category": getattr(cls, "category", "legacy"),
                    "version": getattr(cls, "version", "0.0.1"),
                    "dependencies": getattr(cls, "dependencies", []),
                    "file": getattr(cls, "__module__", "unknown"),
                    "is_legacy": name in self.legacy,
                }
            except Exception:
                info[name] = {"module_name": name, "error": "info failed"}
        return info


# Singleton accessor
_global_registry: Optional[AutoModuleRegistry] = None


def get_auto_registry() -> AutoModuleRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = AutoModuleRegistry()
        _global_registry.discover()
    return _global_registry
