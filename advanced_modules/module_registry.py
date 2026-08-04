"""Compatibility registry for active organism modules.

Historical documentation referenced this module before the canonical runtime
was consolidated.  It now delegates to ``autonomy`` and does not instantiate
legacy research modules with missing or unsafe dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AutoModuleRegistry:
    """Discover and inspect modules implementing the active base contract."""

    def __init__(self, packages: Optional[tuple[str, ...]] = None):
        self.packages = packages or ("autonomy", "core", "advanced_modules")
        self.modules: Dict[str, Any] = {}

    def discover(self) -> Dict[str, Any]:
        from autonomy.organism import ModuleAutoDiscovery

        self.modules = ModuleAutoDiscovery.discover_decorated()
        for package in self.packages:
            self.modules.update(ModuleAutoDiscovery.discover_in_package(package))
        return dict(self.modules)

    def get(self, name: str) -> Any:
        if not self.modules:
            self.discover()
        return self.modules.get(name)

    def names(self) -> list[str]:
        if not self.modules:
            self.discover()
        return sorted(self.modules)


# Common functional API.
def get_auto_registry(packages: Optional[tuple[str, ...]] = None) -> AutoModuleRegistry:
    registry = AutoModuleRegistry(packages=packages)
    registry.discover()
    return registry


__all__ = ["AutoModuleRegistry", "get_auto_registry"]
