"""Advanced modules package.

This package intentionally avoids eager imports so optional dependencies
(e.g. ccxt, mplfinance) do not break unrelated module imports.
Import submodules directly, e.g. `advanced_modules.port_activity_analyzer`.
"""

__all__ = []
