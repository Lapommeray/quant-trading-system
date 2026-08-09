"""Core module for quant trading system.

Institutional-grade pre-broker core.
All good modules registered here for organism discovery + one-organism wiring.
"""

# Force register the 8 institutional self-coding modules + base
# This ensures get_registered_modules() + organism discover sees them.
import importlib

_INSTITUTIONAL_MODULES = [
    "ofi_detector",
    "whale_flow_detector",
    "mm_intent_detector",
    "cvd_indicator",
    "funding_indicator",
    "volume_profile",
    "cross_asset_leader",
    "real_fed_model",
]

for _mod in _INSTITUTIONAL_MODULES:
    try:
        importlib.import_module(f".{_mod}", package=__name__)
    except Exception:
        # Fail-soft; organism will log
        pass

# Also ensure real_enhanced and institutional
try:
    importlib.import_module(".real_enhanced_indicator", package=__name__)
except Exception:
    pass

__all__ = [
    "HestonVolatility",
    "ML_RSI",
    "OrderFlowImbalance",
    "RegimeDetector",
    "InstitutionalSignalOrchestrator",
    "TradeDecision",
    "ModuleMeshConfig",
    "ModuleMeshOrchestrator",
    "ModuleResult",
    "UnifiedAction",
    "UnifiedTradeResult",
    # Institutional self-coding
    "OFIDetector",
    "WhaleFlowDetector",
    "MMIntentDetector",
    "CVDDetector",
    "FundingIndicator",
    "VolumeProfile",
    "CrossAssetLeader",
    "RealFedModel",
]


def __getattr__(name):
    if name in {
        "HestonVolatility",
        "ML_RSI",
        "OrderFlowImbalance",
        "RegimeDetector",
    }:
        from .indicators import (
            HestonVolatility,
            ML_RSI,
            OrderFlowImbalance,
            RegimeDetector,
        )

        return {
            "HestonVolatility": HestonVolatility,
            "ML_RSI": ML_RSI,
            "OrderFlowImbalance": OrderFlowImbalance,
            "RegimeDetector": RegimeDetector,
        }[name]
    if name in {"InstitutionalSignalOrchestrator", "TradeDecision"}:
        from .institutional_signal_orchestrator import (
            InstitutionalSignalOrchestrator,
            TradeDecision,
        )

        return {
            "InstitutionalSignalOrchestrator": InstitutionalSignalOrchestrator,
            "TradeDecision": TradeDecision,
        }[name]
    if name in {
        "ModuleMeshConfig",
        "ModuleMeshOrchestrator",
        "ModuleResult",
        "UnifiedAction",
        "UnifiedTradeResult",
    }:
        from .module_mesh_orchestrator import (
            ModuleMeshConfig,
            ModuleMeshOrchestrator,
            ModuleResult,
            UnifiedAction,
            UnifiedTradeResult,
        )

        return {
            "ModuleMeshConfig": ModuleMeshConfig,
            "ModuleMeshOrchestrator": ModuleMeshOrchestrator,
            "ModuleResult": ModuleResult,
            "UnifiedAction": UnifiedAction,
            "UnifiedTradeResult": UnifiedTradeResult,
        }[name]
    # Direct access for institutional
    if name == "OFIDetector":
        from .ofi_detector import OFIDetector

        return OFIDetector
    if name == "WhaleFlowDetector":
        from .whale_flow_detector import WhaleFlowDetector

        return WhaleFlowDetector
    if name in {"MMIntentDetector", "MarketMakerIntentDetector"}:
        from .mm_intent_detector import MarketMakerIntentDetector

        return MarketMakerIntentDetector
    if name == "CVDDetector":
        from .cvd_indicator import CVDDetector

        return CVDDetector
    if name == "FundingIndicator":
        from .funding_indicator import FundingIndicator

        return FundingIndicator
    if name == "VolumeProfile":
        from .volume_profile import VolumeProfile

        return VolumeProfile
    if name == "CrossAssetLeader":
        from .cross_asset_leader import CrossAssetLeader

        return CrossAssetLeader
    if name == "RealFedModel":
        from .real_fed_model import RealFedModel

        return RealFedModel
    if name == "RealEnhancedIndicator":
        from .real_enhanced_indicator import RealEnhancedIndicator

        return RealEnhancedIndicator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
