"""Core module for quant trading system."""

__all__ = [
    "HestonVolatility",
    "ML_RSI",
    "OrderFlowImbalance",
    "RegimeDetector",
    "InstitutionalSignalOrchestrator",
    "TradeDecision",
]


def __getattr__(name):
    if name in {"HestonVolatility", "ML_RSI", "OrderFlowImbalance", "RegimeDetector"}:
        from .indicators import HestonVolatility, ML_RSI, OrderFlowImbalance, RegimeDetector
        return {
            "HestonVolatility": HestonVolatility,
            "ML_RSI": ML_RSI,
            "OrderFlowImbalance": OrderFlowImbalance,
            "RegimeDetector": RegimeDetector,
        }[name]
    if name in {"InstitutionalSignalOrchestrator", "TradeDecision"}:
        from .institutional_signal_orchestrator import InstitutionalSignalOrchestrator, TradeDecision
        return {
            "InstitutionalSignalOrchestrator": InstitutionalSignalOrchestrator,
            "TradeDecision": TradeDecision,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
