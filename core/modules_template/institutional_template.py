"""
Template for every auto self-coding institutional module.
Copy this to core/ofi_detector.py etc. and implement compute logic.
Follows ONE ORGANISM spec: register, event bus, learn, diagnose, self_code.
"""
from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from core.data_ring import get_data_ring
import time
import numpy as np

@register_module
class InstitutionalModuleTemplate(BaseTradingModule):
    module_name = "template_module"  # CHANGE THIS
    category = "microstructure"  # general, microstructure, macro, whale, risk
    version = "1.0.0"
    dependencies = ["regime_detector"]  # ensure regime loads first

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        # Adaptive params ONLY these can be auto-tuned by self-coder
        self.config.setdefault("adaptive", {
            "confidence_floor": 0.62,
            "weight_multiplier": 1.0,
            "lookback": 100,
            "cooldown_seconds": 1.0,
            "volatility_multiplier": 1.0,
            "regime_affinity_multiplier": 1.0
        })
        self.last_signal_ts = 0.0
        self.last_whale_event = None
        self.last_regime = "unknown"

    def initialize(self) -> bool:
        bus = get_event_bus()
        # ONE ORGANISM: subscribe to siblings
        bus.subscribe("REGIME_CHANGE", self.on_event)
        bus.subscribe("WHALE_FLOW", self.on_event)
        bus.subscribe("MM_INTENT", self.on_event)
        bus.subscribe("ALPHA_SIGNAL", self.on_event)  # cross-pollination
        return True

    def analyze(self, market_data: dict) -> ModuleResult:
        """
        market_data must contain symbol and data_ring ref
        This is called by organism every tick or every bar.
        Must be <10ms.
        """
        start = time.time()
        symbol = market_data.get("symbol", "BTCUSDT")
        # Read pre-broker core data ring - ZERO COPY, before broker
        ring = market_data.get("data_ring") or get_data_ring(symbol)
        ticks = ring.latest(self.config["adaptive"]["lookback"])

        if len(ticks) < 10 or ring.latency_ms() > 5000:
            # stale feed -> no signal, fail closed
            return ModuleResult(module_name=self.module_name, signal="NEUTRAL", confidence=0.0)

        # ---- YOUR INSTITUTIONAL LOGIC HERE ----
        # Example OFI calculation (replace with real)
        ofi_score = self.compute_ofi(ticks)

        signal = "NEUTRAL"
        conf = 0.0
        if ofi_score > 0.6:
            signal = "BUY"
            conf = min(1.0, ofi_score)
        elif ofi_score < -0.6:
            signal = "SELL"
            conf = min(1.0, abs(ofi_score))

        # Apply regime filter - if crash regime, reduce confidence
        if self.last_regime == "crash":
            conf *= 0.3

        # Apply whale filter - if distribution whale event, flip BUY to HOLD
        if self.last_whale_event and self.last_whale_event.get("type") == "distribution" and signal == "BUY":
            conf *= 0.5

        # Cooldown
        now = time.time()
        if now - self.last_signal_ts < self.config["adaptive"]["cooldown_seconds"]:
            signal = "NEUTRAL"
            conf = 0.0
        else:
            self.last_signal_ts = now

        # Confidence floor gate
        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        latency = (time.time() - start) * 1000

        result = ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=conf * self.config["adaptive"]["weight_multiplier"],
            features={
                "ofi_score": float(ofi_score),
                "regime": self.last_regime,
                "whale": self.last_whale_event,
                "bid": float(ticks[-1]["bid"]),
                "ask": float(ticks[-1]["ask"])
            },
            latency_ms=latency
        )

        # Publish to ONE ORGANISM bus - OPERATIONAL lane
        try:
            get_event_bus().publish(
                "ALPHA_SIGNAL",
                result.to_dict(),
                source=self.module_name,
                priority=EventPriority.OPERATIONAL
            )
        except Exception:
            pass

        return result

    def on_event(self, event):
        """Interconnect: listen to whale, regime, other alphas"""
        if event.event_type == "REGIME_CHANGE":
            self.last_regime = event.payload.get("label", "unknown")
            if self.last_regime == "crash":
                # reduce weight in crash
                self.config["adaptive"]["weight_multiplier"] = 0.3
            else:
                self.config["adaptive"]["weight_multiplier"] = 1.0

        if event.event_type == "WHALE_FLOW":
            self.last_whale_event = event.payload

        if event.event_type == "ALPHA_SIGNAL":
            # Cross-pollination: if funding module says crowded long, reduce BUY conf
            if event.payload.get("module_name") == "funding_detector" and event.payload.get("signal") == "SELL":
                # funding crowded long -> our BUY less attractive
                pass

    def learn_from_outcome(self, outcome: dict) -> dict:
        """
        Called by organism after trade closes. Auto-learn from mistakes.
        """
        if outcome.get("module_name") != self.module_name:
            return {"learned": False}

        pnl = outcome.get("pnl", 0)
        if pnl < 0:
            # Mistake: raise floor
            self.config["adaptive"]["confidence_floor"] = min(0.85, self.config["adaptive"]["confidence_floor"] + 0.02)
            return {
                "learned": True,
                "lesson": f"Raised confidence_floor to {self.config['adaptive']['confidence_floor']:.2f} after loss {pnl:.4f} in regime {outcome.get('regime')}",
                "params": {"confidence_floor": self.config["adaptive"]["confidence_floor"]}
            }
        elif pnl > 0:
            # Reinforce: slightly lower floor
            self.config["adaptive"]["confidence_floor"] = max(0.55, self.config["adaptive"]["confidence_floor"] - 0.005)
            return {"learned": True, "lesson": f"Lowered floor to {self.config['adaptive']['confidence_floor']:.2f} after win"}

        return {"learned": False}

    def diagnose(self, context: dict) -> dict:
        stats = context.get("stats", {})
        win_rate = stats.get("win_rate", 0.5)
        mistake_rate = stats.get("mistake_rate", 0.0)

        if win_rate < 0.48:
            return {
                "issue": "low_win_rate",
                "suggestion": "tighten threshold",
                "params": {"confidence_floor": min(0.80, self.config["adaptive"]["confidence_floor"] + 0.05)}
            }
        if mistake_rate > 0.35:
            return {
                "issue": "high_mistake_rate",
                "suggestion": "increase cooldown",
                "params": {"cooldown_seconds": self.config["adaptive"]["cooldown_seconds"] + 0.5}
            }

        return {"issue": "none"}

    def self_code(self, coder, context=None, apply=True):
        """Delegate to bounded self-coding engine"""
        return super().self_code(coder, context or {}, apply=apply)

    # ---- Institutional compute (numba for speed) ----
    def compute_ofi(self, ticks):
        """Replace with real OFI using bid/ask sizes. Keep fast."""
        # Simple proxy: if bid_size > ask_size => positive OFI
        if len(ticks) < 2:
            return 0.0
        avg_bid = np.mean(ticks["bid_size"][-10:])
        avg_ask = np.mean(ticks["ask_size"][-10:])
        total = avg_bid + avg_ask
        if total == 0:
            return 0.0
        return (avg_bid - avg_ask) / total
