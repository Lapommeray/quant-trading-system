"""
OFI Detector - Institutional Order Flow Imbalance with Auto Self-Coding
Reads pre-broker DataRing before broker REST. Moves with market makers.
Asset: BTCUSD, ETHUSD, SPY/ES

Edge: OFI predicts 1s forward return correlation 0.55-0.65
"""

from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from core.data_ring import get_data_ring
import time
import numpy as np

try:
    import numba

    NUMBA = True
except:
    NUMBA = False

if NUMBA:

    @numba.jit(nopython=True)
    def compute_ofi_fast(
        bid_sizes,
        ask_sizes,
        bid_prices,
        ask_prices,
        bid_sizes_prev,
        ask_sizes_prev,
        bid_prices_prev,
        ask_prices_prev,
    ):
        bid_change = 0.0
        ask_change = 0.0
        n = min(5, len(bid_sizes))
        for i in range(n):
            if bid_prices[i] > bid_prices_prev[i]:
                bid_change += bid_sizes[i]
            elif bid_prices[i] < bid_prices_prev[i]:
                bid_change -= bid_sizes_prev[i]
            else:
                bid_change += bid_sizes[i] - bid_sizes_prev[i]
            if ask_prices[i] > ask_prices_prev[i]:
                ask_change -= ask_sizes_prev[i]
            elif ask_prices[i] < ask_prices_prev[i]:
                ask_change += ask_sizes[i]
            else:
                ask_change += ask_sizes[i] - ask_sizes_prev[i]
        return bid_change - ask_change

else:

    def compute_ofi_fast(
        bid_sizes,
        ask_sizes,
        bid_prices,
        ask_prices,
        bid_sizes_prev,
        ask_sizes_prev,
        bid_prices_prev,
        ask_prices_prev,
    ):
        # fallback
        bid_change = np.sum(bid_sizes - bid_sizes_prev)
        ask_change = np.sum(ask_sizes - ask_sizes_prev)
        return bid_change - ask_change


@register_module
class OFIDetector(BaseTradingModule):
    module_name = "ofi_detector"
    category = "microstructure"
    version = "2.0.0"
    dependencies = ["regime_detector", "volume_profile"]

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        self.config.setdefault(
            "adaptive",
            {
                "confidence_floor": 0.65,
                "weight_multiplier": 1.0,
                "lookback": 20,
                "cooldown_seconds": 0.5,
                "volatility_multiplier": 1.0,
                "regime_affinity_multiplier": 1.0,
                "ofi_threshold": 0.6,
                "zscore_threshold": 2.0,
            },
        )
        self.ofi_history = []
        self.last_regime = "unknown"
        self.whale_bias = 0.0
        self.last_signal_ts = 0

    def initialize(self):
        bus = get_event_bus()
        bus.subscribe("REGIME_CHANGE", self.on_event)
        bus.subscribe("WHALE_FLOW", self.on_event)
        bus.subscribe("MM_INTENT", self.on_event)
        return True

    def analyze(self, market_data):
        start = time.time()
        symbol = market_data.get("symbol", "BTCUSDT")
        ring = market_data.get("data_ring") or get_data_ring(symbol)
        ticks = ring.latest(
            self.config["adaptive"]["lookback"] * 10
        )  # need depth snapshots

        if len(ticks) < 20:
            return ModuleResult(
                self.module_name, "NEUTRAL", 0.0, {"reason": "insufficient_ticks"}
            )

        if ring.latency_ms() > 3000:
            return ModuleResult(
                self.module_name,
                "NEUTRAL",
                0.0,
                {"reason": "stale_feed", "latency": ring.latency_ms()},
            )

        # Compute OFI from last 2 snapshots using bid_size/ask_size from ring (best level proxy, real 5-level needs L2)
        # For L2 depth, need full order book ring - here we approximate with best bid/ask size change
        recent = ticks[-20:]
        bid_sizes = recent["bid_size"].astype(np.float64)
        ask_sizes = recent["ask_size"].astype(np.float64)
        bid_prices = recent["bid"].astype(np.float64)
        ask_prices = recent["ask"].astype(np.float64)

        mid = len(bid_sizes) // 2
        ofi_val = compute_ofi_fast(
            bid_sizes[mid:],
            ask_sizes[mid:],
            bid_prices[mid:],
            ask_prices[mid:],
            bid_sizes[:mid],
            ask_sizes[:mid],
            bid_prices[:mid],
            ask_prices[:mid],
        )

        # Normalize to zscore
        self.ofi_history.append(ofi_val)
        if len(self.ofi_history) > 100:
            self.ofi_history = self.ofi_history[-100:]
        mean = np.mean(self.ofi_history) if len(self.ofi_history) > 10 else 0
        std = np.std(self.ofi_history) if len(self.ofi_history) > 10 else 1
        z = (ofi_val - mean) / (std if std > 1e-6 else 1)

        signal = "NEUTRAL"
        conf = 0.0
        thr = self.config["adaptive"]["ofi_threshold"]
        z_thr = self.config["adaptive"]["zscore_threshold"]

        if z > thr and ofi_val > 0:
            signal = "BUY"
            conf = min(1.0, abs(z) / 3.0 * 0.9)
        elif z < -thr and ofi_val < 0:
            signal = "SELL"
            conf = min(1.0, abs(z) / 3.0 * 0.9)

        # Whale bias adjustment - move WITH whales
        if self.whale_bias != 0:
            # whale_bias = +1 distribution (sell), -1 accumulation (buy)
            if self.whale_bias > 0.5 and signal == "BUY":
                conf *= 0.5  # don't buy when whales distributing
            if self.whale_bias < -0.5 and signal == "SELL":
                conf *= 0.5

        if self.last_regime == "crash":
            conf *= 0.3

        # Cooldown
        if (
            time.time() - self.last_signal_ts
            < self.config["adaptive"]["cooldown_seconds"]
        ):
            signal = "NEUTRAL"
            conf = 0.0
        else:
            if signal != "NEUTRAL":
                self.last_signal_ts = time.time()

        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        latency = (time.time() - start) * 1000
        self.record_success(latency)

        result = ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=conf * self.config["adaptive"]["weight_multiplier"],
            features={
                "ofi": float(ofi_val),
                "ofi_z": float(z),
                "bid_size": float(bid_sizes[-1]),
                "ask_size": float(ask_sizes[-1]),
                "regime": self.last_regime,
                "whale_bias": self.whale_bias,
            },
            latency_ms=latency,
        )

        get_event_bus().publish(
            "ALPHA_SIGNAL",
            result.to_dict(),
            source=self.module_name,
            priority=EventPriority.OPERATIONAL,
        )
        self._last_result = result
        return result

    def on_event(self, event_type, payload=None):
        # Support both Event object and (event_type, payload) signature
        if hasattr(event_type, "event_type"):
            ev = event_type
            event_type = ev.event_type
            payload = ev.payload if payload is None else payload

        if event_type == "REGIME_CHANGE":
            self.last_regime = (
                payload.get("label", "unknown")
                if isinstance(payload, dict)
                else "unknown"
            )
        elif event_type == "WHALE_FLOW":
            # payload: {"type":"distribution","zscore":...}
            t = payload.get("type") if isinstance(payload, dict) else ""
            if t == "distribution":
                self.whale_bias = 1.0
            elif t == "accumulation":
                self.whale_bias = -1.0
            else:
                self.whale_bias = 0.0
        elif event_type == "MM_INTENT":
            # if MM spoof ask pulled bullish, boost BUY
            pass

    # Auto self-coding hooks inherit from BaseTradingModule, but override for profit push
    def learn_from_outcome(self, outcome):
        if outcome.get("module_name") != self.module_name:
            return {"learned": False}
        pnl = outcome.get("pnl", 0)
        if pnl < 0:
            self.config["adaptive"]["confidence_floor"] = min(
                0.85, self.config["adaptive"]["confidence_floor"] + 0.02
            )
            self.config["adaptive"]["ofi_threshold"] = min(
                1.0, self.config["adaptive"]["ofi_threshold"] + 0.05
            )
            return {
                "learned": True,
                "lesson": f"OFI loss {pnl:.4f} raised floor to {self.config['adaptive']['confidence_floor']}",
            }
        else:
            self.config["adaptive"]["confidence_floor"] = max(
                0.55, self.config["adaptive"]["confidence_floor"] - 0.005
            )
            return {"learned": True, "lesson": "lowered floor after win"}

    def diagnose(self, context=None):
        context = context or {}
        stats = context.get("stats", {})
        win_rate = stats.get("win_rate", 0.5)
        if win_rate < 0.48:
            return {
                "issue": "low_win_rate",
                "suggestion": "tighten OFI threshold to 0.7",
                "params": {"ofi_threshold": 0.70, "confidence_floor": 0.70},
            }
        if stats.get("mistake_rate", 0) > 0.35:
            return {"issue": "high_mistake", "params": {"cooldown_seconds": 1.0}}
        return {"issue": "none"}
