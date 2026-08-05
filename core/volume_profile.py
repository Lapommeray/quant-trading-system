"""
Volume Profile - Auto Self-Coding Institutional Module
Computes POC, VAH, VAL, LVN (low volume nodes) for mean reversion and breakout targets.

Edge: Price moves fast through LVN (low volume), slows at HVN/POC.
Target: When price sweeps high and rejects, TP = next LVN or POC.

Reads pre-broker DataRing aggregated by price bins.

Asset: BTCUSD, ETHUSD, SPY
"""
from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from core.data_ring import get_data_ring
import time
import numpy as np

@register_module
class VolumeProfileDetector(BaseTradingModule):
    module_name = "volume_profile"
    category = "microstructure"
    version = "2.0.0"
    dependencies = []

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        self.config.setdefault("adaptive", {
            "confidence_floor": 0.60,
            "weight_multiplier": 1.0,
            "lookback": 1440,  # 1d of 1m bars ~1440, but for ticks 5000
            "cooldown_seconds": 2.0,
            "value_area_pct": 0.70,
            "lvn_threshold_pct": 0.15,
            "bins": 100
        })
        self.last_poc = None
        self.last_vah = None
        self.last_val = None
        self.last_regime = "unknown"
        self.last_signal_ts = 0

    def initialize(self):
        bus = get_event_bus()
        bus.subscribe("REGIME_CHANGE", self.on_event)
        return True

    def analyze(self, market_data):
        start = time.time()
        symbol = market_data.get("symbol", "BTCUSDT")
        # For volume profile, we need OHLCV or tick price+qty
        # Try to get from market_data history or from ring
        ring = market_data.get("data_ring") or get_data_ring(symbol)
        ticks = ring.latest(self.config["adaptive"]["lookback"])

        if len(ticks) < 100:
            return ModuleResult(self.module_name, "NEUTRAL", 0.0, {"reason": "insufficient"})

        prices = ticks["price"].astype(np.float64)
        qtys = ticks["qty"].astype(np.float64)

        # Remove nans
        mask = ~np.isnan(prices) & ~np.isnan(qtys)
        prices = prices[mask]
        qtys = qtys[mask]

        if len(prices) < 50:
            return ModuleResult(self.module_name, "NEUTRAL", 0.0, {"reason": "no_valid"})

        # Histogram by price
        bins = self.config["adaptive"]["bins"]
        hist, bin_edges = np.histogram(prices, bins=bins, weights=qtys)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # POC = bin with max volume
        poc_idx = np.argmax(hist)
        poc = float(bin_centers[poc_idx])
        self.last_poc = poc

        # Value Area 70% around POC
        total_vol = np.sum(hist)
        target_vol = total_vol * self.config["adaptive"]["value_area_pct"]
        # Expand from POC outward
        vol_at_poc = hist[poc_idx]
        cum_vol = vol_at_poc
        l = poc_idx
        r = poc_idx
        while cum_vol < target_vol and (l > 0 or r < len(hist)-1):
            left_vol = hist[l-1] if l>0 else 0
            right_vol = hist[r+1] if r < len(hist)-1 else 0
            if left_vol > right_vol:
                l -= 1
                cum_vol += left_vol
            else:
                r += 1
                cum_vol += right_vol
        vah = float(bin_centers[r]) if r < len(bin_centers) else float(bin_centers[-1])
        val = float(bin_centers[l]) if l >=0 else float(bin_centers[0])
        self.last_vah = vah
        self.last_val = val

        # LVN detection: bins with volume <15% of avg
        avg_vol = np.mean(hist)
        lvn_threshold = avg_vol * self.config["adaptive"]["lvn_threshold_pct"]
        lvn_indices = np.where(hist < lvn_threshold)[0]
        lvn_prices = [float(bin_centers[i]) for i in lvn_indices]

        current_price = float(prices[-1])

        # Signal logic: mean reversion to POC
        signal = "NEUTRAL"
        conf = 0.0

        # Distance from POC in ATR-like units
        atr_proxy = float(np.std(prices[-100:])) if len(prices)>=100 else float(np.std(prices))
        if atr_proxy == 0:
            atr_proxy = current_price * 0.005

        dist_to_poc = (current_price - poc) / atr_proxy

        # If price far above VAH and above POC >2 ATR, mean reversion short
        if current_price > vah and dist_to_poc > 2.0:
            signal = "SELL"
            conf = min(1.0, (dist_to_poc - 2.0) / 2.0 * 0.8 + 0.5)
        elif current_price < val and dist_to_poc < -2.0:
            signal = "BUY"
            conf = min(1.0, (-dist_to_poc - 2.0) / 2.0 * 0.8 + 0.5)

        # If in LVN after sweep, momentum continuation
        # Check if current price inside LVN region
        in_lvn = any(abs(current_price - lvn_p)/current_price < 0.001 for lvn_p in lvn_prices)
        if in_lvn:
            # If coming from above and entering LVN downwards, expect fast move down
            # Use recent price trend
            recent_trend = np.mean(prices[-10:]) - np.mean(prices[-20:-10]) if len(prices)>=20 else 0
            if recent_trend < 0 and current_price < poc:
                # continuation down through LVN
                signal = "SELL"
                conf = max(conf, 0.65)
            elif recent_trend > 0 and current_price > poc:
                signal = "BUY"
                conf = max(conf, 0.65)

        if time.time() - self.last_signal_ts < self.config["adaptive"]["cooldown_seconds"]:
            signal = "NEUTRAL"
            conf = 0.0
        else:
            if signal != "NEUTRAL":
                self.last_signal_ts = time.time()

        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        latency = (time.time() - start)*1000
        self.record_success(latency)

        features = {
            "poc": poc,
            "vah": vah,
            "val": val,
            "current": current_price,
            "dist_to_poc_atr": float(dist_to_poc),
            "lvn_count": len(lvn_prices),
            "lvn_prices": lvn_prices[:5],  # top few
            "atr_proxy": atr_proxy
        }

        result = ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=conf * self.config["adaptive"]["weight_multiplier"],
            features=features,
            latency_ms=latency
        )

        try:
            get_event_bus().publish("ALPHA_SIGNAL", result.to_dict(), source=self.module_name, priority=EventPriority.OPERATIONAL)
            get_event_bus().publish("VOLUME_PROFILE", features, source=self.module_name, priority=EventPriority.OPERATIONAL)
        except:
            pass

        self._last_result = result
        return result

    def on_event(self, event_type, payload=None):
        if hasattr(event_type, 'event_type'):
            ev = event_type
            event_type = ev.event_type
            payload = ev.payload
        if event_type == "REGIME_CHANGE":
            self.last_regime = payload.get("label", "unknown") if isinstance(payload, dict) else "unknown"

    def learn_from_outcome(self, outcome):
        if outcome.get("module_name") != self.module_name:
            return {"learned": False}
        pnl = outcome.get("pnl", 0)
        if pnl < 0:
            self.config["adaptive"]["confidence_floor"] = min(0.85, self.config["adaptive"]["confidence_floor"] + 0.02)
            return {"learned": True}
        else:
            self.config["adaptive"]["confidence_floor"] = max(0.55, self.config["adaptive"]["confidence_floor"] - 0.005)
            return {"learned": True}
        return {"learned": False}

    def diagnose(self, context=None):
        stats = (context or {}).get("stats", {})
        if stats.get("win_rate", 0.5) < 0.48:
            return {"issue": "low_win_rate", "params": {"confidence_floor": 0.70, "value_area_pct": 0.75}}
        return {"issue": "none"}
