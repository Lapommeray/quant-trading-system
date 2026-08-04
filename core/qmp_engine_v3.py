"""
QMPUltraEngine v3 - QuantConnect REMOVED, pure Python version.

This is a cleaned version post PR #117 that removes all QuantConnect dependencies.
No AlgorithmImports. No QCAlgorithm. Uses only local modules and stubs.

For legacy QC version, see archive/ or Deco_* folders.
Active runtime uses autonomy/organism.py + autonomy/consensus.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime as _dt
from typing import Any, Dict

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ---- Pure Python TradeBar (no QC) ----
class TradeBar:
    def __init__(self):
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0
        self._end_time = None

    @property
    def Open(self): return self._open
    @Open.setter
    def Open(self, v): self._open = float(v)
    @property
    def High(self): return self._high
    @High.setter
    def High(self, v): self._high = float(v)
    @property
    def Low(self): return self._low
    @Low.setter
    def Low(self, v): self._low = float(v)
    @property
    def Close(self): return self._close
    @Close.setter
    def Close(self, v): self._close = float(v)
    @property
    def Volume(self): return self._volume
    @Volume.setter
    def Volume(self, v): self._volume = float(v)
    @property
    def EndTime(self): return self._end_time
    @EndTime.setter
    def EndTime(self, v): self._end_time = v


def _stub(name):
    class Stub:
        def __init__(self, *a, **k): pass
        def decode(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def detect(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def predict(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def check_transfers(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def check_movements(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def analyze(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def calculate_risk(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def select_hierarchy(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def fuse_signals(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def predict_price_direction(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def route_trading_signal(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}
        def is_isolated(self, **kw): return False
        def get_isolation_info(self, **kw): return None
        def protect(self, *a, **k): return True
        def pre_trade_check(self, *a, **k): return {"compliant": True}
    Stub.__name__ = name
    return Stub


# All modules as stubs or real if available without QC
try:
    from advanced_modules.human_lag_exploit import HumanLagExploit
except Exception:
    HumanLagExploit = _stub("HumanLagExploit")

try:
    from advanced_modules.invisible_data_miner import InvisibleDataMiner
except Exception:
    class InvisibleDataMiner:
        def __init__(self, *a, **k): pass
        def _extract_patterns(self, *a, **k): return {"confidence": 0.5, "direction": "NEUTRAL"}

try:
    from advanced_modules.meta_adaptive_ai import MetaAdaptiveAI
except Exception:
    MetaAdaptiveAI = _stub("MetaAdaptiveAI")

try:
    from advanced_modules.self_destruct_protocol import SelfDestructProtocol
except Exception:
    SelfDestructProtocol = _stub("SelfDestructProtocol")

try:
    from advanced_modules.quantum_sentiment_decoder import QuantumSentimentDecoder
except Exception:
    QuantumSentimentDecoder = _stub("QuantumSentimentDecoder")

try:
    from advanced_modules.btc_offchain_monitor import BTCOffchainMonitor
except Exception:
    BTCOffchainMonitor = _stub("BTCOffchainMonitor")

try:
    from advanced_modules.fed_jet_monitor import FedJetMonitor
except Exception:
    FedJetMonitor = _stub("FedJetMonitor")

try:
    from advanced_modules.spoofing_detector import SpoofingDetector
except Exception:
    SpoofingDetector = _stub("SpoofingDetector")

try:
    from advanced_modules.compliance_check import ComplianceCheck
except Exception:
    ComplianceCheck = _stub("ComplianceCheck")

try:
    from advanced_modules.stress_detector import StressDetector
except Exception:
    StressDetector = _stub("StressDetector")

try:
    from advanced_modules.port_activity_analyzer import PortActivityAnalyzer
except Exception:
    PortActivityAnalyzer = _stub("PortActivityAnalyzer")

try:
    from advanced_modules.dna_breath import DNABreath
except Exception:
    DNABreath = _stub("DNABreath")

try:
    from advanced_modules.dna_overlord import DNAOverlord
except Exception:
    DNAOverlord = _stub("DNAOverlord")

try:
    from advanced_modules.spectral_signal_fusion import SpectralSignalFusion
except Exception:
    SpectralSignalFusion = _stub("SpectralSignalFusion")

try:
    from advanced_modules.time_fractal_fft import TimeFractalFFT
except Exception:
    TimeFractalFFT = _stub("TimeFractalFFT")

try:
    from advanced_modules.void_trader_chart_renderer import VoidTraderChartRenderer
except Exception:
    VoidTraderChartRenderer = _stub("VoidTraderChartRenderer")

try:
    from advanced_modules.meta_conscious_routing_layer import MetaConsciousRoutingLayer
except Exception:
    MetaConsciousRoutingLayer = _stub("MetaConsciousRoutingLayer")

try:
    from defense.atlantean_shield import AtlanteanShield
except Exception:
    AtlanteanShield = _stub("AtlanteanShield")


# Ultra modules - all stubbed (no QC)
EmotionDNADecoder = _stub("EmotionDNADecoder")
FractalResonanceGate = _stub("FractalResonanceGate")
QuantumTremorScanner = _stub("QuantumTremorScanner")
IntentionDecoder = _stub("IntentionDecoder")
SacredEventAlignment = _stub("SacredEventAlignment")
AstroGeoSync = _stub("AstroGeoSync")
FutureShadowDecoder = _stub("FutureShadowDecoder")
BlackSwanProtector = _stub("BlackSwanProtector")
MarketThoughtFormInterpreter = _stub("MarketThoughtFormInterpreter")
RealityDisplacementMatrix = _stub("RealityDisplacementMatrix")


try:
    from core.event_bus import get_event_bus
    _EVENT_BUS = get_event_bus()
except Exception:
    _EVENT_BUS = None


class QMPUltraEngine:
    """QC-free Ultra Engine - uses only local modules. No AlgorithmImports."""

    def __init__(self, algorithm=None):
        # algorithm param ignored - kept for backward compat but not used
        # No QC algorithm object required
        self._dummy_time = _dt.now()
        self.history = []
        self.gate_scores = {}
        self.last_confidence = 0.0
        self.last_signal = None
        self.last_signal_time = None
        self._local_event_bus = _EVENT_BUS

        def _make(cls):
            try:
                return cls()
            except TypeError:
                try:
                    return cls(None)
                except Exception:
                    # Fallback to stub instance
                    s = _stub(cls.__name__ if hasattr(cls, "__name__") else "stub")
                    return s()

        self.modules = {
            "emotion_dna": _make(EmotionDNADecoder),
            "fractal_resonance": _make(FractalResonanceGate),
            "quantum_tremor": _make(QuantumTremorScanner),
            "intention": _make(IntentionDecoder),
            "sacred_event": _make(SacredEventAlignment),
            "astro_geo": _make(AstroGeoSync),
            "future_shadow": _make(FutureShadowDecoder),
            "black_swan": _make(BlackSwanProtector),
            "market_thought": _make(MarketThoughtFormInterpreter),
            "reality_matrix": _make(RealityDisplacementMatrix),
            "human_lag": _make(HumanLagExploit),
            "invisible_data": _make(InvisibleDataMiner),
            "meta_adaptive": _make(MetaAdaptiveAI),
            "quantum_sentiment": _make(QuantumSentimentDecoder),
            "btc_offchain": _make(BTCOffchainMonitor),
            "fed_jet": _make(FedJetMonitor),
            "spoofing": _make(SpoofingDetector),
            "stress": _make(StressDetector),
            "port_activity": _make(PortActivityAnalyzer),
            "dna_breath": _make(DNABreath),
            "dna_overlord": _make(DNAOverlord),
            "spectral_fusion": _make(SpectralSignalFusion),
            "time_fractal_fft": _make(TimeFractalFFT),
            "void_renderer": _make(VoidTraderChartRenderer),
            "meta_routing": _make(MetaConsciousRoutingLayer),
        }

        self.compliance = _make(ComplianceCheck)
        self.self_destruct = _make(SelfDestructProtocol)

        self.module_weights = {
            "emotion_dna": 0.06, "fractal_resonance": 0.06, "quantum_tremor": 0.06, "intention": 0.08,
            "sacred_event": 0.03, "astro_geo": 0.03, "future_shadow": 0.08, "black_swan": 0.06,
            "market_thought": 0.06, "reality_matrix": 0.06, "human_lag": 0.06, "invisible_data": 0.06,
            "meta_adaptive": 0.06, "quantum_sentiment": 0.06, "btc_offchain": 0.04, "fed_jet": 0.04,
            "spoofing": 0.04, "stress": 0.04, "port_activity": 0.04, "dna_breath": 0.05,
            "dna_overlord": 0.05, "spectral_fusion": 0.05, "time_fractal_fft": 0.04,
            "void_renderer": 0.05, "meta_routing": 0.05,
        }

        self.confidence_threshold = 0.7
        self.min_gate_score = 0.6
        self.confidence_field_map = {"future_shadow": "confidence", "black_swan": "black_swan_risk", "market_thought": "confidence", "reality_matrix": "confidence"}
        self.direction_field_map = {"future_shadow": "future_direction", "market_thought": "collective_intent", "reality_matrix": "primary_direction"}
        self.activated_modules = {"atlantean_shield": AtlanteanShield()}

    def _publish_signal(self, payload: Dict[str, Any]):
        if self._local_event_bus:
            try:
                self._local_event_bus.publish("SIGNAL_GENERATED", payload, source="QMPUltraEngine")
            except Exception:
                pass

    def generate_signal(self, symbol, history_data):
        if not self._validate_history_data(history_data):
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}, "symbol": symbol}
            self._publish_signal(res)
            return res

        if self.self_destruct.is_isolated(symbol=symbol):
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}, "symbol": symbol}
            self._publish_signal(res)
            return res

        if not self.activated_modules["atlantean_shield"].protect(symbol):
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}, "symbol": symbol}
            self._publish_signal(res)
            return res

        history_bars = self._convert_history_to_tradebars(history_data["1m"])
        gate_scores = {}
        directions = {}
        module_results = {}

        for module_name, module in self.modules.items():
            if self.self_destruct.is_isolated(module_name=module_name):
                continue
            try:
                if module_name == "human_lag":
                    result = module.detect(symbol, history_data)
                elif module_name == "invisible_data":
                    result = module._extract_patterns(history_data["1m"], "1m", str(symbol))
                elif module_name == "meta_adaptive":
                    result = module.predict(history_data["1m"].values)
                elif module_name == "quantum_sentiment":
                    result = module.decode(symbol, history_data)
                elif module_name == "btc_offchain":
                    result = module.check_transfers(self._dummy_time)
                elif module_name == "fed_jet":
                    result = module.check_movements({})
                elif module_name == "spoofing":
                    result = module.detect(symbol, history_data)
                elif module_name == "stress":
                    result = module.detect(symbol, self._dummy_time)
                elif module_name == "port_activity":
                    result = module.analyze({})
                elif module_name == "dna_breath":
                    result = module.calculate_risk("neutral", 0.05)
                elif module_name == "dna_overlord":
                    result = module.select_hierarchy()
                elif module_name == "spectral_fusion":
                    class MockComponents:
                        emotion = 0.5; volatility = 0.3; volume = 0.2; entropy = 0.4
                    result = module.fuse_signals("crypto", MockComponents())
                elif module_name == "void_renderer":
                    result = {"void_signals": [], "confidence": 0.5, "direction": "NEUTRAL"}
                elif module_name == "meta_routing":
                    result = module.route_trading_signal({"direction": "BUY", "confidence": 0.7}, {"entropy": 0.5, "liquidity": 0.8}, {})
                elif module_name == "time_fractal_fft":
                    closes = history_data["1m"]["close"].values if "close" in history_data["1m"].columns else np.random.randn(100)
                    result = module.predict_price_direction(closes)
                else:
                    result = module.decode(symbol, history_bars)
            except Exception:
                result = {"confidence": 0.0, "direction": "NEUTRAL"}

            module_results[module_name] = result
            gate_scores[module_name] = self._extract_confidence(module_name, result)
            direction = self._extract_direction(module_name, result)
            if direction:
                directions[module_name] = direction

        if "black_swan" in gate_scores:
            gate_scores["black_swan"] = 1.0 - gate_scores["black_swan"]

        confidence = sum(gate_scores.get(k, 0) * self.module_weights.get(k, 0) for k in gate_scores)
        self.gate_scores = gate_scores
        self.last_confidence = confidence

        gates_pass = all(score >= self.min_gate_score for score in gate_scores.values())

        black_swan_active = False
        if "black_swan" in module_results and isinstance(module_results["black_swan"], dict):
            black_swan_active = module_results["black_swan"].get("protection_active", False)

        if black_swan_active:
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol}
            self._publish_signal(res)
            return res

        compliance_result = self.compliance.pre_trade_check(symbol)
        if not compliance_result.get("compliant", False):
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol}
            self._publish_signal(res)
            return res

        if gates_pass and confidence >= self.confidence_threshold:
            direction_votes = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}
            for module, direction in directions.items():
                if direction in direction_votes:
                    direction_votes[direction] += self.module_weights.get(module, 0.1)
            final_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            if final_direction == "NEUTRAL" or direction_votes[final_direction] < 0.5:
                res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol}
                self._publish_signal(res)
                return res

            if "meta_adaptive" in module_results and isinstance(module_results["meta_adaptive"], dict):
                meta_confidence = module_results["meta_adaptive"].get("confidence", 0.0)
                meta_direction = module_results["meta_adaptive"].get("direction", None)
                if meta_direction and meta_direction != final_direction and meta_confidence > 0.8:
                    final_direction = meta_direction

            self.last_signal = final_direction
            self.last_signal_time = _dt.now()
            res = {"final_signal": final_direction, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol, "votes": direction_votes, "weighted_confidence": confidence}
            self._publish_signal(res)
            return res
        else:
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol}
            self._publish_signal(res)
            return res

    def _extract_confidence(self, module_name, result):
        if not isinstance(result, dict):
            return 1.0 if result and result != "WAIT" else 0.0
        confidence_field = self.confidence_field_map.get(module_name, "confidence")
        if confidence_field in result:
            try: return float(result[confidence_field])
            except: return 0.5
        elif "confidence" in result:
            try: return float(result["confidence"])
            except: return 0.5
        else:
            return 0.5

    def _extract_direction(self, module_name, result):
        if not isinstance(result, dict):
            if isinstance(result, str) and result in ["BUY", "SELL", "NEUTRAL", "WAIT"]:
                return "NEUTRAL" if result == "WAIT" else result
            return None
        direction_field = self.direction_field_map.get(module_name, "direction")
        if direction_field in result:
            return result[direction_field]
        elif "direction" in result:
            return result["direction"]
        else:
            return None

    def _validate_history_data(self, history_data):
        required = ["1m", "5m", "10m", "15m", "20m", "25m"]
        if not all(tf in history_data for tf in required):
            return False
        if len(history_data["1m"]) < 60:
            return False
        return True

    def _convert_history_to_tradebars(self, df):
        if df.empty:
            return []
        trade_bars = []
        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            return []
        for idx, row in df.iterrows():
            bar = TradeBar()
            bar.Open = row["Open"]; bar.High = row["High"]; bar.Low = row["Low"]; bar.Close = row["Close"]; bar.Volume = row["Volume"] if "Volume" in row else 0; bar.EndTime = idx
            trade_bars.append(bar)
        return trade_bars
