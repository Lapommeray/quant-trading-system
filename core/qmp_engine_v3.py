# QuantConnect removal: AlgorithmImports is now optional, replaced by local simulation.
# Historical QC code is archived under legacy/ and docs/ARCHIVE_CAVEAT.md
# This engine now works standalone and integrates with Organism + EventBus + OKX.

from __future__ import annotations

import logging
import sys
from datetime import datetime as _dt
from typing import Any, Dict, Optional

# --- AlgorithmImports optional ---
try:
    from AlgorithmImports import *  # type: ignore
except ImportError:
    try:
        from mock_algorithm_imports import *  # type: ignore
    except ImportError:
        import pandas as pd
        import numpy as np

        class QCAlgorithm:  # type: ignore
            def __init__(self):
                self.Time = _dt.now()
                self.debug_messages = []

            def Debug(self, msg):
                print(f"DEBUG: {msg}")
                self.debug_messages.append(msg)

            def Log(self, msg):
                print(f"LOG: {msg}")

        class TradeBar:  # type: ignore
            def __init__(self):
                self._open = 0.0
                self._high = 0.0
                self._low = 0.0
                self._close = 0.0
                self._volume = 0
                self._end_time = None

            @property
            def Open(self):
                return self._open

            @Open.setter
            def Open(self, v):
                self._open = float(v)

            @property
            def High(self):
                return self._high

            @High.setter
            def High(self, v):
                self._high = float(v)

            @property
            def Low(self):
                return self._low

            @Low.setter
            def Low(self, v):
                self._low = float(v)

            @property
            def Close(self):
                return self._close

            @Close.setter
            def Close(self, v):
                self._close = float(v)

            @property
            def Volume(self):
                return self._volume

            @Volume.setter
            def Volume(self, v):
                self._volume = float(v)

            @property
            def EndTime(self):
                return self._end_time

            @EndTime.setter
            def EndTime(self, v):
                self._end_time = v

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# --- ultra modules stubs (optional) ---
def _stub_decoder(name):
    class Stub:
        def __init__(self, *a, **k):
            pass

        def decode(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}

    Stub.__name__ = name
    return Stub


try:
    from ultra_modules.emotion_dna_decoder import EmotionDNADecoder  # type: ignore
except Exception:
    EmotionDNADecoder = _stub_decoder("EmotionDNADecoder")

try:
    from ultra_modules.fractal_resonance_gate import FractalResonanceGate  # type: ignore
except Exception:
    FractalResonanceGate = _stub_decoder("FractalResonanceGate")

try:
    from ultra_modules.quantum_tremor_scanner import QuantumTremorScanner  # type: ignore
except Exception:
    QuantumTremorScanner = _stub_decoder("QuantumTremorScanner")

try:
    from ultra_modules.intention_decoder import IntentionDecoder  # type: ignore
except Exception:
    IntentionDecoder = _stub_decoder("IntentionDecoder")

try:
    from ultra_modules.sacred_event_alignment import SacredEventAlignment  # type: ignore
except Exception:
    SacredEventAlignment = _stub_decoder("SacredEventAlignment")

try:
    from ultra_modules.astro_geo_sync import AstroGeoSync  # type: ignore
except Exception:
    AstroGeoSync = _stub_decoder("AstroGeoSync")

try:
    from ultra_modules.future_shadow_decoder import FutureShadowDecoder  # type: ignore
except Exception:

    class FutureShadowDecoder:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def decode(self, *a, **k):
            return {"confidence": 0.5, "direction": "BUY", "future_direction": "BUY"}


try:
    from ultra_modules.black_swan_protector import BlackSwanProtector  # type: ignore
except Exception:

    class BlackSwanProtector:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def decode(self, *a, **k):
            return {"black_swan_risk": 0.1, "protection_active": False, "confidence": 0.9}


try:
    from ultra_modules.market_thought_form_interpreter import MarketThoughtFormInterpreter  # type: ignore
except Exception:
    MarketThoughtFormInterpreter = _stub_decoder("MarketThoughtFormInterpreter")

try:
    from ultra_modules.reality_displacement_matrix import RealityDisplacementMatrix  # type: ignore
except Exception:
    RealityDisplacementMatrix = _stub_decoder("RealityDisplacementMatrix")


# --- advanced modules optional ---
try:
    from advanced_modules.human_lag_exploit import HumanLagExploit  # type: ignore
except Exception:

    class HumanLagExploit:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def detect(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.invisible_data_miner import InvisibleDataMiner  # type: ignore
except Exception:

    class InvisibleDataMiner:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def _extract_patterns(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.meta_adaptive_ai import MetaAdaptiveAI  # type: ignore
except Exception:

    class MetaAdaptiveAI:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.self_destruct_protocol import SelfDestructProtocol  # type: ignore
except Exception:

    class SelfDestructProtocol:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def is_isolated(self, **kw):
            return False

        def get_isolation_info(self, **kw):
            return None


try:
    from advanced_modules.quantum_sentiment_decoder import QuantumSentimentDecoder  # type: ignore
except Exception:
    QuantumSentimentDecoder = _stub_decoder("QuantumSentimentDecoder")

try:
    from advanced_modules.btc_offchain_monitor import BTCOffchainMonitor  # type: ignore
except Exception:

    class BTCOffchainMonitor:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def check_transfers(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.fed_jet_monitor import FedJetMonitor  # type: ignore
except Exception:

    class FedJetMonitor:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def check_movements(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.spoofing_detector import SpoofingDetector  # type: ignore
except Exception:

    class SpoofingDetector:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def detect(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.compliance_check import ComplianceCheck  # type: ignore
except Exception:

    class ComplianceCheck:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def pre_trade_check(self, *a, **k):
            return {"compliant": True}


try:
    from advanced_modules.stress_detector import StressDetector  # type: ignore
except Exception:

    class StressDetector:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def detect(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.port_activity_analyzer import PortActivityAnalyzer  # type: ignore
except Exception:

    class PortActivityAnalyzer:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def analyze(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.dna_breath import DNABreath  # type: ignore
except Exception:

    class DNABreath:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def calculate_risk(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.dna_overlord import DNAOverlord  # type: ignore
except Exception:

    class DNAOverlord:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def select_hierarchy(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.spectral_signal_fusion import SpectralSignalFusion  # type: ignore
except Exception:

    class SpectralSignalFusion:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def fuse_signals(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.time_fractal_fft import TimeFractalFFT  # type: ignore
except Exception:

    class TimeFractalFFT:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def predict_price_direction(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.void_trader_chart_renderer import VoidTraderChartRenderer  # type: ignore
except Exception:

    class VoidTraderChartRenderer:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def render_chart(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from advanced_modules.meta_conscious_routing_layer import MetaConsciousRoutingLayer  # type: ignore
except Exception:

    class MetaConsciousRoutingLayer:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def route_trading_signal(self, *a, **k):
            return {"confidence": 0.5, "direction": "NEUTRAL"}


try:
    from defense.atlantean_shield import AtlanteanShield  # type: ignore
except Exception:

    class AtlanteanShield:  # type: ignore
        def __init__(self, *a, **k):
            pass

        def protect(self, *a, **k):
            return True


# EventBus optional
try:
    from core.event_bus import get_event_bus  # type: ignore

    _EVENT_BUS = get_event_bus()
except Exception:
    _EVENT_BUS = None


class QMPUltraEngine:
    """QC-free Ultra Engine with optional EventBus + Organism compatibility."""

    def __init__(self, algorithm=None):
        if algorithm is None:
            try:
                from mock_algorithm_imports import QCAlgorithm as _QC

                algorithm = _QC()
            except Exception:
                algorithm = None

        self.algo = algorithm if algorithm is not None else self._make_dummy_algo()
        self.history = []
        self.gate_scores = {}
        self.last_confidence = 0.0
        self.last_signal = None
        self.last_signal_time = None
        self._local_event_bus = _EVENT_BUS

        # modules
        algo_ref = self.algo
        self.modules = {
            "emotion_dna": EmotionDNADecoder(algo_ref),
            "fractal_resonance": FractalResonanceGate(algo_ref),
            "quantum_tremor": QuantumTremorScanner(algo_ref),
            "intention": IntentionDecoder(algo_ref),
            "sacred_event": SacredEventAlignment(algo_ref),
            "astro_geo": AstroGeoSync(algo_ref),
            "future_shadow": FutureShadowDecoder(algo_ref),
            "black_swan": BlackSwanProtector(algo_ref),
            "market_thought": MarketThoughtFormInterpreter(algo_ref),
            "reality_matrix": RealityDisplacementMatrix(algo_ref),
            "human_lag": HumanLagExploit(algo_ref),
            "invisible_data": InvisibleDataMiner(algo_ref),
            "meta_adaptive": MetaAdaptiveAI(algo_ref),
            "quantum_sentiment": QuantumSentimentDecoder(algo_ref),
            "btc_offchain": BTCOffchainMonitor(algo_ref),
            "fed_jet": FedJetMonitor(algo_ref),
            "spoofing": SpoofingDetector(algo_ref),
            "stress": StressDetector(algo_ref),
            "port_activity": PortActivityAnalyzer(algo_ref),
            "dna_breath": DNABreath(),
            "dna_overlord": DNAOverlord(),
            "spectral_fusion": SpectralSignalFusion(),
            "time_fractal_fft": TimeFractalFFT(),
            "void_renderer": VoidTraderChartRenderer(
                type(
                    "MockDataFeed",
                    (),
                    {"get_ohlcv": lambda self, symbol, timeframe: [[1704067200000, 50000, 50100, 49900, 50050, 1000]]},
                )()
            ),
            "meta_routing": MetaConsciousRoutingLayer(),
        }

        self.compliance = ComplianceCheck(algo_ref)
        self.self_destruct = SelfDestructProtocol(algo_ref)

        self.module_weights = {
            "emotion_dna": 0.06,
            "fractal_resonance": 0.06,
            "quantum_tremor": 0.06,
            "intention": 0.08,
            "sacred_event": 0.03,
            "astro_geo": 0.03,
            "future_shadow": 0.08,
            "black_swan": 0.06,
            "market_thought": 0.06,
            "reality_matrix": 0.06,
            "human_lag": 0.06,
            "invisible_data": 0.06,
            "meta_adaptive": 0.06,
            "quantum_sentiment": 0.06,
            "btc_offchain": 0.04,
            "fed_jet": 0.04,
            "spoofing": 0.04,
            "stress": 0.04,
            "port_activity": 0.04,
            "dna_breath": 0.05,
            "dna_overlord": 0.05,
            "spectral_fusion": 0.05,
            "time_fractal_fft": 0.04,
            "void_renderer": 0.05,
            "meta_routing": 0.05,
        }

        self.confidence_threshold = 0.7
        self.min_gate_score = 0.6

        self.confidence_field_map = {
            "future_shadow": "confidence",
            "black_swan": "black_swan_risk",
            "market_thought": "confidence",
            "reality_matrix": "confidence",
        }

        self.direction_field_map = {
            "future_shadow": "future_direction",
            "market_thought": "collective_intent",
            "reality_matrix": "primary_direction",
        }

        self.activated_modules = {}
        self.activated_modules["atlantean_shield"] = AtlanteanShield()

    def _make_dummy_algo(self):
        class Dummy:
            def __init__(self):
                self.Time = _dt.now()
                self.debug_messages = []

            def Debug(self, msg):
                log.debug(msg)
                self.debug_messages.append(msg)

            def Log(self, msg):
                log.info(msg)

        return Dummy()

    def _debug(self, msg: str):
        try:
            if self.algo and hasattr(self.algo, "Debug"):
                self.algo.Debug(msg)
            else:
                log.debug(msg)
        except Exception:
            log.debug(msg)

    def _publish_signal(self, payload: Dict[str, Any]):
        if self._local_event_bus:
            try:
                self._local_event_bus.publish("SIGNAL_GENERATED", payload, source="QMPUltraEngine")
            except Exception:
                pass

    def generate_signal(self, symbol, history_data):
        if not self._validate_history_data(history_data):
            self._debug(f"QMPUltra: Insufficient history data for {symbol}")
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}}
            self._publish_signal({"symbol": symbol, **res})
            return res

        if self.self_destruct.is_isolated(symbol=symbol):
            info = self.self_destruct.get_isolation_info(symbol=symbol)
            reason = "Unknown" if info is None else info.get("reason", "Unknown")
            self._debug(f"QMPUltra: {symbol} is isolated by Self-Destruct Protocol. Reason: {reason}")
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}}
            self._publish_signal({"symbol": symbol, **res})
            return res

        if not self.activated_modules["atlantean_shield"].protect(symbol):
            self._debug(f"QMPUltra: {symbol} - Atlantean Shield activated! No signal generated.")
            res = {"final_signal": None, "confidence": 0.0, "gate_scores": {}}
            self._publish_signal({"symbol": symbol, **res})
            return res

        history_bars = self._convert_history_to_tradebars(history_data["1m"])

        gate_scores = {}
        directions = {}
        module_results = {}

        for module_name, module in self.modules.items():
            if self.self_destruct.is_isolated(module_name=module_name):
                self._debug(f"QMPUltra: Module {module_name} is isolated by Self-Destruct Protocol")
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
                    result = module.check_transfers(self.algo.Time)
                elif module_name == "fed_jet":
                    result = module.check_movements({})
                elif module_name == "spoofing":
                    result = module.detect(symbol, history_data)
                elif module_name == "stress":
                    result = module.detect(symbol, self.algo.Time)
                elif module_name == "port_activity":
                    result = module.analyze({})
                elif module_name == "dna_breath":
                    result = module.calculate_risk("neutral", 0.05)
                elif module_name == "dna_overlord":
                    result = module.select_hierarchy()
                elif module_name == "spectral_fusion":

                    class MockComponents:
                        def __init__(self):
                            self.emotion = 0.5
                            self.volatility = 0.3
                            self.volume = 0.2
                            self.entropy = 0.4

                    result = module.fuse_signals("crypto", MockComponents())
                elif module_name == "void_renderer":
                    result = {"void_signals": [], "confidence": 0.5, "direction": "NEUTRAL"}
                elif module_name == "meta_routing":
                    mock_signal = {"direction": "BUY", "confidence": 0.7}
                    mock_market_data = {"entropy": 0.5, "liquidity": 0.8}
                    result = module.route_trading_signal(mock_signal, mock_market_data, {})
                elif module_name == "time_fractal_fft":
                    closes = (
                        history_data["1m"]["close"].values
                        if "close" in history_data["1m"].columns
                        else np.random.randn(100)
                    )
                    result = module.predict_price_direction(closes)
                else:
                    result = module.decode(symbol, history_bars)
            except Exception as exc:
                log.debug("Module %s failed: %s", module_name, exc)
                result = {"confidence": 0.0, "direction": "NEUTRAL"}

            module_results[module_name] = result
            gate_scores[module_name] = self._extract_confidence(module_name, result)
            direction = self._extract_direction(module_name, result)
            if direction:
                directions[module_name] = direction

        if "black_swan" in gate_scores:
            gate_scores["black_swan"] = 1.0 - gate_scores["black_swan"]

        confidence = sum(gate_scores[key] * self.module_weights.get(key, 0) for key in gate_scores.keys())

        self.gate_scores = gate_scores
        self.last_confidence = confidence

        gates_pass = all(score >= self.min_gate_score for score in gate_scores.values())

        black_swan_active = False
        if "black_swan" in module_results and isinstance(module_results["black_swan"], dict):
            black_swan_active = module_results["black_swan"].get("protection_active", False)

        if black_swan_active:
            self._debug(f"QMPUltra: {symbol} - Black Swan protection active! No signal generated.")
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores}
            self._publish_signal({"symbol": symbol, **res})
            return res

        compliance_result = self.compliance.pre_trade_check(symbol)
        if not compliance_result.get("compliant", False):
            reason = ", ".join(compliance_result.get("issues", ["Unknown compliance issue"]))
            self._debug(f"QMPUltra: {symbol} - Compliance check failed. Reason: {reason}")
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores}
            self._publish_signal({"symbol": symbol, **res})
            return res

        if gates_pass and confidence >= self.confidence_threshold:
            direction_votes = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}

            for module, direction in directions.items():
                if direction in direction_votes:
                    direction_votes[direction] += self.module_weights.get(module, 0.1)

            final_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])

            if final_direction == "NEUTRAL" or direction_votes[final_direction] < 0.5:
                self._debug(f"QMPUltra: {symbol} - No clear direction consensus. Confidence: {confidence:.2f}")
                res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores}
                self._publish_signal({"symbol": symbol, **res})
                return res

            if "meta_adaptive" in module_results and isinstance(module_results["meta_adaptive"], dict):
                meta_confidence = module_results["meta_adaptive"].get("confidence", 0.0)
                meta_direction = module_results["meta_adaptive"].get("direction", None)

                if meta_direction and meta_direction != final_direction and meta_confidence > 0.8:
                    self._debug(f"QMPUltra: Meta-Adaptive AI override! Changed {final_direction} to {meta_direction}")
                    final_direction = meta_direction

            self._debug(f"QMPUltra: {symbol} - Signal: {final_direction}, Confidence: {confidence:.2f}")
            self.last_signal = final_direction
            self.last_signal_time = self.algo.Time

            self._log_gate_details(symbol, gate_scores, final_direction)

            res = {"final_signal": final_direction, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol, "votes": direction_votes, "weighted_confidence": confidence}
            self._publish_signal(res)
            return res
        else:
            self._debug(f"QMPUltra: {symbol} - No signal, Confidence: {confidence:.2f}")
            res = {"final_signal": None, "confidence": confidence, "gate_scores": gate_scores, "symbol": symbol}
            self._publish_signal(res)
            return res

    def _extract_confidence(self, module_name, result):
        if not isinstance(result, dict):
            return 1.0 if result and result != "WAIT" else 0.0

        confidence_field = self.confidence_field_map.get(module_name, "confidence")

        if confidence_field in result:
            try:
                return float(result[confidence_field])
            except Exception:
                return 0.5
        elif "confidence" in result:
            try:
                return float(result["confidence"])
            except Exception:
                return 0.5
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

    def record_feedback(self, symbol, gate_scores, result, trade_data=None):
        entry = gate_scores.copy()
        entry["result"] = result
        entry["symbol"] = str(symbol)

        if trade_data:
            for key, value in trade_data.items():
                if key not in entry and not isinstance(value, (dict, list)):
                    entry[key] = value

        self.history.append(entry)

        if len(self.history) >= 10:
            df = pd.DataFrame(self.history)
            self._train_model(df)

    def _validate_history_data(self, history_data):
        required_timeframes = ["1m", "5m", "10m", "15m", "20m", "25m"]

        if not all(tf in history_data for tf in required_timeframes):
            return False

        if len(history_data["1m"]) < 60:
            return False

        return True

    def _convert_history_to_tradebars(self, df):
        if df.empty:
            return []

        trade_bars = []
        trade_bars_append = trade_bars.append

        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            self._debug("Missing required columns in history data")
            return []

        for idx, row in df.iterrows():
            bar = TradeBar()
            bar.Open = row["Open"]
            bar.High = row["High"]
            bar.Low = row["Low"]
            bar.Close = row["Close"]
            bar.Volume = row["Volume"] if "Volume" in row else 0
            bar.EndTime = idx
            trade_bars_append(bar)

        return trade_bars

    def _log_gate_details(self, symbol, gate_scores, direction):
        log_msg = f"QMPUltra Gate Details for {symbol} - Direction: {direction}\n"
        for gate, score in gate_scores.items():
            log_msg += f"  {gate}: {score:.2f}\n"
        self._debug(log_msg)

    def _train_model(self, df):
        try:
            win_rate = df["result"].mean() * 100
        except Exception:
            win_rate = 0.0
        self._debug(f"QMPUltra: Training model with {len(df)} samples. Win rate: {win_rate:.1f}%")

        correlations = {}
        for col in df.columns:
            if col != "result":
                try:
                    correlations[col] = df[col].corr(df["result"])
                except Exception:
                    correlations[col] = 0.0

        self._debug(f"QMPUltra: Gate correlations with success: {correlations}")

    def generate_new_strategy(self, market_state):
        if "quantum_code_generator" in self.modules:
            return self.modules["quantum_code_generator"].generate_new_strategy(market_state)
        return {"error": "quantum_code_generator not loaded"}

    def rewrite_history(self):
        if "anti_stuck" in self.modules:
            self.modules["anti_stuck"].rewrite_history()

    def summon_alternative_reality(self):
        if "anti_stuck" in self.modules:
            self.modules["anti_stuck"].summon_alternative_reality()


class TradeBar:
    def __init__(self):
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0
        self._end_time = None

    @property
    def Open(self):
        return self._open

    @Open.setter
    def Open(self, value):
        self._open = float(value)

    @property
    def High(self):
        return self._high

    @High.setter
    def High(self, value):
        self._high = float(value)

    @property
    def Low(self):
        return self._low

    @Low.setter
    def Low(self, value):
        self._low = float(value)

    @property
    def Close(self):
        return self._close

    @Close.setter
    def Close(self, value):
        self._close = float(value)

    @property
    def Volume(self):
        return self._volume

    @Volume.setter
    def Volume(self, value):
        self._volume = float(value)

    @property
    def EndTime(self):
        return self._end_time

    @EndTime.setter
    def EndTime(self, value):
        self._end_time = value
