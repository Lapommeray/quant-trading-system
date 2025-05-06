from AlgorithmImports import *
import pandas as pd
import numpy as np

from ultra_modules.emotion_dna_decoder import EmotionDNADecoder
from ultra_modules.fractal_resonance_gate import FractalResonanceGate
from ultra_modules.quantum_tremor_scanner import QuantumTremorScanner
from ultra_modules.intention_decoder import IntentionDecoder
from ultra_modules.sacred_event_alignment import SacredEventAlignment
from ultra_modules.astro_geo_sync import AstroGeoSync
from ultra_modules.future_shadow_decoder import FutureShadowDecoder
from ultra_modules.black_swan_protector import BlackSwanProtector
from ultra_modules.market_thought_form_interpreter import MarketThoughtFormInterpreter
from ultra_modules.reality_displacement_matrix import RealityDisplacementMatrix

from advanced_modules.human_lag_exploit import HumanLagExploit
from advanced_modules.invisible_data_miner import InvisibleDataMiner
from advanced_modules.meta_adaptive_ai import MetaAdaptiveAI
from advanced_modules.self_destruct_protocol import SelfDestructProtocol
from advanced_modules.quantum_sentiment_decoder import QuantumSentimentDecoder
from advanced_modules.btc_offchain_monitor import BTCOffchainMonitor
from advanced_modules.fed_jet_monitor import FedJetMonitor
from advanced_modules.spoofing_detector import SpoofingDetector
from advanced_modules.compliance_check import ComplianceCheck
from advanced_modules.stress_detector import StressDetector
from advanced_modules.port_activity_analyzer import PortActivityAnalyzer
from advanced_modules.dna_breath import DNABreath
from advanced_modules.dna_overlord import DNAOverlord
from advanced_modules.spectral_signal_fusion import SpectralSignalFusion
from advanced_modules.void_trader_chart_renderer import VoidTraderChartRenderer
from core.meta_conscious_routing_layer import MetaConsciousRoutingLayer

class QMPUltraEngine:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.history = []
        self.gate_scores = {}
        self.last_confidence = 0.0
        self.last_signal = None
        self.last_signal_time = None
        
        self.modules = {
            'emotion_dna': EmotionDNADecoder(algorithm),
            'fractal_resonance': FractalResonanceGate(algorithm),
            'quantum_tremor': QuantumTremorScanner(algorithm),
            'intention': IntentionDecoder(algorithm),
            'sacred_event': SacredEventAlignment(algorithm),
            'astro_geo': AstroGeoSync(algorithm),
            'future_shadow': FutureShadowDecoder(algorithm),
            'black_swan': BlackSwanProtector(algorithm),
            'market_thought': MarketThoughtFormInterpreter(algorithm),
            'reality_matrix': RealityDisplacementMatrix(algorithm),
            
            'human_lag': HumanLagExploit(algorithm),
            'invisible_data': InvisibleDataMiner(algorithm),
            'meta_adaptive': MetaAdaptiveAI(algorithm),
            'quantum_sentiment': QuantumSentimentDecoder(algorithm),
            
            'btc_offchain': BTCOffchainMonitor(algorithm),
            'fed_jet': FedJetMonitor(algorithm),
            'spoofing': SpoofingDetector(algorithm),
            'stress': StressDetector(algorithm),
            'port_activity': PortActivityAnalyzer(algorithm),
            
            'dna_breath': DNABreath(),
            'dna_overlord': DNAOverlord(),
            'spectral_fusion': SpectralSignalFusion(),
            'void_renderer': VoidTraderChartRenderer(),
            'meta_routing': MetaConsciousRoutingLayer()
        }
        
        self.compliance = ComplianceCheck(algorithm)
        self.self_destruct = SelfDestructProtocol(algorithm)
        
        self.module_weights = {
            'emotion_dna': 0.06,
            'fractal_resonance': 0.06,
            'quantum_tremor': 0.06,
            'intention': 0.08,
            'sacred_event': 0.03,
            'astro_geo': 0.03,
            'future_shadow': 0.08,
            'black_swan': 0.06,
            'market_thought': 0.06,
            'reality_matrix': 0.06,
            
            'human_lag': 0.06,
            'invisible_data': 0.06,
            'meta_adaptive': 0.06,
            'quantum_sentiment': 0.06,
            
            'btc_offchain': 0.04,
            'fed_jet': 0.04,
            'spoofing': 0.04,
            'stress': 0.04,
            'port_activity': 0.04,
            
            'dna_breath': 0.05,
            'dna_overlord': 0.05,
            'spectral_fusion': 0.05,
            'void_renderer': 0.05,
            'meta_routing': 0.05
        }
        
        self.confidence_threshold = 0.7
        self.min_gate_score = 0.6
        
        self.confidence_field_map = {
            'future_shadow': 'confidence',
            'black_swan': 'black_swan_risk',  # Will be inverted
            'market_thought': 'confidence',
            'reality_matrix': 'confidence'
        }
        
        self.direction_field_map = {
            'future_shadow': 'future_direction',
            'market_thought': 'collective_intent',
            'reality_matrix': 'primary_direction'
        }
        
    def generate_signal(self, symbol, history_data):
        """
        Generate trading signal based on all ultra and advanced modules
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - direction: "BUY", "SELL", or None
        - confidence: Signal confidence score (0.0-1.0)
        - gate_scores: Dictionary of individual gate scores
        """
        if not self._validate_history_data(history_data):
            self.algo.Debug(f"QMPUltra: Insufficient history data for {symbol}")
            return None, 0.0, {}
        
        if self.self_destruct.is_isolated(symbol=symbol):
            isolation_info = self.self_destruct.get_isolation_info(symbol=symbol)
            reason = "Unknown" if isolation_info is None else isolation_info.get('reason', 'Unknown')
            self.algo.Debug(f"QMPUltra: {symbol} is isolated by Self-Destruct Protocol. Reason: {reason}")
            return None, 0.0, {}
        
        history_bars = self._convert_history_to_tradebars(history_data['1m'])
        
        gate_scores = {}
        directions = {}
        module_results = {}
        
        for module_name, module in self.modules.items():
            if self.self_destruct.is_isolated(module_name=module_name):
                self.algo.Debug(f"QMPUltra: Module {module_name} is isolated by Self-Destruct Protocol")
                continue
                
            if module_name == 'human_lag':
                result = module.detect(symbol, history_data)
            elif module_name == 'invisible_data':
                result = module.extract_patterns(symbol, history_data)
            elif module_name == 'meta_adaptive':
                result = module.predict(symbol, history_data)
            elif module_name == 'quantum_sentiment':
                result = module.decode(symbol, history_data)
            elif module_name == 'btc_offchain':
                result = module.check_transfers(self.algo.Time)
            elif module_name == 'fed_jet':
                result = module.check_movements(self.algo.Time)
            elif module_name == 'spoofing':
                result = module.detect(symbol, history_data)
            elif module_name == 'stress':
                result = module.detect(symbol, self.algo.Time)
            elif module_name == 'port_activity':
                result = module.analyze(self.algo.Time)
            elif module_name == 'dna_breath':
                result = module.calculate_risk('neutral', 0.05)  # Example usage
            elif module_name == 'dna_overlord':
                result = module.select_hierarchy()
            elif module_name == 'spectral_fusion':
                result = module.fuse_signals('crypto', SpectralComponents(0.5, 0.3, 0.2))  # Example usage
            elif module_name == 'void_renderer':
                result = module.render_chart(symbol, history_data['1m'])
            elif module_name == 'meta_routing':
                result = module.route_path('crypto', 0.5, 0.8)  # Example usage
            else:
                result = module.decode(symbol, history_bars)
                
            module_results[module_name] = result
            
            gate_scores[module_name] = self._extract_confidence(module_name, result)
            
            direction = self._extract_direction(module_name, result)
            if direction:
                directions[module_name] = direction
        
        if 'black_swan' in gate_scores:
            gate_scores['black_swan'] = 1.0 - gate_scores['black_swan']
        
        confidence = sum(gate_scores[key] * self.module_weights[key] 
                         for key in gate_scores.keys())
        
        self.gate_scores = gate_scores
        self.last_confidence = confidence
        
        gates_pass = all(score >= self.min_gate_score for score in gate_scores.values())
        
        black_swan_active = False
        if 'black_swan' in module_results and isinstance(module_results['black_swan'], dict):
            black_swan_active = module_results['black_swan'].get('protection_active', False)
        
        if black_swan_active:
            self.algo.Debug(f"QMPUltra: {symbol} - Black Swan protection active! No signal generated.")
            return None, confidence, gate_scores
        
        compliance_result = self.compliance.pre_trade_check(symbol)
        if not compliance_result.get('compliant', False):
            reason = ', '.join(compliance_result.get('issues', ['Unknown compliance issue']))
            self.algo.Debug(f"QMPUltra: {symbol} - Compliance check failed. Reason: {reason}")
            return None, confidence, gate_scores
            
        if gates_pass and confidence >= self.confidence_threshold:
            direction_votes = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}
            
            for module, direction in directions.items():
                if direction in direction_votes:
                    direction_votes[direction] += self.module_weights.get(module, 0.1)
            
            final_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            
            if final_direction == "NEUTRAL" or direction_votes[final_direction] < 0.5:
                self.algo.Debug(f"QMPUltra: {symbol} - No clear direction consensus. Confidence: {confidence:.2f}")
                return None, confidence, gate_scores
            
            if 'meta_adaptive' in module_results and isinstance(module_results['meta_adaptive'], dict):
                meta_confidence = module_results['meta_adaptive'].get('confidence', 0.0)
                meta_direction = module_results['meta_adaptive'].get('direction', None)
                
                if meta_direction and meta_direction != final_direction and meta_confidence > 0.8:
                    self.algo.Debug(f"QMPUltra: Meta-Adaptive AI override! Changed {final_direction} to {meta_direction}")
                    final_direction = meta_direction
            
            self.algo.Debug(f"QMPUltra: {symbol} - Signal: {final_direction}, Confidence: {confidence:.2f}")
            self.last_signal = final_direction
            self.last_signal_time = self.algo.Time
            
            self._log_gate_details(symbol, gate_scores, final_direction)
            
            return final_direction, confidence, gate_scores
        else:
            self.algo.Debug(f"QMPUltra: {symbol} - No signal, Confidence: {confidence:.2f}")
            return None, confidence, gate_scores
    
    def _extract_confidence(self, module_name, result):
        """Extract confidence score from module result with proper handling for different return types"""
        if not isinstance(result, dict):
            return 1.0 if result and result != "WAIT" else 0.0
            
        confidence_field = self.confidence_field_map.get(module_name, 'confidence')
        
        if confidence_field in result:
            return result[confidence_field]
        elif 'confidence' in result:
            return result['confidence']
        else:
            return 0.5  # Default confidence if not found
    
    def _extract_direction(self, module_name, result):
        """Extract direction from module result with proper handling for different return types"""
        if not isinstance(result, dict):
            if isinstance(result, str) and result in ["BUY", "SELL", "NEUTRAL", "WAIT"]:
                return "NEUTRAL" if result == "WAIT" else result
            return None
            
        direction_field = self.direction_field_map.get(module_name, 'direction')
        
        if direction_field in result:
            return result[direction_field]
        elif 'direction' in result:
            return result['direction']
        else:
            return None
    
    def record_feedback(self, symbol, gate_scores, result, trade_data=None):
        """
        Record trade result for AI learning
        
        Parameters:
        - symbol: Trading symbol
        - gate_scores: Dictionary of gate scores
        - result: 1 for profit, 0 for loss
        - trade_data: Optional dictionary with additional trade data
        """
        entry = gate_scores.copy()
        entry['result'] = result
        entry['symbol'] = str(symbol)
        
        if trade_data:
            for key, value in trade_data.items():
                if key not in entry and not isinstance(value, (dict, list)):
                    entry[key] = value
        
        self.history.append(entry)
        
        if len(self.history) >= 10:
            df = pd.DataFrame(self.history)
            self._train_model(df)
    
    def _validate_history_data(self, history_data):
        """Check if we have sufficient history data"""
        required_timeframes = ['1m', '5m', '10m', '15m', '20m', '25m']
        
        if not all(tf in history_data for tf in required_timeframes):
            return False
            
        if len(history_data['1m']) < 60:  # Need at least 60 bars for analysis
            return False
            
        return True
    
    def _convert_history_to_tradebars(self, df):
        """
        Convert pandas DataFrame to TradeBar format for modules
        
        Parameters:
        - df: pandas DataFrame with OHLCV data
        
        Returns:
        - List of TradeBar objects
        """
        if df.empty:
            return []
            
        trade_bars = []
        trade_bars_append = trade_bars.append  # Local reference for faster append
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            self.algo.Debug("Missing required columns in history data")
            return []
        
        for idx, row in df.iterrows():
            bar = TradeBar()
            bar.Open = row['Open']
            bar.High = row['High']
            bar.Low = row['Low']
            bar.Close = row['Close']
            bar.Volume = row['Volume'] if 'Volume' in row else 0
            bar.EndTime = idx
            trade_bars_append(bar)
            
        return trade_bars
    
    def _log_gate_details(self, symbol, gate_scores, direction):
        """Log detailed gate information for analysis"""
        log_msg = f"QMPUltra Gate Details for {symbol} - Direction: {direction}\n"
        for gate, score in gate_scores.items():
            log_msg += f"  {gate}: {score:.2f}\n"
        self.algo.Debug(log_msg)
    
    def _train_model(self, df):
        """Train AI model on historical results"""
        win_rate = df['result'].mean() * 100
        self.algo.Debug(f"QMPUltra: Training model with {len(df)} samples. Win rate: {win_rate:.1f}%")
        
        correlations = {}
        for col in df.columns:
            if col != 'result':
                correlations[col] = df[col].corr(df['result'])
                
        self.algo.Debug(f"QMPUltra: Gate correlations with success: {correlations}")

class TradeBar:
    """
    Simple TradeBar implementation for module compatibility.
    Provides a standardized interface for price data.
    """
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
    def Open(self, value): self._open = float(value)
    
    @property
    def High(self): return self._high
    @High.setter
    def High(self, value): self._high = float(value)
    
    @property
    def Low(self): return self._low
    @Low.setter
    def Low(self, value): self._low = float(value)
    
    @property
    def Close(self): return self._close
    @Close.setter
    def Close(self, value): self._close = float(value)
    
    @property
    def Volume(self): return self._volume
    @Volume.setter
    def Volume(self, value): self._volume = float(value)
    
    @property
    def EndTime(self): return self._end_time
    @EndTime.setter
    def EndTime(self, value): self._end_time = value
