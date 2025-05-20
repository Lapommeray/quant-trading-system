
from core.qmp_engine_v3 import QMPUltraEngine
from core.oversoul.oversoul_director import OverSoulDirector

class QMPOversoulEngine:
    """
    Higher-order intelligence layer that manages the QMPUltraEngine
    and provides meta-control over which modules are active based on
    market conditions and other environmental factors.
    """
    def __init__(self, algorithm):
        self.algo = algorithm
        self.ultra_engine = QMPUltraEngine(algorithm)
        self.oversoul = OverSoulDirector(algorithm)
        
        self.environment_state = {
            'market_mode': 'normal',  # normal, volatile, coiled, trending
            'noise_level': 'medium',  # low, medium, high
            'liquidity': 'normal',    # low, normal, high
            'sentiment': 'neutral'    # bullish, bearish, neutral
        }
        
        self.user_state = {
            'clarity': 'high',        # low, medium, high
            'fatigue': 'low',         # low, medium, high
            'focus': 'high'           # low, medium, high
        }
        
        self.signals_generated = 0
        self.signals_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
    def generate_signal(self, symbol, history_data):
        """
        Generate trading signal with oversoul intelligence layer
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        
        Returns:
        - direction: "BUY", "SELL", or None
        - confidence: Signal confidence score (0.0-1.0)
        - gate_scores: Dictionary of individual gate scores
        - diagnostics: List of diagnostic messages from the oversoul
        """
        self._update_environment_state(symbol, history_data)
        
        direction, confidence, gate_scores = self.ultra_engine.generate_signal(symbol, history_data)
        
        gate_results = {k: (v >= self.ultra_engine.min_gate_score) for k, v in gate_scores.items()}
        
        oversoul_decision = self.oversoul.evaluate_state(
            gate_results, 
            self.user_state, 
            self.environment_state
        )
        
        if 'diagnostics' in oversoul_decision:
            for msg in oversoul_decision['diagnostics']:
                self.algo.Debug(f"OverSoul: {msg}")
        
        if 'modules' in oversoul_decision:
            self._update_module_activation(oversoul_decision['modules'])
        
        if oversoul_decision['action'] == 'EXECUTE':
            self.signals_executed += 1
            return direction, confidence, gate_scores, oversoul_decision.get('diagnostics', [])
        else:
            reason = oversoul_decision.get('reason', 'OverSoul override')
            self.algo.Debug(f"OverSoul override: {reason}")
            return None, confidence, gate_scores, oversoul_decision.get('diagnostics', [])
    
    def _update_environment_state(self, symbol, history_data):
        """Update environment state based on market conditions"""
        if not history_data or '1m' not in history_data or history_data['1m'].empty:
            return
            
        recent_data = history_data['1m'].tail(30)
        
        if len(recent_data) >= 20:
            volatility = recent_data['Close'].pct_change().std() * 100
            
            if volatility > 1.5:
                self.environment_state['market_mode'] = 'volatile'
                self.environment_state['noise_level'] = 'high'
            elif volatility < 0.3:
                self.environment_state['market_mode'] = 'coiled'
                self.environment_state['noise_level'] = 'low'
            else:
                self.environment_state['market_mode'] = 'normal'
                self.environment_state['noise_level'] = 'medium'
        
        if len(recent_data) >= 20:
            close_prices = recent_data['Close'].values
            first_half = close_prices[:10].mean()
            second_half = close_prices[-10:].mean()
            
            if second_half > first_half * 1.01:
                self.environment_state['sentiment'] = 'bullish'
            elif second_half < first_half * 0.99:
                self.environment_state['sentiment'] = 'bearish'
            else:
                self.environment_state['sentiment'] = 'neutral'
    
    def _update_module_activation(self, module_states):
        """Update which modules are active based on oversoul decision"""
        module_map = {
            'emotion_dna': 'emotion_dna',
            'fractal_resonance': 'fractal_resonance',
            'intention_decoder': 'intention',
            'timeline_fork': 'future_shadow',
            'astro_sync': 'astro_geo',
            'black_swan_protector': 'black_swan'
        }
        
        for oversoul_name, is_active in module_states.items():
            if oversoul_name in module_map:
                ultra_name = module_map[oversoul_name]
                if ultra_name in self.ultra_engine.module_weights:
                    if not is_active:
                        self.ultra_engine.module_weights[ultra_name] = 0.0
                        self.algo.Debug(f"OverSoul disabled module: {ultra_name}")
    
    def record_feedback(self, gate_scores, result):
        """Record trade result for AI learning"""
        if result == 1:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
            
        self.ultra_engine.record_feedback(gate_scores, result)
