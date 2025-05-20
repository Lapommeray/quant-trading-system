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
            'sentiment': 'neutral',   # bullish, bearish, neutral
            'fomc_week': False,       # True during FOMC meeting weeks
            'earnings_season': False, # True during earnings seasons
            'volatility': 'normal',   # low, normal, high
            'compliance_level': 'strict' # relaxed, normal, strict
        }
        
        self.user_state = {
            'clarity': 'high',        # low, medium, high
            'fatigue': 'low',         # low, medium, high
            'focus': 'high'           # low, medium, high
        }
        
        self.monitoring_results = {}
        
        self.gate_scores = {}
        self.last_confidence = 0.0
        
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
        
        monitoring_results = {}
        
        if 'btc_offchain' in self.ultra_engine.modules:
            monitoring_results['btc_offchain'] = self.ultra_engine.modules['btc_offchain'].check_transfers(self.algo.Time)
            
        if 'fed_jet' in self.ultra_engine.modules:
            monitoring_results['fed_jet'] = self.ultra_engine.modules['fed_jet'].check_movements(self.algo.Time)
            
        if 'spoofing' in self.ultra_engine.modules:
            monitoring_results['spoofing'] = self.ultra_engine.modules['spoofing'].detect(symbol, history_data)
            
        if 'stress' in self.ultra_engine.modules:
            monitoring_results['stress'] = self.ultra_engine.modules['stress'].detect(symbol, self.algo.Time)
            
        if 'port_activity' in self.ultra_engine.modules:
            monitoring_results['port_activity'] = self.ultra_engine.modules['port_activity'].analyze(self.algo.Time)
            
        if hasattr(self.ultra_engine, 'compliance'):
            try:
                monitoring_results['compliance'] = {
                    'approved': True,  # Default to approved
                    'reason': ''
                }
                
                self.monitoring_results = monitoring_results
            except Exception as e:
                self.algo.Debug(f"Error in compliance check: {e}")
        
        oversoul_decision = self.oversoul.evaluate_state(
            gate_results, 
            self.user_state, 
            self.environment_state,
            monitoring_results
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
                self.environment_state['volatility'] = 'high'
            elif volatility < 0.3:
                self.environment_state['market_mode'] = 'coiled'
                self.environment_state['noise_level'] = 'low'
                self.environment_state['volatility'] = 'low'
            else:
                self.environment_state['market_mode'] = 'normal'
                self.environment_state['noise_level'] = 'medium'
                self.environment_state['volatility'] = 'normal'
        
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
        
        current_day = self.algo.Time.day
        current_month = self.algo.Time.month
        
        if 15 <= current_day <= 21:
            self.environment_state['fomc_week'] = True
        else:
            self.environment_state['fomc_week'] = False
            
        earnings_months = [1, 4, 7, 10]
        if current_month in earnings_months:
            self.environment_state['earnings_season'] = True
        else:
            self.environment_state['earnings_season'] = False
    
    def _update_module_activation(self, module_states):
        """Update which modules are active based on oversoul decision"""
        module_map = {
            'emotion_dna': 'emotion_dna',
            'fractal_resonance': 'fractal_resonance',
            'intention_decoder': 'intention',
            'timeline_fork': 'future_shadow',
            'astro_sync': 'astro_geo',
            'black_swan_protector': 'black_swan',
            'quantum_tremor': 'quantum_tremor',
            'market_thought': 'market_thought',
            'reality_matrix': 'reality_matrix',
            
            'human_lag': 'human_lag',
            'invisible_data': 'invisible_data',
            'meta_adaptive': 'meta_adaptive',
            'quantum_sentiment': 'quantum_sentiment',
            
            'btc_offchain': 'btc_offchain',
            'fed_jet': 'fed_jet',
            'spoofing': 'spoofing',
            'stress': 'stress',
            'port_activity': 'port_activity'
        }
        
        for oversoul_name, is_active in module_states.items():
            if oversoul_name in module_map:
                ultra_name = module_map[oversoul_name]
                if ultra_name in self.ultra_engine.module_weights:
                    if not is_active:
                        self.ultra_engine.module_weights[ultra_name] = 0.0
                        self.algo.Debug(f"OverSoul disabled module: {ultra_name}")
                    else:
                        if self.ultra_engine.module_weights[ultra_name] == 0.0:
                            default_weights = {
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
                                
                                'btc_offchain': 0.06,
                                'fed_jet': 0.06,
                                'spoofing': 0.06,
                                'stress': 0.06,
                                'port_activity': 0.06
                            }
                            self.ultra_engine.module_weights[ultra_name] = default_weights.get(ultra_name, 0.08)
                            self.algo.Debug(f"OverSoul re-enabled module: {ultra_name}")
    
    def record_feedback(self, symbol, gate_scores, result, trade_data=None):
        """
        Record trade result for AI learning and Self-Destruct Protocol
        
        Parameters:
        - symbol: Trading symbol
        - gate_scores: Dictionary of gate scores
        - result: 1 for profit, 0 for loss
        - trade_data: Optional dictionary with additional trade data
        """
        if result == 1:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
        
        self.ultra_engine.record_feedback(symbol, gate_scores, result)
        
        if trade_data is None:
            trade_data = {}
            
        trade_data.update({
            'symbol': str(symbol),
            'timestamp': self.algo.Time,
            'result': result,
            'gate_scores': gate_scores
        })
        
        if hasattr(self.ultra_engine, 'self_destruct'):
            if hasattr(self.ultra_engine.self_destruct, 'record_trade_result'):
                self.ultra_engine.self_destruct.record_trade_result(
                    symbol=symbol,
                    result=result,
                    trade_data=trade_data
                )
            
            if hasattr(self.ultra_engine.self_destruct, 'check_isolation_criteria'):
                isolation_updates = self.ultra_engine.self_destruct.check_isolation_criteria()
                
                if isolation_updates:
                    for update in isolation_updates:
                        if update['action'] == 'isolate':
                            self.algo.Debug(f"Self-Destruct Protocol: Isolating {update['target']} due to {update['reason']}")
                        elif update['action'] == 'recover':
                            self.algo.Debug(f"Self-Destruct Protocol: Recovering {update['target']} from isolation")
