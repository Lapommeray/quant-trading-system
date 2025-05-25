
# oversoul_director.py
# The OverSoul Layer – Supreme Sentient Director for QMP Overrider System

class OverSoulDirector:
    def __init__(self, algorithm):
        self.algo = algorithm
        self.enabled_modules = {
            'emotion_dna': True,
            'fractal_resonance': True,
            'intention_decoder': True,
            'timeline_fork': True,
            'astro_sync': True,
            'black_swan_protector': True,
            'quantum_tremor': True,
            'future_shadow': True,
            'market_thought': True,
            'reality_matrix': True,
            
            'human_lag': True,
            'invisible_data': True,
            'meta_adaptive': True,
            'quantum_sentiment': True,
            'imperceptible_pattern': True,  # New module for detecting patterns humans can't see
            'market_glitch_detector': True,  # New module for detecting market glitches
            'advanced_noise_filter': True,  # New module for removing data noise
            
            'btc_offchain': True,
            'fed_jet': True,
            'spoofing': True,
            'stress': True,
            'port_activity': True
        }
        
        self.module_tiers = {
            'black_swan_protector': 3,  # Highest priority - safety
            'compliance': 3,            # Regulatory compliance
            
            'future_shadow': 2,         # High priority - predictive
            'meta_adaptive': 2,
            'quantum_tremor': 2,
            'imperceptible_pattern': 2, # High priority - human-imperceptible patterns
            'market_glitch_detector': 2, # High priority - glitch detection
            'advanced_noise_filter': 3, # Highest priority - data quality
            
            'emotion_dna': 1,           # Standard priority
            'fractal_resonance': 1,
            'intention_decoder': 1,
            'timeline_fork': 1,
            'astro_sync': 1,
            'market_thought': 1,
            'reality_matrix': 1,
            'human_lag': 1,
            'invisible_data': 1,
            'quantum_sentiment': 1,
            
            'btc_offchain': 1,
            'fed_jet': 1,
            'spoofing': 1,
            'stress': 1,
            'port_activity': 1
        }

    def evaluate_state(self, gate_results, user_state=None, environment_state=None, monitoring_results=None):
        """
        Accepts outputs from modules and environment awareness, then decides
        which modules to amplify, ignore, or delay.

        :param gate_results: Dictionary of current gate output (True/False or scores)
        :param user_state: Optional dict of user's clarity, fatigue, focus (if connected)
        :param environment_state: Optional dict of market calm/chaos, time-of-day, etc.
        :param monitoring_results: Optional dict of results from monitoring tools
        :return: Action recommendation or module adjustments
        """

        if not gate_results:
            return {'action': 'WAIT', 'reason': 'No valid input'}

        signals = []
        diagnostics = []
        
        btc_offchain_alert = False
        fed_movement_alert = False
        spoofing_alert = False
        stress_alert = False
        port_disruption_alert = False
        event_probability_alert = False
        
        if monitoring_results:
            if 'btc_offchain' in monitoring_results and monitoring_results['btc_offchain']:
                btc_result = monitoring_results['btc_offchain']
                if btc_result.get('large_transfer_detected', False):
                    btc_offchain_alert = True
                    diagnostics.append(f"BTC off-chain alert: {btc_result.get('transfer_amount', 0)} BTC moved")
                    
            if 'fed_jet' in monitoring_results and monitoring_results['fed_jet']:
                fed_result = monitoring_results['fed_jet']
                if fed_result.get('unusual_movement_detected', False):
                    fed_movement_alert = True
                    officials = ', '.join(fed_result.get('officials', []))
                    diagnostics.append(f"Fed movement alert: {officials}")
                    
            if 'spoofing' in monitoring_results and monitoring_results['spoofing']:
                spoofing_result = monitoring_results['spoofing']
                if spoofing_result.get('spoofing_detected', False):
                    spoofing_alert = True
                    diagnostics.append(f"Spoofing alert: Volume ratio {spoofing_result.get('volume_ratio', 0):.2f}")
                    
            if 'stress' in monitoring_results and monitoring_results['stress']:
                stress_result = monitoring_results['stress']
                if stress_result.get('extreme_stress', False):
                    stress_alert = True
                    diagnostics.append(f"Extreme stress detected in earnings call: {stress_result.get('stress_score', 0):.2f}")
                    
            if 'port_activity' in monitoring_results and monitoring_results['port_activity']:
                port_result = monitoring_results['port_activity']
                if port_result.get('disruption_detected', False):
                    port_disruption_alert = True
                    diagnostics.append("Supply chain disruption detected in port activity")
                    
            if 'event_probability' in monitoring_results and monitoring_results['event_probability']:
                event_result = monitoring_results['event_probability']
                
                if 'probabilities' in event_result:
                    probabilities = event_result['probabilities']
                    highest_event = event_result.get('highest_probability_event', '')
                    highest_value = event_result.get('highest_probability_value', 0.0)
                    
                    if highest_value >= 50.0:  # 50% threshold for alert
                        event_probability_alert = True
                        diagnostics.append(f"Event Probability Alert: {highest_event} ({highest_value:.1f}%)")
                        
                        for event_type, probability in probabilities.items():
                            if probability >= 30.0:  # Log events with at least 30% probability
                                diagnostics.append(f"  - {event_type}: {probability:.1f}%")
                
                if 'decisions' in event_result:
                    decisions = event_result['decisions']
                    
                    if decisions.get('disable_aggressive_mode', False):
                        diagnostics.append("Event Probability: Disabling aggressive mode")
                        self.enabled_modules['human_lag'] = False
                        self.enabled_modules['meta_adaptive'] = False
                    
                    if decisions.get('enable_capital_defense', False):
                        diagnostics.append("Event Probability: Enabling capital defense")
                        self.enabled_modules['black_swan_protector'] = True
                    
                    if decisions.get('pause_new_entries', False):
                        diagnostics.append("Event Probability: Recommending pause on new entries")
                        return {'action': 'HOLD', 'reason': 'High probability event detected', 'diagnostics': diagnostics}

        # Sample judgment: suppress gates during distorted emotional cycles
        if gate_results.get('emotion_dna') is False and environment_state and environment_state.get('noise_level') == 'high':
            diagnostics.append("Suppressing Emotion Gate due to high noise.")
            self.enabled_modules['emotion_dna'] = False

        # Prioritize timeline if market is calm
        if environment_state and environment_state.get('market_mode') == 'coiled':
            diagnostics.append("Market in compression mode — boosting timeline fork.")
            self.enabled_modules['timeline_fork'] = True
            
        alert_count = sum([btc_offchain_alert, fed_movement_alert, spoofing_alert, stress_alert, port_disruption_alert])
        if alert_count >= 2:
            diagnostics.append(f"Multiple alerts detected ({alert_count}) — activating Black Swan Protector")
            self.enabled_modules['black_swan_protector'] = True
            
        if environment_state and environment_state.get('fomc_week', False):
            diagnostics.append("FOMC week — activating Fed Jet Monitor")
            self.enabled_modules['fed_jet'] = True
            
        if environment_state and environment_state.get('earnings_season', False):
            diagnostics.append("Earnings season — activating Stress Detector")
            self.enabled_modules['stress'] = True
            
        if environment_state and environment_state.get('volatility', 'normal') == 'high':
            diagnostics.append("High volatility — activating Spoofing Detector")
            self.enabled_modules['spoofing'] = True

        # If user is flagged tired (optional user input from future integration)
        if user_state and user_state.get('clarity') == 'low':
            diagnostics.append("User clarity low — hold entries.")
            return {'action': 'HOLD', 'reason': 'Low user clarity'}
            
        if monitoring_results and 'compliance' in monitoring_results and monitoring_results['compliance']:
            compliance_result = monitoring_results['compliance']
            if not compliance_result.get('approved', True):
                reason = compliance_result.get('reason', 'Unknown compliance issue')
                diagnostics.append(f"Compliance check failed: {reason}")
                return {'action': 'ABORT', 'reason': reason, 'diagnostics': diagnostics}

        # If everything is aligned
        if all(gate_results.values()):
            diagnostics.append("All gates passed.")
            return {'action': 'EXECUTE', 'modules': self.enabled_modules, 'diagnostics': diagnostics}

        return {'action': 'WAIT', 'modules': self.enabled_modules, 'diagnostics': diagnostics}
