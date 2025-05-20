
from core.qmp_ai import QMPAIAgent
import pandas as pd
from modules.alien_decoder import decode_alien_numerical_patterns
from modules.cosmic_channeler import get_cosmic_alignment_score
from modules.zero_point import is_zero_point_state
from modules.angel_decoder import get_angelic_alignment_score
from modules.energy_filter import is_global_shock_active, is_cosmic_energy_disturbed, is_today_a_sacred_entry_day
from modules.big_move_detector import detect_big_move_setup
from modules.divine_sync import sacred_369_alignment, check_divine_timing
from modules.timeline_selector import evaluate_possible_timelines
from modules.multiverse_sync import confirm_multiverse_alignment
from data.data_fetcher import get_latest_price
from data.macro_fetcher import is_macro_environment_favorable

class QMPOverrider:
    def __init__(self):
        self.ai = QMPAIAgent()
        self.history = []
        self.gate_scores = {}
        self.last_confidence = 0.0

    def generate_signal(self, timestamp):
        price_data = get_latest_price()
        
        gate_scores = {
            'alien': decode_alien_numerical_patterns(price_data),
            'cosmic': get_cosmic_alignment_score(timestamp),
            'quantum': 1.0 if is_zero_point_state(price_data) else 0.0,
            'emotion': 0.0 if is_global_shock_active() else 1.0,
            'angelic': get_angelic_alignment_score(price_data),
            'divine_timing': 1.0 if check_divine_timing(timestamp) else 0.0,
            'sacred_date': 1.0 if is_today_a_sacred_entry_day() else 0.0
        }
        
        big_move = detect_big_move_setup(price_data)
        gate_scores['big_move'] = 1.0 if big_move['compression_detected'] else 0.0
        
        timelines = evaluate_possible_timelines(price_data)
        gate_scores['timeline'] = 1.0 if timelines['winning_timeline'] else 0.0
        
        is_bullish = big_move['big_buy_imminent'] or timelines['bullish_timeline_dominance'] > timelines['bearish_timeline_dominance']
        
        passed_all_gates = all([
            gate_scores['alien'] > 0.85,
            gate_scores['cosmic'] > 0.9,
            gate_scores['quantum'] > 0.92,
            gate_scores['emotion'] > 0.7,
            gate_scores['angelic'] > 0.5,
            gate_scores['divine_timing'] > 0.5,
            gate_scores['sacred_date'] > 0.0,
            gate_scores['big_move'] > 0.5,
            gate_scores['timeline'] > 0.5
        ])
        
        macro_favorable = is_macro_environment_favorable()
        gate_scores['macro'] = 1.0 if macro_favorable else 0.0
        
        self.gate_scores = gate_scores
        
        # Let AI decide if the gates are worth trusting
        if not self.ai.predict_gate_pass(gate_scores):
            return None, None
        
        confidence = sum(gate_scores.values()) / len(gate_scores)
        self.last_confidence = confidence
        
        direction = "BUY" if is_bullish else "SELL"
        
        return direction, confidence

    def record_feedback(self, gate_scores, result):
        entry = gate_scores.copy()
        entry['result'] = result  # 1 = profit, 0 = loss
        self.history.append(entry)
        df = pd.DataFrame(self.history)
        self.ai.train(df)
