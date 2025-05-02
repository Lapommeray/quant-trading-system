
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
            'black_swan_protector': True
        }

    def evaluate_state(self, gate_results, user_state=None, environment_state=None):
        """
        Accepts outputs from modules and environment awareness, then decides
        which modules to amplify, ignore, or delay.

        :param gate_results: Dictionary of current gate output (True/False or scores)
        :param user_state: Optional dict of user's clarity, fatigue, focus (if connected)
        :param environment_state: Optional dict of market calm/chaos, time-of-day, etc.
        :return: Action recommendation or module adjustments
        """

        if not gate_results:
            return {'action': 'WAIT', 'reason': 'No valid input'}

        signals = []
        diagnostics = []

        # Sample judgment: suppress gates during distorted emotional cycles
        if gate_results.get('emotion_dna') is False and environment_state and environment_state.get('noise_level') == 'high':
            diagnostics.append("Suppressing Emotion Gate due to high noise.")
            self.enabled_modules['emotion_dna'] = False

        # Prioritize timeline if market is calm
        if environment_state and environment_state.get('market_mode') == 'coiled':
            diagnostics.append("Market in compression mode — boosting timeline fork.")
            self.enabled_modules['timeline_fork'] = True

        # If user is flagged tired (optional user input from future integration)
        if user_state and user_state.get('clarity') == 'low':
            diagnostics.append("User clarity low — hold entries.")
            return {'action': 'HOLD', 'reason': 'Low user clarity'}

        # If everything is aligned
        if all(gate_results.values()):
            diagnostics.append("All gates passed.")
            return {'action': 'EXECUTE', 'modules': self.enabled_modules, 'diagnostics': diagnostics}

        return {'action': 'WAIT', 'modules': self.enabled_modules, 'diagnostics': diagnostics}
