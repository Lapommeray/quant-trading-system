# File: tests/test_event_probability.py

def test_traceback_logging(self):
    """Verify complete traceback logging"""
    epm = EventProbabilityModule()
    with patch.object(epm.encryption_engine, 'encrypt', side_effect=Exception("Test error")):
        with self.assertLogs('EventProbabilityModule', level='ERROR') as cm:
            epm.update_indicators({"Test": 0.5})

    log_output = '\n'.join(cm.output)
    self.assertIn('Traceback', log_output)
    self.assertIn('Test error', log_output)
