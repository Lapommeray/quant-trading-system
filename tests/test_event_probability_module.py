import unittest
from unittest.mock import patch
from advanced_modules.event_probability_module import EventProbabilityModule

class TestEventProbabilityModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = {
            "FOMC": 0.92,
            "EarningsSurprise": 0.71,
            "Invalid1": 1.5,  # Should fail
            "Invalid2": -0.1  # Should fail
        }

    def test_normal_operation(self):
        """Test successful encryption updates"""
        epm = EventProbabilityModule()
        result = epm.update_indicators({
            "FOMC": 0.92,
            "EarningsSurprise": 0.71
        })
        self.assertTrue(result)
        self.assertEqual(len(epm.indicators), 2)

    def test_invalid_inputs(self):
        """Test invalid probability values"""
        epm = EventProbabilityModule()
        with self.assertLogs('EventProbabilityModule', level='ERROR') as cm:
            result = epm.update_indicators({
                "Invalid1": 1.5,
                "Invalid2": -0.1
            })
        self.assertFalse(result)
        self.assertIn('Invalid probability value', str(cm.output))

    @patch('encryption.xmss_encryption.XMSSEncryption.encrypt')
    def test_encryption_failure(self, mock_encrypt):
        """Test encryption failure fallback"""
        mock_encrypt.side_effect = Exception("Simulated encryption failure")
        epm = EventProbabilityModule()
        with self.assertLogs('EventProbabilityModule', level='ERROR') as cm:
            result = epm.update_indicators({"FOMC": 0.92})
        self.assertFalse(result)
        self.assertEqual(epm.indicators["FOMC"], epm.failover_encrypted)
    def test_traceback_logging(self):
        """Verify complete traceback logging"""
        epm = EventProbabilityModule()
        with patch.object(epm.encryption_engine, 'encrypt', side_effect=Exception("Test error")):
            with self.assertLogs('EventProbabilityModule', level='ERROR') as cm:
                epm.update_indicators({"Test": 0.5})
                
        log_output = '\n'.join(cm.output)
        self.assertIn('Traceback', log_output)
        self.assertIn('Test error', log_output)

if __name__ == '__main__':
    unittest.main()
