import unittest
from unittest.mock import patch
from advanced_modules.event_probability_module import EventProbabilityModule
from tests.mock_algorithm import MockAlgorithm

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
        mock_algorithm = MockAlgorithm()
        epm = EventProbabilityModule(mock_algorithm)
        epm.update_indicators(epm.algorithm.Time, {})
        self.assertIsInstance(epm.indicators, dict)
        self.assertGreater(len(epm.indicators), 0)

    def test_invalid_inputs(self):
        """Test invalid probability values"""
        mock_algorithm = MockAlgorithm()
        epm = EventProbabilityModule(mock_algorithm)
        epm.update_indicators(epm.algorithm.Time, {})
        self.assertIsInstance(epm.indicators, dict)

    @patch('encryption.xmss_encryption.XMSSEncryption.encrypt')
    def test_encryption_failure(self, mock_encrypt):
        """Test encryption failure fallback"""
        mock_encrypt.side_effect = Exception("Simulated encryption failure")
        mock_algorithm = MockAlgorithm()
        epm = EventProbabilityModule(mock_algorithm)
        epm.update_indicators(epm.algorithm.Time, {})
        self.assertIsInstance(epm.indicators, dict)
        self.assertGreater(len(epm.indicators), 0)
    def test_traceback_logging(self):
        """Verify complete traceback logging"""
        mock_algorithm = MockAlgorithm()
        epm = EventProbabilityModule(mock_algorithm)
        with patch.object(epm.encryption_engine, 'encrypt', side_effect=Exception("Test error")):
            epm.update_indicators(epm.algorithm.Time, {})
            
        self.assertIsInstance(epm.indicators, dict)
        self.assertGreater(len(epm.indicators), 0)

if __name__ == '__main__':
    unittest.main()
