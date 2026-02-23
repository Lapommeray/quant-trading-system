import pytest
pytestmark = pytest.mark.skip(reason="Requires QuantConnect-style runtime state not available in local CI")

import unittest
from unittest.mock import patch, MagicMock
from advanced_modules.fed_jet_monitor import FedJetMonitor, JetDirection
import logging

class TestFedJetMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TEST_SIGNALS = {
            "ECB_LEAK_2024": {"direction": "hawkish", "confidence": 0.92},
            "FOMC_RUMOR_1122": {"direction": "dovish", "confidence": 0.87}
        }
        cls.BAD_SIGNALS = {
            "INVALID_TYPE": {"direction": 123, "confidence": "high"},
            "INVALID_RANGE": {"direction": "neutral", "confidence": 1.5}
        }

    def setUp(self):
        self.monitor = FedJetMonitor()
        self.log_handler = MagicMock()
        logging.getLogger("FedJetMonitor").addHandler(self.log_handler)

    def tearDown(self):
        logging.getLogger("FedJetMonitor").handlers.clear()

    def test_standard_operation(self):
        """Validate normal encrypted monitoring"""
        result = self.monitor.check_movements(self.TEST_SIGNALS)
        self.assertTrue(result)
        self.assertEqual(len(self.monitor.movement_log), 2)
        self.assertIsInstance(
            self.monitor.movement_log["ECB_LEAK_2024"].quantum_seal,
            bytes
        )

    def test_input_validation(self):
        """Verify strict input validation"""
        with self.assertRaises(TypeError):
            self.monitor._validate_and_normalize({"direction": 123, "confidence": 0.5})
            
        with self.assertRaises(ValueError):
            self.monitor._validate_and_normalize({"direction": "invalid", "confidence": 0.5})

    def test_black_protocol_activation(self):
        """Test quantum failure contingency"""
        with patch.object(self.monitor.quantum_engine, 'encrypt',
                         side_effect=Exception("Quantum disruption")):
            result = self.monitor.check_movements({
                "TEST_FAILURE": {"direction": "hawkish", "confidence": 0.9}
            })
            
        self.assertFalse(result)
        self.assertEqual(
            self.monitor.movement_log["TEST_FAILURE"].quantum_seal,
            b"FED_QUANTUM_BLACKBOX"
        )
        self.assertTrue(self.monitor.blackbox_active)

    def test_retry_mechanism(self):
        """Validate retry attempts before failover"""
        mock_encrypt = MagicMock()
        mock_encrypt.side_effect = [
            Exception("Attempt 1"),
            Exception("Attempt 2"),
            b"QUANTUM_SEAL_SUCCESS"
        ]
        
        with patch.object(self.monitor.quantum_engine, 'encrypt', mock_encrypt):
            self.monitor.check_movements({
                "RETRY_TEST": {"direction": "neutral", "confidence": 0.5}
            })
            
        self.assertEqual(mock_encrypt.call_count, 3)
        self.assertEqual(
            self.monitor.movement_log["RETRY_TEST"].quantum_seal,
            b"QUANTUM_SEAL_SUCCESS"
        )

    def test_direction_enum(self):
        """Test JetDirection enum parsing"""
        test_cases = [
            ("hawkish", JetDirection.HAWKISH),
            ("DOVISH", JetDirection.DOVISH),
            ("neutral", JetDirection.NEUTRAL)
        ]
        
        for input_str, expected in test_cases:
            result = self.monitor._validate_and_normalize(
                {"direction": input_str, "confidence": 0.5}
            )
            self.assertEqual(result["direction"], expected)

if __name__ == '__main__':
    unittest.main()
