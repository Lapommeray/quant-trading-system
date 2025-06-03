import unittest
from unittest.mock import patch, MagicMock
from advanced_modules.port_activity_analyzer import PortActivityAnalyzer, PortRiskAssessment
from tests.mock_algorithm import MockAlgorithm
import logging

class TestPortActivityAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.VALID_PORTS = {
            "JPN_TKY": {"cargo_volatility": 0.8},
            "USA_LAX": {"cargo_volatility": 0.3}
        }
        cls.INVALID_PORTS = {
            "BAD_TYPE": {"cargo_volatility": "high"},
            "BAD_RANGE": {"cargo_volatility": 2.0}
        }

    def setUp(self):
        self.mock_algorithm = MockAlgorithm()
        self.analyzer = PortActivityAnalyzer(self.mock_algorithm)

    def tearDown(self):
        logging.getLogger("PortActivityAnalyzer").handlers.clear()

    def test_normal_operation(self):
        """Test valid port risk assessment"""
        result = self.analyzer.analyze(self.VALID_PORTS)
        self.assertTrue(result)
        self.assertEqual(len(self.analyzer.assessments), 2)
        self.assertIsInstance(
            self.analyzer.assessments["JPN_TKY"].encrypted_score,
            bytes
        )

    def test_input_validation(self):
        """Verify score calculation and validation"""
        result = self.analyzer._validate_metrics({"cargo_volatility": 0.5})
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
            
        with self.assertRaises(ValueError):
            self.analyzer._validate_metrics({"cargo_volatility": 2.0})

    def test_quantum_failure(self):
        """Test encryption failover protocol"""
        with patch.object(self.analyzer.quantum_engine, 'encrypt',
                         side_effect=Exception("Quantum disruption")):
            result = self.analyzer.analyze({
                "TEST_PORT": {"cargo_volatility": 0.5}
            })
            
        self.assertFalse(result)
        self.assertEqual(
            self.analyzer.assessments["TEST_PORT"].encrypted_score,
            b"PORT_EMERGENCY_BLOB"
        )
        self.assertTrue(self.analyzer.assessments["TEST_PORT"].used_failover)

    def test_protocol_activation(self):
        """Verify nautical protocol triggers after 2+ failures"""
        with patch.object(self.analyzer.quantum_engine, 'encrypt',
                         side_effect=Exception("Test")):
            self.analyzer.analyze({
                "FAIL1": {"cargo_volatility": 0.1},
                "FAIL2": {"cargo_volatility": 0.2}
            })
            
        self.assertTrue(self.analyzer.protocol_engaged)
        self.assertEqual(self.analyzer.failover_count, 2)

    def test_score_calculation(self):
        """Validate risk score computation"""
        self.assertAlmostEqual(
            self.analyzer._validate_metrics({"cargo_volatility": 0.5}),
            0.35  # 0.5 * 0.7
        )

if __name__ == '__main__':
    unittest.main()
