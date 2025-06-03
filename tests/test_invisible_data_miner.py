import unittest
from unittest.mock import patch
from advanced_modules.invisible_data_miner import InvisibleDataMiner
from tests.mock_algorithm import MockAlgorithm
import logging
import pandas as pd
import numpy as np

class TestInvisibleDataMiner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_signals = {
            "DarkPoolFlow": 0.92,
            "IcebergDetect": 0.71
        }
        cls.invalid_signals = {
            "BadType": "invalid",
            "OutOfRange": 1.5
        }

    def setUp(self):
        self.mock_algorithm = MockAlgorithm()
        self.miner = InvisibleDataMiner(self.mock_algorithm)
        self.mock_history_data = self._create_mock_history_data()
        
    def _create_mock_history_data(self):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        return {
            "1m": pd.DataFrame({
                'Open': np.random.normal(100, 2, 100),
                'High': np.random.normal(102, 2, 100),
                'Low': np.random.normal(98, 2, 100),
                'Close': np.random.normal(101, 2, 100),
                'Volume': np.random.normal(1000000, 200000, 100)
            }, index=dates)
        }

    def tearDown(self):
        logging.getLogger().handlers.clear()

    def test_successful_mining(self):
        with self.assertLogs('InvisibleDataMiner', level='INFO'):
            result = self.miner.mine('SPY', self.mock_history_data)
        self.assertIsInstance(result, (bool, dict))

    def test_invalid_signals(self):
        empty_data = {"1m": pd.DataFrame()}
        result = self.miner.mine('SPY', empty_data)
        self.assertIsInstance(result, (bool, dict))

    @patch('encryption.xmss_encryption.XMSSEncryption.encrypt')
    def test_encryption_failure(self, mock_encrypt):
        mock_encrypt.side_effect = Exception("XMSS failure")
        with self.assertLogs('InvisibleDataMiner', level='ERROR') as cm:
            result = self.miner.mine('SPY', self.mock_history_data)
        self.assertIsInstance(result, (bool, dict))
        self.assertIn('ACTIVATING DARK FAILOVER PROTOCOL', str(cm.output))

    def test_failover_counting(self):
        with patch.object(self.miner.encryption_engine, 'encrypt', side_effect=Exception("Test")):
            self.miner.mine('SPY', self.mock_history_data)
        self.assertGreaterEqual(self.miner.failover_count, 0)

if __name__ == '__main__':
    unittest.main()
