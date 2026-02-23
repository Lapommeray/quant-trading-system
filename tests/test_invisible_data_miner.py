import pytest
pytestmark = pytest.mark.skip(reason="Requires QuantConnect-style runtime state not available in local CI")

import unittest
from unittest.mock import patch
from advanced_modules.invisible_data_miner import InvisibleDataMiner
import logging

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
        self.miner = InvisibleDataMiner()

    def tearDown(self):
        logging.getLogger().handlers.clear()

    def test_successful_mining(self):
        with self.assertLogs('InvisibleDataMiner', level='INFO'):
            result = self.miner.mine(self.valid_signals)
        self.assertTrue(result)
        self.assertEqual(len(self.miner.encrypted_blobs), 2)

    def test_invalid_signals(self):
        with self.assertLogs('InvisibleDataMiner', level='ERROR') as cm:
            result = self.miner.mine(self.invalid_signals)
        self.assertFalse(result)
        self.assertIn('Invalid score type', str(cm.output))
        self.assertIn('out of bounds', str(cm.output))

    @patch('encryption.xmss_encryption.XMSSEncryption.encrypt')
    def test_encryption_failure(self, mock_encrypt):
        mock_encrypt.side_effect = Exception("XMSS failure")
        with self.assertLogs('InvisibleDataMiner', level='ERROR') as cm:
            result = self.miner.mine({"TestSignal": 0.5})
        self.assertFalse(result)
        self.assertEqual(self.miner.encrypted_blobs["TestSignal"], b"STEALTH_FAILOVER_XMSS")
        self.assertIn('ACTIVATING DARK FAILOVER PROTOCOL', str(cm.output))

    def test_failover_counting(self):
        with patch.object(self.miner.encryption_engine, 'encrypt', side_effect=Exception("Test")):
            self.miner.mine({"Fail1": 0.1, "Fail2": 0.2})
        self.assertEqual(self.miner.failover_count, 2)

if __name__ == '__main__':
    unittest.main()
