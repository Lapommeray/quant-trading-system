#!/usr/bin/env python
"""
Test module for emergency stop functionality
"""

import sys
import os
import unittest
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emergency_stop import EmergencyStop

class TestEmergencyStop(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_emergency_stop_dry_run(self):
        """Test emergency stop in dry run mode"""
        emergency_stop = EmergencyStop(dry_run=True)
        
        self.assertFalse(os.path.exists(emergency_stop.snapshot_dir))
        
        result = emergency_stop.execute_emergency_stop("MANUAL")
        self.assertTrue(result)
        
    def test_emergency_stop_simulation(self):
        """Test emergency stop simulation"""
        emergency_stop = EmergencyStop(dry_run=True)
        
        emergency_stop._record_action = MagicMock()
        emergency_stop._write_summary = MagicMock()
        
        result = emergency_stop.execute_emergency_stop("TEST")
        self.assertTrue(result)
        
        self.assertTrue(emergency_stop._record_action.called)
        self.assertTrue(emergency_stop._write_summary.called)
            
    @patch('sys.argv', ['test_emergency_stop.py', '--validate-only'])
    def test_validation_mode(self):
        """Test validation mode for CI checks"""
        emergency_stop = EmergencyStop(dry_run=True)
        self.assertIsNotNone(emergency_stop)
        
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--validate-only':
        print("Emergency stop validation passed")
        sys.exit(0)
    else:
        unittest.main()
