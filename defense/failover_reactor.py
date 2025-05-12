"""
Failover Reactor

Monitors system modules and auto-switches to backups for the QMP Overrider system.
"""

from AlgorithmImports import *
import logging
import importlib
import sys
import os
import time
import threading
import psutil
from datetime import datetime

class FailoverSystem:
    """
    Monitors system modules and auto-switches to backups.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Failover System.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("FailoverSystem")
        self.logger.setLevel(logging.INFO)
        
        self.modules = {
            'quantum_engine': '/core/qmp_engine.py',
            'risk_manager': '/defense/risk_system.py',
            'oversoul_director': '/core/oversoul_director.py',
            'self_coder': '/core/self_coder.py',
            'meta_learner': '/core/self_reflection_engine.py'
        }
        
        self.module_status = {}
        
        self.backup_modules = {
            'quantum_engine': '/backup/emergency_qmp_engine.py',
            'risk_manager': '/backup/emergency_risk_system.py',
            'oversoul_director': '/backup/emergency_director.py',
            'self_coder': '/backup/emergency_coder.py',
            'meta_learner': '/backup/emergency_learner.py'
        }
        
        self.failover_history = []
        
        for module_name in self.modules:
            self.module_status[module_name] = {
                'status': 'unknown',
                'last_check': datetime.now(),
                'failures': 0,
                'active_backup': False
            }
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Failover System initialized")
        
    def monitor_systems(self):
        """
        Check the status of all monitored modules.
        
        Returns:
        - Dictionary of module statuses
        """
        self.logger.info("Checking system modules")
        
        for name, path in self.modules.items():
            self._check_module(name, path)
        
        return self.module_status
        
    def _check_module(self, module_name, module_path):
        """
        Check the status of a specific module.
        
        Parameters:
        - module_name: Name of the module
        - module_path: Path to the module
        
        Returns:
        - Boolean indicating if module is healthy
        """
        self.logger.debug(f"Checking module: {module_name}")
        
        self.module_status[module_name]['last_check'] = datetime.now()
        
        if not os.path.exists(module_path):
            self.logger.warning(f"Module file not found: {module_path}")
            self.module_status[module_name]['status'] = 'missing'
            self.module_status[module_name]['failures'] += 1
            
            if self.module_status[module_name]['failures'] >= 3:
                self._activate_backup(module_name)
                
            return False
        
        try:
            module_import_name = module_path.replace('/', '.').replace('.py', '')
            
            if module_import_name.startswith('.'):
                module_import_name = module_import_name[1:]
            
            if module_import_name in sys.modules:
                importlib.reload(sys.modules[module_import_name])
            else:
                importlib.import_module(module_import_name)
            
            self.module_status[module_name]['status'] = 'healthy'
            self.module_status[module_name]['failures'] = 0
            
            if self.module_status[module_name]['active_backup']:
                self._deactivate_backup(module_name)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing module {module_name}: {str(e)}")
            self.module_status[module_name]['status'] = 'error'
            self.module_status[module_name]['failures'] += 1
            
            if self.module_status[module_name]['failures'] >= 3:
                self._activate_backup(module_name)
                
            return False
        
    def _activate_backup(self, module_name):
        """
        Activate backup for a failed module.
        
        Parameters:
        - module_name: Name of the module
        
        Returns:
        - Boolean indicating if backup was activated
        """
        if module_name not in self.backup_modules:
            self.logger.error(f"No backup available for module: {module_name}")
            return False
            
        if self.module_status[module_name]['active_backup']:
            self.logger.info(f"Backup already active for module: {module_name}")
            return True
            
        backup_path = self.backup_modules[module_name]
        
        self.logger.warning(f"Activating backup for module {module_name}: {backup_path}")
        
        try:
            backup_import_name = backup_path.replace('/', '.').replace('.py', '')
            
            if backup_import_name.startswith('.'):
                backup_import_name = backup_import_name[1:]
            
            if backup_import_name in sys.modules:
                backup_module = importlib.reload(sys.modules[backup_import_name])
            else:
                backup_module = importlib.import_module(backup_import_name)
            
            if hasattr(backup_module, 'activate'):
                backup_module.activate()
            
            self.module_status[module_name]['active_backup'] = True
            
            self.failover_history.append({
                'timestamp': datetime.now().isoformat(),
                'module': module_name,
                'action': 'activate_backup',
                'backup_path': backup_path
            })
            
            self.logger.info(f"Backup activated for module: {module_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error activating backup for module {module_name}: {str(e)}")
            return False
        
    def _deactivate_backup(self, module_name):
        """
        Deactivate backup for a recovered module.
        
        Parameters:
        - module_name: Name of the module
        
        Returns:
        - Boolean indicating if backup was deactivated
        """
        if not self.module_status[module_name]['active_backup']:
            return True
            
        backup_path = self.backup_modules[module_name]
        
        self.logger.info(f"Deactivating backup for module {module_name}: {backup_path}")
        
        try:
            backup_import_name = backup_path.replace('/', '.').replace('.py', '')
            
            if backup_import_name.startswith('.'):
                backup_import_name = backup_import_name[1:]
            
            if backup_import_name in sys.modules:
                backup_module = sys.modules[backup_import_name]
                
                if hasattr(backup_module, 'deactivate'):
                    backup_module.deactivate()
            
            self.module_status[module_name]['active_backup'] = False
            
            self.failover_history.append({
                'timestamp': datetime.now().isoformat(),
                'module': module_name,
                'action': 'deactivate_backup',
                'backup_path': backup_path
            })
            
            self.logger.info(f"Backup deactivated for module: {module_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deactivating backup for module {module_name}: {str(e)}")
            return False
        
    def _monitor_loop(self):
        """
        Background thread for continuous monitoring.
        """
        while self.monitoring_active:
            try:
                self.monitor_systems()
                
                self._check_system_resources()
                
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {str(e)}")
                time.sleep(60)
        
    def _check_system_resources(self):
        """
        Check system resources and take action if needed.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            self.logger.debug(f"System resources - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
            
            if memory_percent > 95:
                self.logger.warning(f"Critical memory usage: {memory_percent}%")
                self._handle_resource_crisis('memory')
                
            if disk_percent > 95:
                self.logger.warning(f"Critical disk usage: {disk_percent}%")
                self._handle_resource_crisis('disk')
                
            if cpu_percent > 95:
                self.logger.warning(f"Critical CPU usage: {cpu_percent}%")
                self._handle_resource_crisis('cpu')
                
        except Exception as e:
            self.logger.error(f"Error checking system resources: {str(e)}")
        
    def _handle_resource_crisis(self, resource_type):
        """
        Handle critical resource shortage.
        
        Parameters:
        - resource_type: Type of resource in crisis
        """
        self.logger.warning(f"Handling {resource_type} crisis")
        
        self.failover_history.append({
            'timestamp': datetime.now().isoformat(),
            'module': 'system',
            'action': f'{resource_type}_crisis',
            'details': f'Critical {resource_type} usage detected'
        })
        
        if resource_type == 'memory':
            for module_name in ['quantum_engine', 'self_coder']:
                if module_name in self.modules:
                    self._activate_backup(module_name)
        
    def stop_monitoring(self):
        """
        Stop the monitoring thread.
        """
        self.logger.info("Stopping monitoring")
        self.monitoring_active = False
        
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
    def get_module_status(self, module_name=None):
        """
        Get status of modules.
        
        Parameters:
        - module_name: Name of specific module (optional)
        
        Returns:
        - Module status
        """
        if module_name:
            return self.module_status.get(module_name)
        else:
            return self.module_status
        
    def get_failover_history(self):
        """
        Get failover history.
        
        Returns:
        - Failover history
        """
        return self.failover_history
