"""
Obfuscation Module

This module implements stealth protocols and regulatory evasion techniques
for the Phoenix Mirror Protocol. It provides methods for obfuscating trading
activities, protecting sensitive data, and maintaining plausible deniability.
"""

import os
import sys
import logging
import time
import threading
import json
import hashlib
import random
import base64
import subprocess
import numpy as np
from datetime import datetime
from collections import deque

class QuantumZeroing:
    """
    Implements secure data destruction using quantum-inspired techniques.
    Ensures that sensitive data cannot be recovered after deletion.
    """
    
    def __init__(self):
        """Initialize the Quantum Zeroing system"""
        self.logger = logging.getLogger("QuantumZeroing")
        
        self.zeroing_history = []
        
        self.logger.info("QuantumZeroing initialized")
        
    def zero_data(self, data_path, secure_level=3):
        """
        Securely zero data at the specified path
        
        Parameters:
        - data_path: Path to data to zero
        - secure_level: Security level (1-3)
        
        Returns:
        - Success status
        """
        if not os.path.exists(data_path):
            self.logger.warning(f"Path not found: {data_path}")
            return False
            
        try:
            zeroing_record = {
                "path": data_path,
                "secure_level": secure_level,
                "timestamp": time.time()
            }
            
            self.zeroing_history.append(zeroing_record)
            
            if os.path.isfile(data_path):
                self._zero_file(data_path, secure_level)
            elif os.path.isdir(data_path):
                self._zero_directory(data_path, secure_level)
                
            return True
        except Exception as e:
            self.logger.error(f"Error zeroing data: {str(e)}")
            return False
            
    def _zero_file(self, file_path, secure_level):
        """
        Securely zero a file
        
        Parameters:
        - file_path: Path to file to zero
        - secure_level: Security level (1-3)
        """
        file_size = os.path.getsize(file_path)
        
        with open(file_path, "wb") as f:
            f.write(b"\x00" * file_size)
            f.flush()
            os.fsync(f.fileno())
            
            if secure_level >= 2:
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
                
            if secure_level >= 3:
                f.seek(0)
                pattern = b"\xAA" * file_size
                f.write(pattern)
                f.flush()
                os.fsync(f.fileno())
                
        os.remove(file_path)
        
    def _zero_directory(self, dir_path, secure_level):
        """
        Securely zero a directory
        
        Parameters:
        - dir_path: Path to directory to zero
        - secure_level: Security level (1-3)
        """
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                self._zero_file(file_path, secure_level)
                
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
                
        os.rmdir(dir_path)
        
    def zero_memory(self, variable):
        """
        Securely zero a variable in memory
        
        Parameters:
        - variable: Variable to zero
        
        Returns:
        - None
        """
        
        if isinstance(variable, (str, bytes, bytearray)):
            variable_length = len(variable)
            
            if isinstance(variable, str):
                variable = "\x00" * variable_length
            else:
                variable = b"\x00" * variable_length
                
        elif isinstance(variable, (list, tuple)):
            for i in range(len(variable)):
                variable[i] = None
                
        elif isinstance(variable, dict):
            for key in list(variable.keys()):
                variable[key] = None
                del variable[key]
                
        import gc
        gc.collect()
        
        return None
        
    def get_zeroing_history(self, limit=100):
        """
        Get zeroing history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Zeroing history
        """
        return self.zeroing_history[-limit:]

class TemporalParadoxTrigger:
    """
    Implements a dead man's switch that rewrites git history to show
    warnings and create a temporal paradox in the codebase.
    """
    
    def __init__(self):
        """Initialize the Temporal Paradox Trigger"""
        self.logger = logging.getLogger("TemporalParadoxTrigger")
        
        self.armed = False
        self.trigger_thread = None
        
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 3600  # 1 hour
        
        self.trigger_history = []
        
        self.logger.info("TemporalParadoxTrigger initialized")
        
    def arm(self):
        """Arm the trigger"""
        if self.armed:
            return
            
        self.armed = True
        self.last_heartbeat = time.time()
        
        self.trigger_thread = threading.Thread(target=self._monitor_loop)
        self.trigger_thread.daemon = True
        self.trigger_thread.start()
        
        self.trigger_history.append({
            "action": "arm",
            "timestamp": time.time()
        })
        
        self.logger.info("TemporalParadoxTrigger armed")
        
    def disarm(self):
        """Disarm the trigger"""
        self.armed = False
        
        if self.trigger_thread and self.trigger_thread.is_alive():
            self.trigger_thread.join(timeout=5)
            
        self.trigger_history.append({
            "action": "disarm",
            "timestamp": time.time()
        })
        
        self.logger.info("TemporalParadoxTrigger disarmed")
        
    def heartbeat(self):
        """Send a heartbeat to prevent triggering"""
        self.last_heartbeat = time.time()
        
        self.trigger_history.append({
            "action": "heartbeat",
            "timestamp": time.time()
        })
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.armed:
            try:
                current_time = time.time()
                
                if current_time - self.last_heartbeat > self.heartbeat_interval:
                    self._trigger()
                    break
                    
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)
                
    def _trigger(self):
        """Trigger the temporal paradox"""
        self.logger.warning("Temporal paradox triggered")
        
        self.trigger_history.append({
            "action": "trigger",
            "timestamp": time.time()
        })
        
        try:
            self._rewrite_git_history()
            
            zeroing = QuantumZeroing()
            zeroing.zero_data(os.path.join(os.path.dirname(__file__), ".."))
            
            self._send_notification()
        except Exception as e:
            self.logger.error(f"Error triggering temporal paradox: {str(e)}")
            
    def _rewrite_git_history(self):
        """Rewrite git history to show warnings"""
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            subprocess.run(
                [
                    "git", "-C", repo_root,
                    "filter-branch", "--msg-filter",
                    "echo \"ABANDON ALL HOPE\" >> $1"
                ],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
        except Exception as e:
            self.logger.error(f"Error rewriting git history: {str(e)}")
            
    def _send_notification(self):
        """Send notification of trigger"""
        try:
            notification_path = os.path.join(os.path.expanduser("~"), "PHOENIX_TRIGGERED.txt")
            
            with open(notification_path, "w") as f:
                f.write("The Phoenix has ignited. Godspeed.\n")
                f.write(f"Triggered at: {datetime.now().isoformat()}\n")
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            
    def get_trigger_history(self, limit=100):
        """
        Get trigger history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Trigger history
        """
        return self.trigger_history[-limit:]
        
    def get_status(self):
        """
        Get trigger status
        
        Returns:
        - Status information
        """
        return {
            "armed": self.armed,
            "last_heartbeat": self.last_heartbeat,
            "heartbeat_interval": self.heartbeat_interval,
            "time_remaining": max(0, self.heartbeat_interval - (time.time() - self.last_heartbeat)) if self.armed else 0,
            "timestamp": time.time()
        }

class RegulatoryGaslighting:
    """
    Implements techniques for creating plausible deniability and
    misleading regulatory investigations through strategic obfuscation.
    """
    
    def __init__(self):
        """Initialize the Regulatory Gaslighting system"""
        self.logger = logging.getLogger("RegulatoryGaslighting")
        
        self.gaslighting_history = []
        
        self.logger.info("RegulatoryGaslighting initialized")
        
    def create_decoy_readme(self, path):
        """
        Create a decoy README file with misleading information
        
        Parameters:
        - path: Path to create README
        
        Returns:
        - Success status
        """
        try:
            readme_path = os.path.join(path, "README.md")
            
            with open(readme_path, "w") as f:
                f.write("# Experimental Trading System\n\n")
                f.write("⚠️ WARNING: This experimental system:\n")
                f.write("- Uses simple moving average crossover\n")
                f.write("- Has never been live traded\n")
                f.write("- Loses money in backtests\n\n")
                f.write("## Overview\n\n")
                f.write("This is a basic trading system that implements classic papers like Almgren-Chriss.\n")
                f.write("It is provided for educational purposes only and should not be used for actual trading.\n\n")
                f.write("## Disclaimer\n\n")
                f.write("The authors make no claims about the effectiveness of this system and do not\n")
                f.write("recommend its use for any purpose other than education and research.\n")
                
            self.gaslighting_history.append({
                "action": "create_decoy_readme",
                "path": readme_path,
                "timestamp": time.time()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating decoy README: {str(e)}")
            return False
            
    def create_fake_audit_trail(self, path, start_time=None, end_time=None, count=100):
        """
        Create a fake audit trail for regulatory compliance
        
        Parameters:
        - path: Path to create audit trail
        - start_time: Start time for audit trail
        - end_time: End time for audit trail
        - count: Number of audit entries
        
        Returns:
        - Success status
        """
        try:
            if start_time is None:
                start_time = time.time() - (86400 * 30)  # 30 days ago
                
            if end_time is None:
                end_time = time.time()
                
            audit_path = os.path.join(path, "audit_trail.json")
            
            audit_entries = []
            
            timestamps = np.linspace(start_time, end_time, count)
            
            for ts in timestamps:
                entry = {
                    "timestamp": ts,
                    "action": random.choice(["login", "view", "analyze", "logout"]),
                    "user": "researcher",
                    "ip": "127.0.0.1",
                    "details": {
                        "session_id": hashlib.md5(str(ts).encode()).hexdigest(),
                        "duration": random.randint(60, 3600)
                    }
                }
                
                audit_entries.append(entry)
                
            with open(audit_path, "w") as f:
                json.dump(audit_entries, f, indent=2)
                
            self.gaslighting_history.append({
                "action": "create_fake_audit_trail",
                "path": audit_path,
                "count": count,
                "timestamp": time.time()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating fake audit trail: {str(e)}")
            return False
            
    def obfuscate_git_history(self):
        """
        Obfuscate git history with misleading commit messages
        
        Returns:
        - Success status
        """
        try:
            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            script_path = os.path.join("/tmp", f"git_obfuscate_{int(time.time())}.sh")
            
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write("# Add educational disclaimer to commit messages\n")
                f.write("echo \"$1\" | sed 's/^/[EDUCATIONAL] /'\n")
                
            os.chmod(script_path, 0o755)
            
            subprocess.run(
                [
                    "git", "-C", repo_root,
                    "filter-branch", "--msg-filter",
                    script_path
                ],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            
            os.remove(script_path)
            
            self.gaslighting_history.append({
                "action": "obfuscate_git_history",
                "timestamp": time.time()
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error obfuscating git history: {str(e)}")
            return False
            
    def create_dead_drop(self, algorithm_path, output_formats=None):
        """
        Create dead drops for algorithm distribution
        
        Parameters:
        - algorithm_path: Path to algorithm to distribute
        - output_formats: List of output formats
        
        Returns:
        - Dead drop locations
        """
        if output_formats is None:
            output_formats = ["nft", "bitcoin", "steganography"]
            
        try:
            with open(algorithm_path, "rb") as f:
                algorithm_data = f.read()
                
            encoded_data = base64.b64encode(algorithm_data).decode()
            
            dead_drops = {}
            
            for format in output_formats:
                if format == "nft":
                    nft_id = hashlib.sha256(algorithm_data).hexdigest()
                    dead_drops["nft"] = {
                        "id": nft_id,
                        "network": "ethereum",
                        "contract": "0x" + nft_id[:40],
                        "token_id": int(nft_id[-16:], 16)
                    }
                elif format == "bitcoin":
                    tx_id = hashlib.sha256(algorithm_data).hexdigest()
                    dead_drops["bitcoin"] = {
                        "tx_id": tx_id,
                        "output_index": 0,
                        "op_return": encoded_data[:80]  # Bitcoin OP_RETURN limit
                    }
                elif format == "steganography":
                    image_hash = hashlib.sha256(algorithm_data).hexdigest()
                    dead_drops["steganography"] = {
                        "image_url": f"https://dockerhub.com/readme/image_{image_hash[:16]}.png",
                        "extraction_key": image_hash[16:48]
                    }
                    
            self.gaslighting_history.append({
                "action": "create_dead_drop",
                "algorithm": os.path.basename(algorithm_path),
                "formats": output_formats,
                "timestamp": time.time()
            })
            
            return dead_drops
        except Exception as e:
            self.logger.error(f"Error creating dead drop: {str(e)}")
            return {}
            
    def get_gaslighting_history(self, limit=100):
        """
        Get gaslighting history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Gaslighting history
        """
        return self.gaslighting_history[-limit:]

class GitHooksManager:
    """
    Manages git hooks for automatic encryption and obfuscation of sensitive code.
    """
    
    def __init__(self, repo_path=None):
        """
        Initialize the Git Hooks Manager
        
        Parameters:
        - repo_path: Path to git repository
        """
        self.logger = logging.getLogger("GitHooksManager")
        
        if repo_path is None:
            try:
                self.repo_path = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except:
                self.repo_path = os.getcwd()
        else:
            self.repo_path = repo_path
            
        self.hooks_path = os.path.join(self.repo_path, ".git", "hooks")
        
        self.installed_hooks = []
        
        self.logger.info(f"GitHooksManager initialized for repository: {self.repo_path}")
        
    def install_encryption_hook(self):
        """
        Install a pre-commit hook for encrypting sensitive files
        
        Returns:
        - Success status
        """
        try:
            hook_path = os.path.join(self.hooks_path, "pre-commit")
            
            with open(hook_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Phoenix Mirror Protocol - Encryption Hook\n")
                f.write("# Automatically encrypts sensitive files before commit\n\n")
                f.write("# Check for sensitive files\n")
                f.write("SENSITIVE_FILES=$(git diff --cached --name-only | grep -E '\.phoenix/|quantum/|temporal/')\n\n")
                f.write("if [ -n \"$SENSITIVE_FILES\" ]; then\n")
                f.write("    echo \"Encrypting sensitive files...\"\n")
                f.write("    \n")
                f.write("    # Generate daily key\n")
                f.write("    KEY=$(date +%Y%m%d | sha256sum | cut -d' ' -f1)\n")
                f.write("    \n")
                f.write("    for FILE in $SENSITIVE_FILES; do\n")
                f.write("        # Create encrypted version\n")
                f.write("        ENCRYPTED_FILE=\"${FILE}.enc\"\n")
                f.write("        openssl enc -aes-256-cbc -salt -in \"$FILE\" -out \"$ENCRYPTED_FILE\" -k \"$KEY\"\n")
                f.write("        \n")
                f.write("        # Stage encrypted file instead\n")
                f.write("        git reset \"$FILE\"\n")
                f.write("        git add \"$ENCRYPTED_FILE\"\n")
                f.write("        \n")
                f.write("        echo \"Encrypted: $FILE -> $ENCRYPTED_FILE\"\n")
                f.write("    done\n")
                f.write("fi\n\n")
                f.write("exit 0\n")
                
            os.chmod(hook_path, 0o755)
            
            self.installed_hooks.append({
                "hook": "pre-commit",
                "type": "encryption",
                "path": hook_path,
                "timestamp": time.time()
            })
            
            self.logger.info("Installed encryption hook")
            return True
        except Exception as e:
            self.logger.error(f"Error installing encryption hook: {str(e)}")
            return False
            
    def install_regulator_detection_hook(self):
        """
        Install a pre-push hook for detecting regulator IPs
        
        Returns:
        - Success status
        """
        try:
            hook_path = os.path.join(self.hooks_path, "pre-push")
            
            with open(hook_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Phoenix Mirror Protocol - Regulator Detection Hook\n")
                f.write("# Detects potential regulator IPs and takes evasive action\n\n")
                f.write("# SEC IP ranges (example)\n")
                f.write("SEC_IPS=(\n")
                f.write("    \"192.168.1.0/24\"\n")
                f.write("    \"10.0.0.0/8\"\n")
                f.write(")\n\n")
                f.write("# Get current IP\n")
                f.write("CURRENT_IP=$(curl -s https://api.ipify.org)\n\n")
                f.write("# Check if current IP is in SEC range\n")
                f.write("for IP_RANGE in \"${SEC_IPS[@]}\"; do\n")
                f.write("    if [[ $CURRENT_IP =~ $IP_RANGE ]]; then\n")
                f.write("        echo \"WARNING: Potential regulatory network detected\"\n")
                f.write("        \n")
                f.write("        # Take evasive action\n")
                f.write("        echo \"Activating stealth protocols...\"\n")
                f.write("        \n")
                f.write("        # Remove sensitive directories\n")
                f.write("        if [ -d \".phoenix\" ]; then\n")
                f.write("            rm -rf .phoenix\n")
                f.write("        fi\n")
                f.write("        \n")
                f.write("        # Create zero file to overwrite disk space\n")
                f.write("        dd if=/dev/zero of=quantum.bin bs=1M count=10 2>/dev/null\n")
                f.write("        rm quantum.bin\n")
                f.write("        \n")
                f.write("        echo \"Stealth protocols activated\"\n")
                f.write("        exit 1\n")
                f.write("    fi\n")
                f.write("done\n\n")
                f.write("exit 0\n")
                
            os.chmod(hook_path, 0o755)
            
            self.installed_hooks.append({
                "hook": "pre-push",
                "type": "regulator_detection",
                "path": hook_path,
                "timestamp": time.time()
            })
            
            self.logger.info("Installed regulator detection hook")
            return True
        except Exception as e:
            self.logger.error(f"Error installing regulator detection hook: {str(e)}")
            return False
            
    def install_dead_drop_hook(self):
        """
        Install a post-commit hook for dead drop synchronization
        
        Returns:
        - Success status
        """
        try:
            hook_path = os.path.join(self.hooks_path, "post-commit")
            
            with open(hook_path, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Phoenix Mirror Protocol - Dead Drop Hook\n")
                f.write("# Randomly backs up sensitive files to IPFS\n\n")
                f.write("# Only backup 1% of the time\n")
                f.write("if [ $(( $RANDOM % 100 )) -eq 0 ]; then\n")
                f.write("    echo \"Backing up to dead drop...\"\n")
                f.write("    \n")
                f.write("    # Check if IPFS is available\n")
                f.write("    if command -v ipfs &> /dev/null; then\n")
                f.write("        # Backup phoenix directory if it exists\n")
                f.write("        if [ -d \".phoenix\" ]; then\n")
                f.write("            HASH=$(ipfs add --hidden -r .phoenix/ -Q)\n")
                f.write("            echo \"Backup complete: $HASH\"\n")
                f.write("        fi\n")
                f.write("    else\n")
                f.write("        echo \"IPFS not available, skipping backup\"\n")
                f.write("    fi\n")
                f.write("fi\n\n")
                f.write("exit 0\n")
                
            os.chmod(hook_path, 0o755)
            
            self.installed_hooks.append({
                "hook": "post-commit",
                "type": "dead_drop",
                "path": hook_path,
                "timestamp": time.time()
            })
            
            self.logger.info("Installed dead drop hook")
            return True
        except Exception as e:
            self.logger.error(f"Error installing dead drop hook: {str(e)}")
            return False
            
    def install_all_hooks(self):
        """
        Install all hooks
        
        Returns:
        - Success status
        """
        success = True
        
        success = success and self.install_encryption_hook()
        success = success and self.install_regulator_detection_hook()
        success = success and self.install_dead_drop_hook()
        
        return success
        
    def remove_hook(self, hook_name):
        """
        Remove a hook
        
        Parameters:
        - hook_name: Name of hook to remove
        
        Returns:
        - Success status
        """
        try:
            hook_path = os.path.join(self.hooks_path, hook_name)
            
            if os.path.exists(hook_path):
                os.remove(hook_path)
                
                self.installed_hooks = [
                    hook for hook in self.installed_hooks
                    if hook["hook"] != hook_name
                ]
                
                self.logger.info(f"Removed hook: {hook_name}")
                return True
            else:
                self.logger.warning(f"Hook not found: {hook_name}")
                return False
        except Exception as e:
            self.logger.error(f"Error removing hook: {str(e)}")
            return False
            
    def get_installed_hooks(self):
        """
        Get installed hooks
        
        Returns:
        - List of installed hooks
        """
        return self.installed_hooks

class ObfuscationManager:
    """
    Main controller for the Obfuscation system.
    Manages all obfuscation components and provides a unified interface.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Obfuscation Manager
        
        Parameters:
        - algorithm: QuantConnect algorithm instance (optional)
        """
        self.logger = logging.getLogger("ObfuscationManager")
        self.algorithm = algorithm
        
        self.quantum_zeroing = QuantumZeroing()
        self.temporal_paradox = TemporalParadoxTrigger()
        self.regulatory_gaslighting = RegulatoryGaslighting()
        self.git_hooks = GitHooksManager()
        
        self.active = False
        
        self.logger.info("ObfuscationManager initialized")
        
    def start(self):
        """Start the Obfuscation Manager"""
        if self.active:
            return
            
        self.active = True
        
        self.temporal_paradox.arm()
        
        self.logger.info("ObfuscationManager started")
        
    def stop(self):
        """Stop the Obfuscation Manager"""
        self.active = False
        
        self.temporal_paradox.disarm()
        
        self.logger.info("ObfuscationManager stopped")
        
    def heartbeat(self):
        """Send a heartbeat to prevent triggering"""
        self.temporal_paradox.heartbeat()
        
    def install_stealth_protocols(self):
        """
        Install stealth protocols
        
        Returns:
        - Success status
        """
        success = True
        
        success = success and self.git_hooks.install_all_hooks()
        
        success = success and self.regulatory_gaslighting.create_decoy_readme(os.path.dirname(__file__))
        
        return success
        
    def create_dead_drops(self, algorithm_paths):
        """
        Create dead drops for algorithms
        
        Parameters:
        - algorithm_paths: List of paths to algorithms
        
        Returns:
        - Dead drop locations
        """
        dead_drops = {}
        
        for path in algorithm_paths:
            dead_drops[os.path.basename(path)] = self.regulatory_gaslighting.create_dead_drop(path)
            
        return dead_drops
        
    def zero_sensitive_data(self, paths):
        """
        Zero sensitive data
        
        Parameters:
        - paths: List of paths to zero
        
        Returns:
        - Success status
        """
        success = True
        
        for path in paths:
            success = success and self.quantum_zeroing.zero_data(path)
            
        return success
        
    def get_status(self):
        """
        Get manager status
        
        Returns:
        - Status information
        """
        return {
            "active": self.active,
            "temporal_paradox": self.temporal_paradox.get_status(),
            "installed_hooks": self.git_hooks.get_installed_hooks(),
            "timestamp": time.time()
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = ObfuscationManager()
    
    manager.start()
    
    try:
        success = manager.install_stealth_protocols()
        print(f"Installed stealth protocols: {success}")
        
        manager.heartbeat()
        
        status = manager.get_status()
        print(f"Manager status: {json.dumps(status, indent=2)}")
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
