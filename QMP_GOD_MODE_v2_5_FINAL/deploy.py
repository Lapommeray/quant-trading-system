#!/usr/bin/env python
"""
Deployment script for QMP Trading System
This script handles deployment to paper trading and live environments
with configurable capital allocation and dry run capability.
"""

import sys
import os
import argparse
import logging
import json
import time
from datetime import datetime
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Deployment")

class Deployer:
    """Deployment manager for QMP Trading System"""
    
    def __init__(self):
        self.logger = logger
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.deployment_dir = os.path.join(
            self.base_dir,
            "deployments",
            f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.deployment_dir, exist_ok=True)
        
    def deploy(self, env="paper", allocation=1.0, validate_only=False):
        """Deploy the system to the specified environment"""
        self.logger.info(f"Deploying to {env} environment with {allocation*100:.0f}% capital allocation")
        
        if validate_only:
            self.logger.info("Validation mode only - no actual deployment will occur")
            
        self._record_deployment_params(env, allocation, validate_only)
        
        if not self._validate_system_readiness():
            self.logger.error("System validation failed - deployment aborted")
            return False
            
        if validate_only:
            self.logger.info("Validation successful - dry run complete")
            return True
            
        if not self._prepare_deployment_package(env):
            self.logger.error("Failed to prepare deployment package - deployment aborted")
            return False
            
        if not self._configure_environment(env, allocation):
            self.logger.error("Failed to configure environment - deployment aborted")
            return False
            
        if not self._deploy_to_environment(env):
            self.logger.error("Deployment failed")
            return False
            
        if not self._verify_deployment(env):
            self.logger.error("Deployment verification failed")
            return False
            
        self.logger.info(f"Deployment to {env} environment successful")
        return True
        
    def _validate_system_readiness(self):
        """Validate that the system is ready for deployment"""
        self.logger.info("Validating system readiness...")
        
        verification_script = os.path.join(self.base_dir, "launch_verification.sh")
        if not os.path.exists(verification_script):
            self.logger.error("Verification script not found")
            return False
            
        try:
            self.logger.info("Running verification tests...")
            result = subprocess.run(
                [verification_script, "--mode=production", "--stress-level=high"],
                capture_output=True,
                text=True,
                check=False
            )
            
            with open(os.path.join(self.deployment_dir, "verification_output.log"), "w") as f:
                f.write(result.stdout)
                f.write("\n\n")
                f.write(result.stderr)
                
            if result.returncode != 0:
                self.logger.error("Verification tests failed")
                return False
                
            self.logger.info("Verification tests passed")
            
        except Exception as e:
            self.logger.error(f"Error running verification tests: {e}")
            return False
            
        lock_file = os.path.join(self.base_dir, "trading.lock")
        if os.path.exists(lock_file):
            self.logger.error("Trading lock file exists - deployment aborted")
            return False
            
        try:
            self.logger.info("Validating circuit breakers...")
            circuit_breaker_test = subprocess.run(
                ["python", "-m", "tests.test_emergency_stop", "--validate-only"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if circuit_breaker_test.returncode != 0:
                self.logger.error("Circuit breaker validation failed")
                return False
                
            self.logger.info("Circuit breakers validated")
            
        except Exception as e:
            self.logger.error(f"Error validating circuit breakers: {e}")
            return False
            
        try:
            self.logger.info("Validating kill switch...")
            kill_switch_test = subprocess.run(
                ["python", "emergency_stop.py", "--code", "MANUAL", "--dry-run"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if kill_switch_test.returncode != 0:
                self.logger.error("Kill switch validation failed")
                return False
                
            self.logger.info("Kill switch validated")
            
        except Exception as e:
            self.logger.error(f"Error validating kill switch: {e}")
            return False
            
        try:
            self.logger.info("Validating monitoring dashboards...")
            
            dashboard_dir = os.path.join(self.base_dir, "dashboards")
            required_dashboards = ["leakage_detection.json", "tail_risk_monitor.json", "slippage_radar.json"]
            
            missing_dashboards = []
            for dashboard in required_dashboards:
                if not os.path.exists(os.path.join(dashboard_dir, dashboard)):
                    missing_dashboards.append(dashboard)
                    
            if missing_dashboards:
                self.logger.warning(f"Missing dashboards: {', '.join(missing_dashboards)}")
                
            self.logger.info("Monitoring dashboards validated")
            
        except Exception as e:
            self.logger.error(f"Error validating monitoring dashboards: {e}")
            
        return True
        
    def _prepare_deployment_package(self, env):
        """Prepare the deployment package"""
        self.logger.info("Preparing deployment package...")
        
        try:
            package_dir = os.path.join(self.deployment_dir, "package")
            os.makedirs(package_dir, exist_ok=True)
            
            core_dirs = ["core", "data", "modules", "tests"]
            for dir_name in core_dirs:
                src_dir = os.path.join(self.base_dir, dir_name)
                if os.path.exists(src_dir):
                    dst_dir = os.path.join(package_dir, dir_name)
                    shutil.copytree(src_dir, dst_dir)
                    
            shutil.copy(
                os.path.join(self.base_dir, "main.py"),
                os.path.join(package_dir, "main.py")
            )
            
            shutil.copy(
                os.path.join(self.base_dir, "emergency_stop.py"),
                os.path.join(package_dir, "emergency_stop.py")
            )
            
            shutil.copy(
                os.path.join(self.base_dir, "requirements.txt"),
                os.path.join(package_dir, "requirements.txt")
            )
            
            config = {
                "environment": env,
                "timestamp": datetime.now().isoformat(),
                "version": "2.5.0"
            }
            
            with open(os.path.join(package_dir, "deployment_config.json"), "w") as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Deployment package created at {package_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing deployment package: {e}")
            return False
            
    def _configure_environment(self, env, allocation):
        """Configure the environment for deployment"""
        self.logger.info(f"Configuring {env} environment with {allocation*100:.0f}% allocation...")
        
        try:
            package_dir = os.path.join(self.deployment_dir, "package")
            
            env_config = {
                "environment": env,
                "allocation": allocation,
                "timestamp": datetime.now().isoformat(),
                "circuit_breakers": {
                    "max_drawdown": 0.15,
                    "volatility_threshold": 0.02,
                    "news_events": ["NFP", "FOMC", "CPI", "GDP"]
                },
                "risk_parameters": {
                    "max_position_size": 0.1 * allocation,
                    "kelly_fraction": 0.5,
                    "stop_loss": 0.15
                }
            }
            
            with open(os.path.join(package_dir, f"{env}_config.json"), "w") as f:
                json.dump(env_config, f, indent=2)
                
            startup_script = f"""#!/bin/bash
export QMP_ENV="{env}"
export QMP_ALLOCATION="{allocation}"
export QMP_CONFIG_FILE="{env}_config.json"

python main.py --env {env} --allocation {allocation}
"""
            
            with open(os.path.join(package_dir, f"start_{env}.sh"), "w") as f:
                f.write(startup_script)
                
            os.chmod(os.path.join(package_dir, f"start_{env}.sh"), 0o755)
            
            self.logger.info(f"Environment {env} configured with {allocation*100:.0f}% allocation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring environment: {e}")
            return False
            
    def _deploy_to_environment(self, env):
        """Deploy to the target environment"""
        self.logger.info(f"Deploying to {env} environment...")
        
        try:
            package_dir = os.path.join(self.deployment_dir, "package")
            
            if env == "paper":
                
                paper_dir = os.path.join(self.base_dir, "paper_trading")
                os.makedirs(paper_dir, exist_ok=True)
                
                paper_deploy_dir = os.path.join(paper_dir, f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.symlink(package_dir, paper_deploy_dir)
                
                current_link = os.path.join(paper_dir, "current")
                if os.path.exists(current_link):
                    os.remove(current_link)
                os.symlink(paper_deploy_dir, current_link)
                
                self.logger.info(f"Deployed to paper trading environment at {paper_deploy_dir}")
                
            elif env == "live":
                
                live_dir = os.path.join(self.base_dir, "live_trading")
                os.makedirs(live_dir, exist_ok=True)
                
                live_deploy_dir = os.path.join(live_dir, f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.symlink(package_dir, live_deploy_dir)
                
                current_link = os.path.join(live_dir, "current")
                if os.path.exists(current_link):
                    os.remove(current_link)
                os.symlink(live_deploy_dir, current_link)
                
                self.logger.info(f"Deployed to live trading environment at {live_deploy_dir}")
                
            else:
                self.logger.error(f"Unknown environment: {env}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying to environment: {e}")
            return False
            
    def _verify_deployment(self, env):
        """Verify the deployment"""
        self.logger.info(f"Verifying {env} deployment...")
        
        try:
            if env == "paper":
                paper_dir = os.path.join(self.base_dir, "paper_trading", "current")
                if not os.path.exists(paper_dir):
                    self.logger.error("Paper trading deployment not found")
                    return False
                    
                startup_script = os.path.join(paper_dir, "start_paper.sh")
                if not os.path.exists(startup_script):
                    self.logger.error("Paper trading startup script not found")
                    return False
                    
                self.logger.info("Paper trading deployment verified")
                
            elif env == "live":
                live_dir = os.path.join(self.base_dir, "live_trading", "current")
                if not os.path.exists(live_dir):
                    self.logger.error("Live trading deployment not found")
                    return False
                    
                startup_script = os.path.join(live_dir, "start_live.sh")
                if not os.path.exists(startup_script):
                    self.logger.error("Live trading startup script not found")
                    return False
                    
                self.logger.info("Live trading deployment verified")
                
            else:
                self.logger.error(f"Unknown environment: {env}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying deployment: {e}")
            return False
            
    def _record_deployment_params(self, env, allocation, validate_only):
        """Record deployment parameters"""
        params = {
            "environment": env,
            "allocation": allocation,
            "validate_only": validate_only,
            "timestamp": datetime.now().isoformat(),
            "user": os.environ.get("USER", "unknown")
        }
        
        with open(os.path.join(self.deployment_dir, "deployment_params.json"), "w") as f:
            json.dump(params, f, indent=2)
            
def main():
    """Main function to execute deployment"""
    parser = argparse.ArgumentParser(description="Deployment for QMP Trading System")
    parser.add_argument("--env", choices=["paper", "live"], default="paper", 
                        help="Deployment environment")
    parser.add_argument("--allocation", type=float, default=1.0,
                        help="Capital allocation (0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, don't actually deploy")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate only, don't actually deploy")
    
    args = parser.parse_args()
    
    if args.allocation < 0.0 or args.allocation > 1.0:
        print(f"Error: Allocation must be between 0.0 and 1.0, got {args.allocation}")
        sys.exit(1)
        
    deployer = Deployer()
    success = deployer.deploy(
        env=args.env,
        allocation=args.allocation,
        validate_only=args.dry_run or args.validate_only
    )
    
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main()
