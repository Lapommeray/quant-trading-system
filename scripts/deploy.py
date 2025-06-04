#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self, environment: str):
        self.environment = environment
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        config_file = Path(f"deploy/{self.environment}.yaml")
        if not config_file.exists():
            config_file = Path("config.yaml")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def validate_environment(self) -> bool:
        logger.info(f"Validating {self.environment} environment")
        
        required_files = [
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        logger.info("Environment validation passed")
        return True
    
    def run_tests(self) -> bool:
        logger.info("Running test suite")
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("All tests passed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e.stdout}\n{e.stderr}")
            return False
    
    def build_docker_image(self) -> bool:
        logger.info("Building Docker image")
        try:
            tag = f"quant-trading-system:{self.environment}"
            subprocess.run(
                ["docker", "build", "-t", tag, "."],
                check=True
            )
            logger.info(f"Docker image built: {tag}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def deploy_docker(self) -> bool:
        logger.info(f"Deploying to {self.environment}")
        try:
            compose_file = f"docker-compose.{self.environment}.yml"
            if not Path(compose_file).exists():
                compose_file = "docker-compose.yml"
            
            subprocess.run(
                ["docker-compose", "-f", compose_file, "up", "-d"],
                check=True
            )
            logger.info("Deployment completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def health_check(self) -> bool:
        logger.info("Performing health check")
        import time
        import requests
        
        api_url = self.config.get('api', {}).get('url', 'http://localhost:8000')
        max_retries = 30
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Health check passed")
                    return True
            except requests.RequestException:
                pass
            
            logger.info(f"Health check attempt {i+1}/{max_retries}")
            time.sleep(10)
        
        logger.error("Health check failed")
        return False
    
    def rollback(self) -> bool:
        logger.info("Rolling back deployment")
        try:
            subprocess.run(
                ["docker-compose", "down"],
                check=True
            )
            
            previous_tag = f"quant-trading-system:{self.environment}-previous"
            current_tag = f"quant-trading-system:{self.environment}"
            
            subprocess.run(
                ["docker", "tag", previous_tag, current_tag],
                check=True
            )
            
            subprocess.run(
                ["docker-compose", "up", "-d"],
                check=True
            )
            
            logger.info("Rollback completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False) -> bool:
        logger.info(f"Starting deployment to {self.environment}")
        
        if not self.validate_environment():
            return False
        
        if not skip_tests and not self.run_tests():
            return False
        
        if not self.build_docker_image():
            return False
        
        if not self.deploy_docker():
            return False
        
        if not self.health_check():
            logger.warning("Health check failed, considering rollback")
            if input("Rollback? (y/N): ").lower() == 'y':
                return self.rollback()
            return False
        
        logger.info("Deployment completed successfully")
        return True

def main():
    parser = argparse.ArgumentParser(description='Deploy QTS system')
    parser.add_argument('environment', choices=['development', 'staging', 'production'],
                       help='Deployment environment')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests before deployment')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback to previous version')
    
    args = parser.parse_args()
    
    deployment_manager = DeploymentManager(args.environment)
    
    try:
        if args.rollback:
            success = deployment_manager.rollback()
        else:
            success = deployment_manager.deploy(skip_tests=args.skip_tests)
        
        if success:
            print(f"Deployment to {args.environment} completed successfully")
            sys.exit(0)
        else:
            print(f"Deployment to {args.environment} failed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
