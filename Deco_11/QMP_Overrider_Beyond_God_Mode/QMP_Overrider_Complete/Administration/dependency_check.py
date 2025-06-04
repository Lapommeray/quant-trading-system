"""
dependency_check.py

Dependency Checker for QMP Overrider

Checks and installs required dependencies for the QMP Overrider system.
"""

import os
import sys
import subprocess
import importlib
import json
from datetime import datetime

class DependencyChecker:
    """
    Dependency Checker for QMP Overrider
    
    Checks and installs required dependencies for the QMP Overrider system.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Dependency Checker
        
        Parameters:
        - config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        self.dependencies = self._load_dependencies()
        self.check_results = {}
        self.last_check_time = None
    
    def _load_dependencies(self):
        """
        Load dependencies from configuration file
        
        Returns:
        - Dictionary with dependencies
        """
        dependencies = {
            "python": {
                "version": "==3.10.*",
                "required": True
            },
            "packages": {
                "pandas": ">=1.0.0",
                "numpy": ">=1.18.0",
                "scikit-learn": ">=0.22.0",
                "matplotlib": ">=3.1.0",
                "plotly": ">=4.0.0",
                "streamlit": ">=1.0.0",
                "requests": ">=2.22.0",
                "websocket-client": ">=0.57.0",
                "joblib": ">=0.14.0",
                "pytz": ">=2019.3",
                "python-dateutil": ">=2.8.0"
            },
            "optional_packages": {
                "tensorflow": ">=2.0.0",
                "torch": ">=1.4.0",
                "xgboost": ">=1.0.0",
                "lightgbm": ">=2.3.0",
                "statsmodels": ">=0.11.0",
                "ta-lib": ">=0.4.0",
                "alpaca-trade-api": ">=0.51.0",
                "ccxt": ">=1.40.0"
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                    
                    if "dependencies" in config:
                        for key, value in config["dependencies"].items():
                            dependencies[key] = value
            except Exception as e:
                print(f"Error loading dependencies: {e}")
        
        return dependencies
    
    def check_dependencies(self, install_missing=False):
        """
        Check dependencies
        
        Parameters:
        - install_missing: Whether to install missing dependencies (optional)
        
        Returns:
        - Dictionary with check results
        """
        self.check_results = {
            "python": None,
            "packages": {},
            "optional_packages": {},
            "missing_packages": [],
            "missing_optional_packages": [],
            "all_required_satisfied": True,
            "timestamp": datetime.now()
        }
        
        self.check_results["python"] = self._check_python_version()
        
        if not self.check_results["python"] and self.dependencies["python"]["required"]:
            self.check_results["all_required_satisfied"] = False
        
        for package, version in self.dependencies["packages"].items():
            self.check_results["packages"][package] = self._check_package(package, version)
            
            if not self.check_results["packages"][package]:
                self.check_results["missing_packages"].append(package)
                self.check_results["all_required_satisfied"] = False
                
                if install_missing:
                    self._install_package(package, version)
                    
                    self.check_results["packages"][package] = self._check_package(package, version)
                    
                    if self.check_results["packages"][package]:
                        self.check_results["missing_packages"].remove(package)
                        
                        if not self.check_results["missing_packages"]:
                            self.check_results["all_required_satisfied"] = True
        
        for package, version in self.dependencies["optional_packages"].items():
            self.check_results["optional_packages"][package] = self._check_package(package, version)
            
            if not self.check_results["optional_packages"][package]:
                self.check_results["missing_optional_packages"].append(package)
                
                if install_missing:
                    self._install_package(package, version)
                    
                    self.check_results["optional_packages"][package] = self._check_package(package, version)
                    
                    if self.check_results["optional_packages"][package]:
                        self.check_results["missing_optional_packages"].remove(package)
        
        self.last_check_time = self.check_results["timestamp"]
        
        return self.check_results
    
    def _check_python_version(self):
        """
        Check Python version
        
        Returns:
        - True if Python version is compatible, False otherwise
        """
        python_version = sys.version.split()[0]
        
        required_version = self.dependencies["python"]["version"]
        
        if required_version.startswith(">="):
            required_version = required_version[2:]
            return self._compare_versions(python_version, required_version) >= 0
        elif required_version.startswith("<="):
            required_version = required_version[2:]
            return self._compare_versions(python_version, required_version) <= 0
        elif required_version.startswith("=="):
            required_version = required_version[2:]
            return self._compare_versions(python_version, required_version) == 0
        elif required_version.startswith(">"):
            required_version = required_version[1:]
            return self._compare_versions(python_version, required_version) > 0
        elif required_version.startswith("<"):
            required_version = required_version[1:]
            return self._compare_versions(python_version, required_version) < 0
        else:
            return self._compare_versions(python_version, required_version) == 0
    
    def _check_package(self, package, version):
        """
        Check package version
        
        Parameters:
        - package: Package name
        - version: Required version
        
        Returns:
        - True if package is installed and version is compatible, False otherwise
        """
        try:
            module = importlib.import_module(package)
            
            if hasattr(module, "__version__"):
                package_version = module.__version__
            elif hasattr(module, "version"):
                package_version = module.version
            else:
                try:
                    import pkg_resources
                    package_version = pkg_resources.get_distribution(package).version
                except:
                    return True
            
            if version.startswith(">="):
                required_version = version[2:]
                return self._compare_versions(package_version, required_version) >= 0
            elif version.startswith("<="):
                required_version = version[2:]
                return self._compare_versions(package_version, required_version) <= 0
            elif version.startswith("=="):
                required_version = version[2:]
                return self._compare_versions(package_version, required_version) == 0
            elif version.startswith(">"):
                required_version = version[1:]
                return self._compare_versions(package_version, required_version) > 0
            elif version.startswith("<"):
                required_version = version[1:]
                return self._compare_versions(package_version, required_version) < 0
            else:
                return self._compare_versions(package_version, version) == 0
        except ImportError:
            return False
    
    def _compare_versions(self, version1, version2):
        """
        Compare versions
        
        Parameters:
        - version1: First version
        - version2: Second version
        
        Returns:
        - -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2
        """
        v1_parts = version1.split(".")
        v2_parts = version2.split(".")
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1 = int(v1_parts[i]) if i < len(v1_parts) else 0
            v2 = int(v2_parts[i]) if i < len(v2_parts) else 0
            
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        
        return 0
    
    def _install_package(self, package, version):
        """
        Install package
        
        Parameters:
        - package: Package name
        - version: Required version
        
        Returns:
        - True if installation was successful, False otherwise
        """
        try:
            if version.startswith(">="):
                package_spec = f"{package}>={version[2:]}"
            elif version.startswith("<="):
                package_spec = f"{package}<={version[2:]}"
            elif version.startswith("=="):
                package_spec = f"{package}=={version[2:]}"
            elif version.startswith(">"):
                package_spec = f"{package}>{version[1:]}"
            elif version.startswith("<"):
                package_spec = f"{package}<{version[1:]}"
            else:
                package_spec = f"{package}=={version}"
            
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
            
            return True
        except Exception as e:
            print(f"Error installing {package}: {e}")
            return False
    
    def get_check_results(self):
        """
        Get dependency check results
        
        Returns:
        - Dictionary with check results
        """
        return self.check_results
    
    def get_status(self):
        """
        Get Dependency Checker status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "last_check_time": self.last_check_time,
            "all_required_satisfied": self.check_results.get("all_required_satisfied", False),
            "missing_packages": self.check_results.get("missing_packages", []),
            "missing_optional_packages": self.check_results.get("missing_optional_packages", [])
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QMP Overrider Dependency Checker")
    parser.add_argument("--install", action="store_true", help="Install missing dependencies")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    dependency_checker = DependencyChecker(args.config)
    
    results = dependency_checker.check_dependencies(args.install)
    
    print("Python:", "OK" if results["python"] else "FAILED")
    
    print("\nRequired Packages:")
    for package, result in results["packages"].items():
        print(f"  {package}: {'OK' if result else 'MISSING'}")
    
    print("\nOptional Packages:")
    for package, result in results["optional_packages"].items():
        print(f"  {package}: {'OK' if result else 'MISSING'}")
    
    print("\nSummary:")
    print(f"  All Required Dependencies: {'SATISFIED' if results['all_required_satisfied'] else 'NOT SATISFIED'}")
    print(f"  Missing Required Packages: {len(results['missing_packages'])}")
    print(f"  Missing Optional Packages: {len(results['missing_optional_packages'])}")

if __name__ == "__main__":
    main()
