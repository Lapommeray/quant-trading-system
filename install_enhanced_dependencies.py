#!/usr/bin/env python3
"""
Enhanced Dependencies Installer

Installs the new mathematical libraries for the Real No-Hopium Trading System:
- geomstats (hyperbolic geometry)
- giotto-tda (topological data analysis)
- qiskit (quantum computing)
- sympy (symbolic mathematics)
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedDependencyInstaller")

def install_package(package_name, version=None):
    """Install a package with optional version specification"""
    try:
        if version:
            package_spec = f"{package_name}>={version}"
        else:
            package_spec = package_name
            
        logger.info(f"Installing {package_spec}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_spec
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully installed {package_spec}")
            return True
        else:
            logger.error(f"❌ Failed to install {package_spec}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing {package_name}: {str(e)}")
        return False

def verify_installation(package_name, import_name=None):
    """Verify that a package was installed correctly"""
    try:
        if import_name is None:
            import_name = package_name.replace('-', '_')
            
        __import__(import_name)
        logger.info(f"✅ {package_name} import verification successful")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  {package_name} import failed: {str(e)}")
        return False

def main():
    """Install and verify all enhanced mathematical dependencies"""
    logger.info("🚀 Installing Enhanced Mathematical Dependencies for Ultimate Never Loss System")
    logger.info("=" * 80)
    
    packages = [
        ("geomstats", "2.6.0", "geomstats"),
        ("giotto-tda", "0.6.0", "gtda"),
        ("qiskit", "0.45.0", "qiskit"),
        ("sympy", "1.12", "sympy")
    ]
    
    installation_results = []
    verification_results = []
    
    for package_name, version, import_name in packages:
        logger.info(f"\n📦 Processing {package_name}...")
        
        install_success = install_package(package_name, version)
        installation_results.append(install_success)
        
        if install_success:
            verify_success = verify_installation(package_name, import_name)
            verification_results.append(verify_success)
        else:
            verification_results.append(False)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 Installation Summary")
    logger.info("=" * 80)
    
    for i, (package_name, version, import_name) in enumerate(packages):
        install_status = "✅ INSTALLED" if installation_results[i] else "❌ FAILED"
        verify_status = "✅ VERIFIED" if verification_results[i] else "❌ FAILED"
        logger.info(f"{package_name:15} | {install_status:12} | {verify_status}")
    
    total_installed = sum(installation_results)
    total_verified = sum(verification_results)
    
    logger.info(f"\nInstallation: {total_installed}/{len(packages)} packages")
    logger.info(f"Verification: {total_verified}/{len(packages)} packages")
    
    if total_installed == len(packages) and total_verified == len(packages):
        logger.info("\n🎯 ALL ENHANCED MATHEMATICAL DEPENDENCIES READY!")
        logger.info("🚀 Ultimate Never Loss System can now use Real No-Hopium mathematics!")
        return True
    else:
        logger.warning("\n⚠️  Some dependencies had issues")
        logger.info("   System will use mock implementations where needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
