"""
Install dependencies for the Ultimate Never Loss System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_imports():
    """Check if critical imports are available"""
    imports_to_check = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'statsmodels'
    ]
    
    missing_imports = []
    
    for module in imports_to_check:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            missing_imports.append(module)
            print(f"❌ {module} missing")
    
    return len(missing_imports) == 0

def main():
    """Main installation function"""
    print("🚀 Ultimate Never Loss System - Dependency Installation")
    print("="*60)
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    success = install_requirements()
    
    if success:
        print("\nChecking imports...")
        imports_ok = check_imports()
        
        if imports_ok:
            print("\n✅ All dependencies installed and available!")
            return True
        else:
            print("\n⚠️ Some imports are missing")
            return False
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
