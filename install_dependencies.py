"""
Install dependencies for the Institutional Quant Trading System

Robust dependency installation for:
- Google Colab environment
- Local development
- Production deployment
"""

import subprocess
import sys
import os
import platform


def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_notebook():
    """Check if running in a Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except:
        pass
    return False


def install_package(package, upgrade=False):
    """Install a single package"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False


def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        print("Installing core dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            stdout=subprocess.DEVNULL if not os.environ.get("VERBOSE") else None,
            stderr=subprocess.DEVNULL if not os.environ.get("VERBOSE") else None
        )
        print("[OK] Core dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Some core dependencies failed: {e}")
        return False


def install_additional_packages():
    """Install additional packages not in requirements.txt"""
    additional_packages = [
        "ccxt",
        "mplfinance",
        "networkx",
        "pyarrow",
        "PyWavelets",
        "vaderSentiment",
        "textblob",
        "tweepy",
        "polygon-api-client",
        "ib_insync",
        "dask",
        "distributed",
        "aiohttp",
        "websocket-client",
        "apscheduler",
        "river",
    ]
    
    print("\nInstalling additional packages...")
    installed = 0
    failed = []
    
    for package in additional_packages:
        if install_package(package):
            installed += 1
        else:
            failed.append(package)
            
    print(f"[OK] Installed {installed}/{len(additional_packages)} additional packages")
    
    if failed:
        print(f"[WARN] Failed to install: {', '.join(failed)}")
        
    return len(failed) == 0


def install_colab_specific():
    """Install Colab-specific packages"""
    if not is_colab():
        return True
        
    print("\nInstalling Colab-specific packages...")
    
    colab_packages = [
        "pyngrok",
        "google-cloud-storage",
    ]
    
    for package in colab_packages:
        install_package(package)
        
    print("[OK] Colab packages installed")
    return True


def check_imports():
    """Check if critical imports are available"""
    imports_to_check = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'statsmodels': 'statsmodels',
        'torch': 'pytorch (optional)',
        'tensorflow': 'tensorflow (optional)',
        'ccxt': 'ccxt',
        'websocket': 'websocket-client',
    }
    
    available = []
    missing = []
    optional_missing = []
    
    for module, name in imports_to_check.items():
        try:
            __import__(module)
            available.append(name)
        except ImportError:
            if 'optional' in name:
                optional_missing.append(name)
            else:
                missing.append(name)
    
    print(f"\n[OK] Available: {len(available)} packages")
    
    if missing:
        print(f"[WARN] Missing required: {', '.join(missing)}")
        
    if optional_missing:
        print(f"[INFO] Missing optional: {', '.join(optional_missing)}")
    
    return len(missing) == 0


def setup_colab_environment():
    """Setup Colab-specific environment"""
    if not is_colab():
        return
        
    print("\nConfiguring Colab environment...")
    
    try:
        from google.colab import drive
        print("[INFO] Google Drive available for mounting")
    except:
        pass
        
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    print("[OK] Colab environment configured")


def setup_ngrok_tunnel(port=8000):
    """Setup ngrok tunnel for webhook callbacks (Colab only)"""
    if not is_colab():
        print("[INFO] ngrok tunnel only available in Colab")
        return None
        
    try:
        from pyngrok import ngrok
        
        ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
        if ngrok_token:
            ngrok.set_auth_token(ngrok_token)
            
        tunnel = ngrok.connect(port)
        print(f"[OK] ngrok tunnel established: {tunnel.public_url}")
        return tunnel.public_url
        
    except Exception as e:
        print(f"[WARN] Failed to setup ngrok: {e}")
        return None


def verify_system():
    """Verify system is ready for trading"""
    print("\n" + "="*60)
    print("SYSTEM VERIFICATION")
    print("="*60)
    
    checks = {
        "Python version": sys.version.split()[0],
        "Platform": platform.system(),
        "Architecture": platform.machine(),
        "Colab": "Yes" if is_colab() else "No",
        "Notebook": "Yes" if is_notebook() else "No",
    }
    
    for check, value in checks.items():
        print(f"  {check}: {value}")
        
    try:
        import numpy as np
        import pandas as pd
        
        arr = np.random.randn(100)
        df = pd.DataFrame({'test': arr})
        
        print("\n[OK] Core numerical libraries working")
        
    except Exception as e:
        print(f"\n[ERROR] Core library test failed: {e}")
        return False
        
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_comprehensive_test import run_comprehensive_tests
        print("[OK] Trading system modules accessible")
    except Exception as e:
        print(f"[WARN] Trading system import check: {e}")
        
    return True


def main():
    """Main installation function"""
    print("="*60)
    print("INSTITUTIONAL QUANT TRADING SYSTEM")
    print("Dependency Installation & Environment Setup")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if not os.path.exists("requirements.txt"):
        print("[ERROR] requirements.txt not found")
        print(f"[INFO] Current directory: {os.getcwd()}")
        return False
    
    success = install_requirements()
    
    install_additional_packages()
    
    install_colab_specific()
    
    setup_colab_environment()
    
    imports_ok = check_imports()
    
    system_ok = verify_system()
    
    print("\n" + "="*60)
    if imports_ok and system_ok:
        print("INSTALLATION COMPLETE")
        print("="*60)
        print("\nQuick Start:")
        print("  1. Run tests: python run_comprehensive_test.py")
        print("  2. Start system: python main.py")
        print("  3. MT5 deployment: python deploy_mt5.py --paper")
        
        if is_colab():
            print("\nColab Usage:")
            print("  %run main.py")
            print("  %run deploy_mt5.py --paper")
            
        return True
    else:
        print("INSTALLATION INCOMPLETE")
        print("="*60)
        print("\nSome dependencies may be missing.")
        print("The system may still work with reduced functionality.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
