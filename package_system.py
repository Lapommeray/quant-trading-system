"""
Package System Script

This script packages the entire QMP Overrider system into a ZIP file for delivery.
It includes all components, including the QuantConnect integration files.
"""

import os
import sys
import zipfile
import shutil
from datetime import datetime

def package_system(output_path="QMP_Overrider_Beyond_God_Mode.zip"):
    """
    Package the entire system into a ZIP file
    
    Parameters:
    - output_path: Path to the output ZIP file
    
    Returns:
    - Packaging results
    """
    print(f"Packaging system to {output_path}")
    
    temp_dir = "/tmp/qmp_overrider_package"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(repo_dir, os.path.join(temp_dir, "QMP_Overrider_Beyond_God_Mode"), 
                    ignore=shutil.ignore_patterns("*.git*", "*.zip", "*.pyc", "__pycache__"))
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(os.path.join(temp_dir, "QMP_Overrider_Beyond_God_Mode")):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, temp_dir))
    
    shutil.rmtree(temp_dir)
    
    print(f"System packaged to {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    print("\nPackage Contents Summary:")
    print("- Core QMP Overrider components")
    print("- Ultra Modules (Emotion DNA Decoder, Fractal Resonance Gate, etc.)")
    print("- Transcendent Intelligence Layer")
    print("- Dimensional Transcendence Layer")
    print("- Omniscient Core")
    print("- Phoenix Protocol")
    print("- Phase Omega")
    print("- Market Maker Slayer components")
    print("- QuantConnect Integration")
    print("- Google Colab Training Scripts")
    print("- TradingView Integration")
    
    return {
        "status": "SUCCESS",
        "timestamp": datetime.now().timestamp(),
        "output_path": output_path,
        "file_size": os.path.getsize(output_path),
        "message": f"System packaged to {output_path}"
    }

if __name__ == "__main__":
    output_path = "QMP_Overrider_Beyond_God_Mode.zip"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    result = package_system(output_path)
    print(f"System packaged to {result['output_path']}")
    print(f"File size: {result['file_size']} bytes")
