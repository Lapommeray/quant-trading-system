"""
QMP Overrider Complete - Package for Delivery

This script packages the QMP Overrider Complete system for delivery.
It creates a ZIP file with all the required files and directories.
"""

import os
import sys
import shutil
import zipfile
from datetime import datetime

def create_zip(source_dir, output_filename):
    """
    Create a ZIP file from a directory
    
    Parameters:
    - source_dir: Source directory
    - output_filename: Output filename
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                    zipf.write(file_path, arcname)
        
        return True
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        return False

def main():
    """Main function"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    parent_dir = os.path.dirname(current_dir)
    
    output_dir = os.path.join(parent_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"QMP_Overrider_Complete_Final_{timestamp}.zip")
    
    print(f"Creating ZIP file: {output_filename}")
    
    if create_zip(current_dir, output_filename):
        print(f"ZIP file created successfully: {output_filename}")
    else:
        print("Error creating ZIP file")
        sys.exit(1)
    
    print("\nPackage Contents:")
    print("=================")
    
    for root, dirs, files in os.walk(current_dir):
        level = root.replace(current_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        sub_indent = ' ' * 4 * (level + 1)
        
        for file in files:
            print(f"{sub_indent}{file}")
    
    print("\nDeployment Instructions:")
    print("=======================")
    print("1. Extract the ZIP file")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python Administration/config_manager.py --setup")
    print("4. Run: python main.py --mode full")
    
    print("\nVerification:")
    print("=============")
    print("Run: python verify_integration.py")

if __name__ == "__main__":
    main()
