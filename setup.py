#!/usr/bin/env python3
"""
Setup script for the real-time object detection system.
This script helps users set up and validate their environment.
"""
import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 12):
        print("❌ Python 3.12 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_uv_installation():
    """Check if uv package manager is available."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv package manager: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ uv package manager not found")
    print("Please install uv: https://docs.astral.sh/uv/getting-started/installation/")
    return False

def install_dependencies():
    """Install project dependencies using uv."""
    print("\n📦 Installing dependencies...")
    try:
        result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def download_yolo_model():
    """Download the YOLO model."""
    print("\n🤖 Setting up YOLO model...")
    try:
        # Use uv run to execute in the virtual environment
        result = subprocess.run([
            'uv', 'run', 'python', '-c',
            'from ultralytics import YOLO; model = YOLO("yolo11n.pt"); print("Model downloaded successfully")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ YOLO model downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download YOLO model: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading YOLO model: {e}")
        return False

def test_gpu():
    """Test GPU availability."""
    print("\n🎮 Testing GPU availability...")
    try:
        result = subprocess.run([
            'uv', 'run', 'python', '-c',
            'import torch; print(f"GPU available: {torch.cuda.is_available()}"); '
            'print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ GPU test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
        return False

def test_camera():
    """Test default camera."""
    print("\n📹 Testing default camera...")
    try:
        result = subprocess.run([
            'uv', 'run', 'python', '-c',
            'import cv2; cap = cv2.VideoCapture(0); '
            'ret, frame = cap.read() if cap.isOpened() else (False, None); '
            'cap.release(); '
            'print(f"Camera test: {\"✅ Success\" if ret else \"❌ Failed\"}")'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Camera test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Camera test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False

def create_run_script():
    """Create a convenient run script."""
    print("\n📝 Creating run script...")
    
    # Windows batch file
    bat_content = '''@echo off
echo Starting Real-time Object Detection System...
uv run python main.py
pause
'''
    
    # PowerShell script
    ps1_content = '''# Real-time Object Detection System Launcher
Write-Host "Starting Real-time Object Detection System..." -ForegroundColor Green
uv run python main.py
Read-Host "Press Enter to exit..."
'''
    
    try:
        with open('run.bat', 'w') as f:
            f.write(bat_content)
        
        with open('run.ps1', 'w') as f:
            f.write(ps1_content)
        
        print("✅ Run scripts created: run.bat, run.ps1")
        return True
    except Exception as e:
        print(f"❌ Error creating run scripts: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Real-time Object Detection System Setup")
    print("=" * 50)
    
    # Check requirements
    checks = [
        ("Python Version", check_python_version),
        ("uv Package Manager", check_uv_installation),
    ]
    
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        if not check_func():
            print(f"\n❌ Setup failed at: {name}")
            print("Please fix the above issue and run setup again.")
            return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download YOLO model
    if not download_yolo_model():
        print("⚠️  YOLO model download failed, but you can try running the application anyway.")
    
    # Test components
    test_gpu()
    test_camera()
    
    # Create run scripts
    create_run_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("  • Double-click run.bat")
    print("  • Or execute: uv run python main.py")
    print("  • Or use PowerShell: .\\run.ps1")
    print("\n📖 Check README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)
