#!/usr/bin/env python3
"""
Quick Test Script for Entry/Exit Attendance System
====================================================
This script verifies that all components are properly set up.
"""

import os
import sys
from pathlib import Path
import numpy as np


def check_file(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} not found: {path}")
        return False


def check_directory(path, description):
    """Check if a directory exists."""
    if os.path.isdir(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description} not found: {path}")
        return False


def check_imports():
    """Check if required Python packages are installed."""
    print("\n📦 Checking Python packages...")
    
    packages = {
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "ultralytics": "Ultralytics (YOLO)",
        "onnxruntime": "ONNX Runtime",
        "sqlite3": "SQLite3"
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_ok = False
    
    return all_ok


def check_models():
    """Check if required models exist."""
    print("\n🔧 Checking models...")
    
    models = {
        "weights/face_detection/det_10g.onnx": "Face Detection (SCRFD)",
        "weights/face_recognition/w600k_r50.onnx": "Face Recognition (ArcFace)",
    }
    
    all_ok = True
    for path, description in models.items():
        if not check_file(path, description):
            all_ok = False
    
    return all_ok


def check_known_faces():
    """Check if known faces are available."""
    print("\n👥 Checking known faces...")
    
    if not check_directory("known_faces", "Known faces directory"):
        return False
    
    known_faces_dir = Path("known_faces")
    npy_files = list(known_faces_dir.glob("*.npy"))
    
    if not npy_files:
        print("⚠️  No .npy face embedding files found")
        print("   The system will work but won't recognize anyone")
        return True
    
    print(f"\n📋 Found {len(npy_files)} known person(s):")
    for npy_file in sorted(npy_files):
        name = npy_file.stem
        try:
            embeddings = np.load(str(npy_file))
            if embeddings.ndim == 1:
                num_embeddings = 1
            else:
                num_embeddings = len(embeddings)
            print(f"   ✅ {name}: {num_embeddings} embedding(s)")
        except Exception as e:
            print(f"   ❌ {name}: Error loading - {e}")
    
    return True


def check_main_script():
    """Check if main script exists."""
    print("\n📄 Checking main script...")
    return check_file("entry_exit_attendance.py", "Main script")


def main():
    """Run all checks."""
    print("="*60)
    print("Entry/Exit Attendance System - Setup Check")
    print("="*60)
    
    checks = [
        ("Python Packages", check_imports),
        ("Models", check_models),
        ("Known Faces", check_known_faces),
        ("Main Script", check_main_script),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SETUP CHECK SUMMARY")
    print("="*60)
    
    all_ok = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_ok = False
    
    print("="*60)
    
    if all_ok:
        print("\n🎉 All checks passed! The system is ready to use.")
        print("\nTo run the system:")
        print("  ./run_attendance.sh")
        print("\nOr manually:")
        print("  python entry_exit_attendance.py --source 0")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
