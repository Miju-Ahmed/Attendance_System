#!/usr/bin/env python3
"""
Quick test to verify the attendance system can initialize
"""

import sys
import argparse

# Test imports
print("Testing imports...")
try:
    from models.SCRFD import SCRFD
    from models.ArcFace import ArcFace
    from ultralytics import YOLO
    import cv2
    import numpy as np
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test model loading (just check if files exist)
print("\nChecking model files...")
import os

models_to_check = [
    ("yolo26x.pt", "YOLO Model"),
    ("weights/face_detection/det_10g.onnx", "Face Detection"),
    ("weights/face_recognition/w600k_r50.onnx", "Face Recognition"),
]

all_ok = True
for path, name in models_to_check:
    if os.path.exists(path):
        print(f"✅ {name}: {path}")
    else:
        print(f"❌ {name} not found: {path}")
        all_ok = False

if not all_ok:
    print("\n⚠️  Some models are missing")
    sys.exit(1)

# Test model initialization
print("\nTesting model initialization...")
try:
    print("  Loading SCRFD...")
    face_detector = SCRFD(
        model_path="weights/face_detection/det_10g.onnx",
        conf_thres=0.5
    )
    print("  ✅ SCRFD loaded")
    
    print("  Loading ArcFace...")
    face_recognizer = ArcFace(model_path="weights/face_recognition/w600k_r50.onnx")
    print("  ✅ ArcFace loaded")
    
    print("  Loading YOLO...")
    yolo_model = YOLO("yolo26x.pt")
    print("  ✅ YOLO loaded")
    
    print("\n🎉 All models initialized successfully!")
    print("\nThe system is ready to run:")
    print("  python entry_exit_attendance.py --source 0")
    
except Exception as e:
    print(f"\n❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
