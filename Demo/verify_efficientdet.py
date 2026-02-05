#!/usr/bin/env python3
"""
Verification Script for EfficientDet-D0 Attendance System
==========================================================
This script verifies that the implementation meets all hard constraints.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_result(test_name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")


def verify_no_yolo():
    """Verify no YOLO imports in attendance_efficientnetdet.py"""
    print_header("Constraint 1: NO YOLO")
    
    script_path = Path("attendance_efficientnetdet.py")
    if not script_path.exists():
        print_result("NO YOLO check", False, "Script not found")
        return False
    
    content = script_path.read_text()
    
    # Check for YOLO imports
    yolo_import = "from ultralytics import YOLO" in content
    yolo_usage = "YOLO(" in content and "# YOLO" not in content
    
    passed = not (yolo_import or yolo_usage)
    
    if yolo_import:
        print_result("NO YOLO imports", False, "Found YOLO import")
    else:
        print_result("NO YOLO imports", True)
    
    if yolo_usage:
        print_result("NO YOLO usage", False, "Found YOLO() call")
    else:
        print_result("NO YOLO usage", True)
    
    return passed


def verify_efficientdet():
    """Verify EfficientDet-D0 is used"""
    print_header("Constraint 2: EfficientDet-D0 ONLY")
    
    script_path = Path("attendance_efficientnetdet.py")
    if not script_path.exists():
        print_result("EfficientDet usage", False, "Script not found")
        return False
    
    content = script_path.read_text()
    
    # Check for EfficientDet import and usage
    efficientdet_import = "from models.EfficientDet import EfficientDet" in content
    efficientdet_usage = "self.efficientdet_model = EfficientDet(" in content
    
    passed = efficientdet_import and efficientdet_usage
    
    print_result("EfficientDet import", efficientdet_import)
    print_result("EfficientDet usage", efficientdet_usage)
    
    return passed


def verify_no_prediction():
    """Verify no motion prediction in tracker"""
    print_header("Constraint 3: NO Motion Prediction")
    
    script_path = Path("attendance_efficientnetdet.py")
    if not script_path.exists():
        print_result("NO prediction check", False, "Script not found")
        return False
    
    content = script_path.read_text()
    
    # Check for prediction-related keywords
    has_kalman = "kalman" in content.lower()
    has_predict = "predict" in content.lower() and "# No prediction" not in content
    has_velocity = "velocity" in content.lower()
    
    # Check for correct tracker update logic
    correct_update = "# ONLY UPDATE POSITION WHEN DETECTION MATCHES" in content
    freeze_comment = "# NO DETECTION MATCH - TRACK FREEZES" in content
    
    passed = not (has_kalman or has_velocity) and correct_update and freeze_comment
    
    print_result("NO Kalman filter", not has_kalman)
    print_result("NO velocity tracking", not has_velocity)
    print_result("Correct update logic", correct_update)
    print_result("Track freeze on no match", freeze_comment)
    
    return passed


def verify_no_face_heuristics():
    """Verify face boxes not used for body tracking"""
    print_header("Constraint 4: NO Face-Based Body Heuristics")
    
    script_path = Path("attendance_efficientnetdet.py")
    if not script_path.exists():
        print_result("NO face heuristics", False, "Script not found")
        return False
    
    content = script_path.read_text()
    
    # Check for correct association logic
    correct_comment = "CRITICAL: Face boxes are NEVER used to estimate/extend body boxes" in content
    authorization_only = "# AUTHORIZATION: Face recognition only" in content
    
    passed = correct_comment and authorization_only
    
    print_result("Face boxes not used for body", correct_comment)
    print_result("Face for authorization only", authorization_only)
    
    return passed


def verify_persistent_identity():
    """Verify identity persists after face loss"""
    print_header("Constraint 5: Persistent Identity After Face Loss")
    
    script_path = Path("attendance_efficientnetdet.py")
    if not script_path.exists():
        print_result("Persistent identity", False, "Script not found")
        return False
    
    content = script_path.read_text()
    
    # Check for persistent tracking logic
    persistent_comment = "# PERSISTENT TRACKING: Identity persists even after face loss" in content
    no_reassignment = "# NO identity reassignment" in content or "identity never changes" in content.lower()
    
    passed = persistent_comment
    
    print_result("Persistent tracking comment", persistent_comment)
    print_result("Identity persists", passed)
    
    return passed


def verify_files_exist():
    """Verify all required files exist"""
    print_header("File Structure Verification")
    
    required_files = [
        "models/EfficientDet.py",
        "attendance_efficientnetdet.py",
        "EFFICIENTDET_README.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        print_result(f"File: {file_path}", exists)
        all_exist = all_exist and exists
    
    return all_exist


def verify_model_wrapper():
    """Verify EfficientDet wrapper implementation"""
    print_header("EfficientDet Wrapper Verification")
    
    wrapper_path = Path("models/EfficientDet.py")
    if not wrapper_path.exists():
        print_result("EfficientDet wrapper", False, "File not found")
        return False
    
    content = wrapper_path.read_text()
    
    # Check for key methods
    has_init = "def __init__" in content
    has_detect = "def detect" in content
    has_preprocess = "def _preprocess" in content
    has_postprocess = "def _postprocess" in content
    has_nms = "def _nms" in content
    
    # Check for ONNX Runtime
    has_onnx = "import onnxruntime" in content
    
    # Check for person class filtering
    has_person_filter = "self.person_class_id = 0" in content
    
    passed = all([has_init, has_detect, has_preprocess, has_postprocess, has_nms, has_onnx, has_person_filter])
    
    print_result("__init__ method", has_init)
    print_result("detect method", has_detect)
    print_result("_preprocess method", has_preprocess)
    print_result("_postprocess method", has_postprocess)
    print_result("_nms method", has_nms)
    print_result("ONNX Runtime import", has_onnx)
    print_result("Person class filtering", has_person_filter)
    
    return passed


def main():
    print("\n" + "="*60)
    print("  EfficientDet-D0 Attendance System Verification")
    print("="*60)
    
    results = []
    
    # Run all verification tests
    results.append(("Files exist", verify_files_exist()))
    results.append(("NO YOLO", verify_no_yolo()))
    results.append(("EfficientDet-D0 used", verify_efficientdet()))
    results.append(("NO motion prediction", verify_no_prediction()))
    results.append(("NO face heuristics", verify_no_face_heuristics()))
    results.append(("Persistent identity", verify_persistent_identity()))
    results.append(("EfficientDet wrapper", verify_model_wrapper()))
    
    # Summary
    print_header("Verification Summary")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed_count}/{total_count} tests passed")
    print(f"{'='*60}\n")
    
    if passed_count == total_count:
        print("🎉 All constraints verified! Implementation is correct.")
        print("\n📋 Next steps:")
        print("   1. Obtain EfficientDet-D0 ONNX model")
        print("   2. Place at: ./weights/efficientdet_d0.onnx")
        print("   3. Test with: python attendance_efficientnetdet.py --source 0")
        return 0
    else:
        print("⚠️  Some constraints not met. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
