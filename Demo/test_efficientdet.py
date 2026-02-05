#!/usr/bin/env python3
"""
Test EfficientDet-D0 Detection
===============================
Simple test to verify EfficientDet model is working correctly.
"""

import cv2
import logging
from models.EfficientDet import EfficientDet

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

def test_efficientdet():
    print("\n" + "="*60)
    print("  EfficientDet-D0 Detection Test")
    print("="*60 + "\n")
    
    # Load model
    print("Loading EfficientDet-D0 model...")
    try:
        model = EfficientDet(
            model_path="./weights/efficientdet-d0.onnx",
            conf_thres=0.3,
            nms_thres=0.5,
            providers=("CPUExecutionProvider",),
        )
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return
    print("✓ Webcam opened\n")
    
    print("="*60)
    print("  Press 'q' to quit")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        display = frame.copy()
        
        # Detect persons
        detections = model.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display info
        cv2.putText(display, f"Persons: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame_count}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Print info every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Detected {len(detections)} persons")
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf = det
                print(f"  Person {i+1}: conf={conf:.3f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        
        cv2.imshow("EfficientDet-D0 Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")


if __name__ == "__main__":
    test_efficientdet()
