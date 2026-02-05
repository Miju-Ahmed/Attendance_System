#!/usr/bin/env python3
"""
Face Recognition Test Script
=============================
Test face detection and recognition to diagnose issues.
"""

import cv2
import numpy as np
import logging
from models.SCRFD import SCRFD
from models.ArcFace import ArcFace
from attendance_efficientnetdet import AttendanceDatabase, FaceAuthorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Configuration
FACE_DETECTION_MODEL = "./weights/face_detection/det_10g.onnx"
FACE_RECOGNITION_MODEL = "./weights/face_recognition/w600k_r50.onnx"
DATABASE_PATH = "attendance.db"
KNOWN_FACES_TABLE = "face_embeddings"
FACE_CONFIDENCE = 0.5
SIMILARITY_THRESHOLD = 0.40
MIN_FACE_SIZE = 20

def test_face_recognition():
    """Test face detection and recognition on webcam."""
    
    print("\n" + "="*60)
    print("  Face Recognition Test")
    print("="*60 + "\n")
    
    # Load models
    print("Loading models...")
    try:
        face_detector = SCRFD(
            model_path=FACE_DETECTION_MODEL,
            conf_thres=FACE_CONFIDENCE,
        )
        print("✓ SCRFD face detector loaded")
    except Exception as e:
        print(f"✗ Failed to load SCRFD: {e}")
        return
    
    try:
        face_recognizer = ArcFace(model_path=FACE_RECOGNITION_MODEL)
        print("✓ ArcFace recognizer loaded")
    except Exception as e:
        print(f"✗ Failed to load ArcFace: {e}")
        return
    
    # Load known faces
    print("\nLoading known faces from database...")
    try:
        db = AttendanceDatabase(DATABASE_PATH, KNOWN_FACES_TABLE)
        known_faces = db.load_known_faces()
        print(f"✓ Loaded {len(known_faces)} known persons:")
        for stable_id, data in known_faces.items():
            name = data['name']
            num_embeddings = len(data['embeddings'])
            print(f"  - ID {stable_id}: {name} ({num_embeddings} embeddings)")
        
        authorizer = FaceAuthorizer(known_faces, SIMILARITY_THRESHOLD)
        print(f"✓ Face authorizer ready (threshold: {SIMILARITY_THRESHOLD})")
    except Exception as e:
        print(f"✗ Failed to load known faces: {e}")
        return
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return
    print("✓ Webcam opened")
    
    print("\n" + "="*60)
    print("  Press 'q' to quit, 's' to save current face")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        display = frame.copy()
        
        # Detect faces
        try:
            bboxes, kpss = face_detector.detect(frame, max_num=0)
            
            if bboxes is not None and len(bboxes) > 0:
                for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                    x1, y1, x2, y2, score = bbox[:5]
                    
                    # Filter by confidence
                    if score < FACE_CONFIDENCE:
                        continue
                    
                    # Filter by size
                    w = x2 - x1
                    h = y2 - y1
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        continue
                    
                    # Get embedding
                    try:
                        embedding = face_recognizer(frame, kps)
                        
                        # Recognize
                        name, stable_id, similarity = authorizer.identify(embedding)
                        
                        # Draw bounding box
                        if name is not None:
                            color = (0, 255, 0)  # Green for recognized
                            label = f"{name} ({similarity:.2f})"
                            status = "RECOGNIZED"
                        else:
                            color = (0, 0, 255)  # Red for unknown
                            label = f"Unknown ({similarity:.2f})"
                            status = "UNKNOWN"
                        
                        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(display, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Print info every 30 frames
                        if frame_count % 30 == 0:
                            print(f"Face {i+1}: {status} - {label}")
                    
                    except Exception as e:
                        print(f"Error processing face {i+1}: {e}")
                        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
        
        except Exception as e:
            if frame_count % 30 == 0:
                print(f"Face detection error: {e}")
        
        # Display info
        cv2.putText(display, f"Faces: {len(bboxes) if bboxes is not None else 0}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Known: {len(known_faces)}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Saving frame...")
            cv2.imwrite("test_frame.jpg", frame)
            print("✓ Saved to test_frame.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("\n✓ Test complete")


if __name__ == "__main__":
    test_face_recognition()
