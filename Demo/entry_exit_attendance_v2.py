"""
Entry/Exit Attendance System - Face-Based Only
===============================================
This script tracks faces crossing a red line for entry/exit attendance:
- Uses SCRFD for face detection (NO YOLO - direct face detection)
- Uses ArcFace for face recognition
- Multi-Embedding Fusion (MEF) for better accuracy
- Tracks entry/exit based on red line crossing
- Stores attendance with date/time in SQLite database
- Loads known faces from known_faces folder
"""

import os
import cv2
import numpy as np
import sqlite3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

from ultralytics import YOLO
from models.SCRFD import SCRFD
from models.ArcFace import ArcFace

# =====================
# CONFIGURATION
# =====================

# Model paths
FACE_DETECTION_MODEL = "./weights/face_detection/det_10g.onnx"
FACE_RECOGNITION_MODEL = "./weights/face_recognition/w600k_r50.onnx"
YOLO_MODEL_PATH = "yolo26x.pt"  # YOLO model for person detection
KNOWN_FACES_DIR = "./known_faces"

# Detection parameters
FACE_CONFIDENCE = 0.35  # Lowered from 0.5 for better face detection
FACE_SIMILARITY_THRESHOLD = 0.45  # Threshold for face matching
YOLO_CONFIDENCE = 0.3  # Lowered from 0.4 for better body detection
PERSON_CLASS_ID = 0  # YOLO class ID for person

# Face-to-Body association
FACE_TO_BODY_MAX_DISTANCE = 150  # Maximum distance to associate face with body

# Line crossing parameters
MIN_MOVEMENT = 3.0  # Minimum pixels to consider movement
COOLDOWN_FRAMES = 15  # Reduced from 30 for faster re-detection (0.5s at 30fps)

# Multi-Embedding Fusion parameters
MEF_BUFFER_SIZE = 5  # Number of embeddings to keep for fusion
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]  # Weights for recent embeddings

# Tracking parameters
MAX_TRACK_DISTANCE = 150  # Increased from 100 to allow larger movements
MIN_TRACK_CONFIDENCE = 1  # Reduced from 3 for faster counting
USE_BODY_TRACKING = True  # Enable YOLO body tracking (set to False for face-only mode)

# Database
DATABASE_PATH = "attendance.db"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class AttendanceDatabase:
    """Handles SQLite database operations for attendance records."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create attendance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence REAL,
                track_id INTEGER
            )
        """)
        
        self.conn.commit()
        logging.info(f"Database initialized at: {self.db_path}")
    
    def record_entry(self, person_name: str, confidence: float, track_id: int):
        """Record an entry event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            INSERT INTO attendance (person_name, event_type, timestamp, confidence, track_id)
            VALUES (?, ?, ?, ?, ?)
        """, (person_name, "ENTRY", timestamp, confidence, track_id))
        self.conn.commit()
        logging.info(f"✅ ENTRY: {person_name} at {timestamp} (Track {track_id}, Conf: {confidence:.2f})")
    
    def record_exit(self, person_name: str, confidence: float, track_id: int):
        """Record an exit event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            INSERT INTO attendance (person_name, event_type, timestamp, confidence, track_id)
            VALUES (?, ?, ?, ?, ?)
        """, (person_name, "EXIT", timestamp, confidence, track_id))
        self.conn.commit()
        logging.info(f"❌ EXIT: {person_name} at {timestamp} (Track {track_id}, Conf: {confidence:.2f})")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class KnownFaceLoader:
    """Loads known faces from the known_faces directory."""
    
    def __init__(self, known_faces_dir: str):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_embeddings = {}  # name -> list of embeddings
        self.known_names = []
        
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load all known faces from .npy files."""
        if not self.known_faces_dir.exists():
            logging.warning(f"Known faces directory not found: {self.known_faces_dir}")
            return
        
        # Load .npy files (pre-computed embeddings)
        for npy_file in self.known_faces_dir.glob("*.npy"):
            name = npy_file.stem
            try:
                embeddings = np.load(str(npy_file))
                # Handle both single embedding and multiple embeddings
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                self.known_embeddings[name] = embeddings
                self.known_names.append(name)
                logging.info(f"Loaded {len(embeddings)} embeddings for: {name}")
            except Exception as e:
                logging.error(f"Failed to load embeddings for {name}: {e}")
        
        logging.info(f"Total known persons loaded: {len(self.known_names)}")
    
    def identify_face(self, embedding: np.ndarray, threshold: float = FACE_SIMILARITY_THRESHOLD) -> Tuple[Optional[str], float]:
        """
        Identify a face by comparing embedding with known faces.
        
        Returns:
            Tuple of (name, similarity) or (None, 0.0) if no match
        """
        if not self.known_embeddings:
            return None, 0.0
        
        best_name = None
        best_similarity = 0.0
        
        # Normalize query embedding
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Compare with all known embeddings
        for name, embeddings in self.known_embeddings.items():
            for known_emb in embeddings:
                # Normalize known embedding
                known_emb = known_emb.flatten()
                known_norm = np.linalg.norm(known_emb)
                if known_norm > 0:
                    known_emb = known_emb / known_norm
                
                # Compute cosine similarity
                similarity = float(np.dot(embedding, known_emb))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
        
        # Return match if above threshold
        if best_similarity >= threshold:
            return best_name, best_similarity
        
        return None, best_similarity


class MultiEmbeddingFusion:
    """Multi-Embedding Fusion for robust face recognition."""
    
    def __init__(self, buffer_size: int = MEF_BUFFER_SIZE, weights: List[float] = None):
        self.buffer_size = buffer_size
        self.weights = weights or MEF_WEIGHTS[:buffer_size]
        self.embedding_buffers = {}
    
    def add_embedding(self, track_id: int, embedding: np.ndarray):
        """Add an embedding to the track's buffer."""
        if track_id not in self.embedding_buffers:
            self.embedding_buffers[track_id] = deque(maxlen=self.buffer_size)
        
        self.embedding_buffers[track_id].append(embedding.copy())
    
    def get_fused_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get the fused embedding using weighted average."""
        if track_id not in self.embedding_buffers:
            return None
        
        buffer = self.embedding_buffers[track_id]
        if len(buffer) == 0:
            return None
        
        # Get weights for current buffer size
        num_embeddings = len(buffer)
        weights = self.weights[:num_embeddings]
        weights = np.array(weights) / sum(weights)
        
        # Compute weighted average (recent = higher weight)
        fused = np.zeros_like(buffer[0])
        for i, emb in enumerate(buffer):
            fused += weights[num_embeddings - 1 - i] * emb
        
        # Normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        
        return fused
    
    def clear_track(self, track_id: int):
        """Clear embedding buffer for a track."""
        if track_id in self.embedding_buffers:
            del self.embedding_buffers[track_id]


class LineCrossDetector:
    """Handles line crossing detection for entry/exit tracking."""
    
    def __init__(self, line_coords: Tuple[int, int, int, int]):
        self.line = line_coords
        self.x1, self.y1, self.x2, self.y2 = line_coords
    
    def get_side(self, point: Tuple[int, int]) -> int:
        """Determine which side of the line a point is on."""
        px, py = point
        cross = (self.x2 - self.x1) * (py - self.y1) - (self.y2 - self.y1) * (px - self.x1)
        
        # Reduced threshold from 1.0 to 0.1 for better crossing detection sensitivity
        if cross > 0.1:
            return 1
        elif cross < -0.1:
            return -1
        else:
            return 0
    
    def check_crossing(self, prev_side: int, curr_side: int) -> Optional[str]:
        """Check if a crossing occurred."""
        if prev_side == 0 or curr_side == 0:
            return None
        
        if prev_side < 0 and curr_side > 0:
            return "ENTRY"
        elif prev_side > 0 and curr_side < 0:
            return "EXIT"
        
        return None


class FaceTrack:
    """Represents a tracked face across frames with persistent identity."""
    
    def __init__(self, track_id: int, bbox: np.ndarray, embedding: np.ndarray, name: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.name = name
        self.confidence = confidence
        self.last_seen_frame = 0
        self.detection_count = 0
        self.last_side = None
        self.cooldown = 0
        self.state = "OUTSIDE"  # Track if person is INSIDE or OUTSIDE
        
        # Body tracking (YOLO person detection) - MUST be initialized before get_center() is called
        self.body_bbox = None  # Body bounding box from YOLO
        self.body_centroid = None  # Body centroid (bottom-center)
        self.has_body = False  # Whether body is currently detected
        self.frames_without_body = 0  # Counter for frames without body detection
        
        # Body-only tracking mode (when face is not detected)
        self.body_only_mode = False  # True when tracking body without face
        self.frames_in_body_only_mode = 0  # Counter for body-only tracking
        self.last_body_position = None  # Last known body centroid
        self.body_size_history = deque(maxlen=5)  # Track body bbox sizes for matching
        self.frames_since_face_detected = 0  # Frames since last face detection
        
        # Now safe to call get_center() which depends on has_body
        self.last_position = self.get_center()
        
        # Persistent tracking enhancements
        self.confirmed_name = name  # Once identified, keep this name
        self.best_confidence = confidence  # Best confidence score seen
        self.position_history = deque(maxlen=30)  # Track movement history
        self.embedding_history = deque(maxlen=10)  # Store embeddings for re-identification
        self.frames_since_identified = 0  # Frames since last positive identification
        self.total_identifications = 0  # Total times successfully identified
        self.is_persistent = False  # Whether this track has confirmed identity
        
        # Entry/exit tracking
        self.entry_events = []  # List of (timestamp, confidence)
        self.exit_events = []  # List of (timestamp, confidence)
        self.first_seen_time = datetime.now()
        self.last_event_time = None
        
        # Initialize history
        if embedding is not None:
            self.embedding_history.append(embedding.copy())
        self.position_history.append(self.get_center())
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point - uses body centroid if available, otherwise face center."""
        if self.has_body and self.body_centroid is not None:
            return self.body_centroid
        
        # Fall back to face center
        x1, y1, x2, y2 = self.bbox[:4]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def update(self, bbox: np.ndarray, embedding: np.ndarray, name: str, confidence: float, frame_idx: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.detection_count += 1
        
        # Update position history
        self.last_position = self.get_center()
        self.position_history.append(self.last_position)
        
        # Update embedding history
        if embedding is not None:
            self.embedding = embedding
            self.embedding_history.append(embedding.copy())
            # Face detected - exit body-only mode
            self.body_only_mode = False
            self.frames_in_body_only_mode = 0
            self.frames_since_face_detected = 0
        else:
            # No face detected
            self.frames_since_face_detected += 1
        
        # Handle identity persistence
        if name is not None and confidence > 0:
            # Positive identification
            self.name = name
            self.confidence = confidence
            self.frames_since_identified = 0
            self.total_identifications += 1
            
            # Update confirmed identity if better match or first confirmation
            if not self.is_persistent or confidence > self.best_confidence:
                self.confirmed_name = name
                self.best_confidence = confidence
                self.is_persistent = True
                logging.debug(f"Track {self.track_id}: Confirmed identity as {name} (conf: {confidence:.2f})")
        else:
            # No identification this frame
            self.frames_since_identified += 1
            
            # Keep persistent identity if we had one
            if self.is_persistent and self.frames_since_identified < 60:  # 2 seconds tolerance
                self.name = self.confirmed_name
                self.confidence = self.best_confidence
                logging.debug(f"Track {self.track_id}: Maintaining persistent identity {self.confirmed_name}")
            else:
                self.name = name
                self.confidence = confidence if confidence > 0 else 0.0
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from history for re-identification."""
        if len(self.embedding_history) == 0:
            return None
        
        avg_embedding = np.mean(list(self.embedding_history), axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding
    
    def record_entry(self, timestamp: datetime, confidence: float):
        """Record an entry event."""
        self.entry_events.append((timestamp, confidence))
        self.last_event_time = timestamp
    
    def record_exit(self, timestamp: datetime, confidence: float):
        """Record an exit event."""
        self.exit_events.append((timestamp, confidence))
        self.last_event_time = timestamp
    
    def get_tracking_quality(self) -> float:
        """Calculate tracking quality score (0-1)."""
        if self.detection_count == 0:
            return 0.0
        
        # Factors: identification rate, confidence, detection consistency
        id_rate = self.total_identifications / max(1, self.detection_count)
        conf_score = self.best_confidence
        persistence_score = 1.0 if self.is_persistent else 0.5
        
        quality = (id_rate * 0.4 + conf_score * 0.4 + persistence_score * 0.2)
        return min(1.0, quality)
    
    def update_body(self, body_bbox: Optional[np.ndarray]):
        """Update body tracking information."""
        if body_bbox is not None:
            self.body_bbox = body_bbox
            # Calculate body centroid (bottom-center for better ground position)
            x1, y1, x2, y2 = body_bbox[:4]
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # Use bottom of bbox (feet position)
            self.body_centroid = (cx, cy)
            self.last_body_position = (cx, cy)
            self.has_body = True
            self.frames_without_body = 0
            
            # Track body size for matching
            body_width = x2 - x1
            body_height = y2 - y1
            self.body_size_history.append((body_width, body_height))
        else:
            self.frames_without_body += 1
            # Keep body info for a few frames to handle temporary detection failures
            if self.frames_without_body > 10:
                self.has_body = False
                self.body_bbox = None
                self.body_centroid = None
    
    def update_body_only(self, body_bbox: np.ndarray, frame_idx: int):
        """Update track with body-only detection (no face detected)."""
        self.last_seen_frame = frame_idx
        self.body_only_mode = True
        self.frames_in_body_only_mode += 1
        self.frames_since_face_detected += 1
        
        # Update body information
        self.update_body(body_bbox)
        
        # Update position history with body centroid
        if self.body_centroid:
            self.last_position = self.body_centroid
            self.position_history.append(self.body_centroid)
        
        # Maintain persistent identity if we have one
        if self.is_persistent and self.frames_since_face_detected < 90:  # 3 seconds tolerance
            # Keep the confirmed name and confidence
            self.name = self.confirmed_name
            self.confidence = self.best_confidence
            logging.debug(f"Track {self.track_id}: Body-only mode - maintaining {self.confirmed_name} ({self.frames_since_face_detected} frames without face)")
        else:
            # Lost identity after too long without face
            if self.is_persistent and self.frames_since_face_detected >= 90:
                logging.info(f"Track {self.track_id}: Lost identity {self.confirmed_name} after {self.frames_since_face_detected} frames without face")
                self.is_persistent = False
                self.name = None
                self.confidence = 0.0





class FaceBasedAttendanceSystem:
    """Main face-based attendance system (NO YOLO - direct face detection)."""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize models
        logging.info("Loading face detection and recognition models...")
        self.face_detector = SCRFD(
            model_path=args.face_det_model,
            conf_thres=args.face_conf
        )
        self.face_recognizer = ArcFace(model_path=args.face_rec_model)
        logging.info("✅ Face models loaded")
        
        # Initialize YOLO for body tracking
        self.yolo_model = None
        self.use_body_tracking = USE_BODY_TRACKING and args.use_body_tracking
        
        if self.use_body_tracking:
            try:
                logging.info("Loading YOLO model for body tracking...")
                self.yolo_model = YOLO(args.yolo_model_path)
                logging.info("✅ YOLO model loaded - Body tracking enabled")
            except Exception as e:
                logging.warning(f"⚠️  Failed to load YOLO model: {e}")
                logging.warning("⚠️  Falling back to face-only tracking mode")
                self.use_body_tracking = False
        else:
            logging.info("ℹ️  Body tracking disabled - using face-only mode")
        
        # Initialize known faces
        self.known_faces = KnownFaceLoader(args.known_faces_dir)
        
        # Initialize MEF
        self.mef = MultiEmbeddingFusion()
        
        # Initialize database
        self.db = AttendanceDatabase(args.database_path)
        
        # Line detector (set interactively)
        self.line_detector = None
        
        # Face tracking
        self.tracks = {}  # track_id -> FaceTrack
        self.next_track_id = 1
        
        # Statistics
        self.entry_count = 0
        self.exit_count = 0
        self.frame_idx = 0
    
    def draw_line_interactive(self, cap):
        """Interactive line drawing."""
        print("\n" + "="*60)
        print("DRAW RED LINE FOR ENTRY/EXIT DETECTION")
        print("="*60)
        print("Click 2 points to define the entry/exit line.")
        print("Press 's' to continue after drawing.")
        print("="*60 + "\n")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow("Draw Line")
        cv2.setMouseCallback("Draw Line", mouse_callback)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            display = frame.copy()
            
            cv2.putText(display, "Draw Red Line (2 points)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Points: {len(points)}/2", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            for i, point in enumerate(points):
                cv2.circle(display, point, 8, (0, 0, 255), -1)
                cv2.putText(display, f"P{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 0, 255), 3)
                cv2.putText(display, "Press 's' to continue", 
                           (10, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Draw Line", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(points) == 2:
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyWindow("Draw Line")
        
        if len(points) == 2:
            return (*points[0], *points[1])
        return None
    
    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect all faces in frame and recognize them.
        
        Returns:
            List of (bbox, kps, embedding, name, confidence)
        """
        results = []
        
        try:
            # Detect faces
            bboxes, kpss = self.face_detector.detect(frame, max_num=0)
            
            if bboxes is None or len(bboxes) == 0:
                return results
            
            # Recognize each face
            for bbox, kps in zip(bboxes, kpss):
                # Get embedding
                embedding = self.face_recognizer(frame, kps)
                
                # Identify face
                name, similarity = self.known_faces.identify_face(embedding, self.args.similarity_threshold)
                
                results.append((bbox, kps, embedding, name, similarity))
        
        except Exception as e:
            logging.debug(f"Face detection error: {e}")
        
        return results
    
    def detect_persons(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect all persons in frame using YOLO.
        
        Returns:
            List of body bounding boxes [x1, y1, x2, y2, conf]
        """
        if not self.use_body_tracking or self.yolo_model is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, conf=self.args.yolo_conf, classes=[PERSON_CLASS_ID], verbose=False)
            
            bodies = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        # Get xyxy format
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        # Create bbox array [x1, y1, x2, y2, conf]
                        bbox = np.array([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
                        bodies.append(bbox)
            
            return bodies
        
        except Exception as e:
            logging.debug(f"YOLO detection error: {e}")
            return []
    
    def associate_faces_to_bodies(self, face_detections: List[Tuple], body_bboxes: List[np.ndarray]) -> Dict[int, Optional[np.ndarray]]:
        """
        Associate detected faces with detected bodies using spatial proximity.
        
        Args:
            face_detections: List of (bbox, kps, embedding, name, confidence)
            body_bboxes: List of body bounding boxes from YOLO
        
        Returns:
            Dict mapping face_idx -> body_bbox (or None if no match)
        """
        if not body_bboxes or not face_detections:
            return {i: None for i in range(len(face_detections))}
        
        associations = {}
        used_bodies = set()
        
        # For each face, find the closest body
        for face_idx, (face_bbox, _, _, _, _) in enumerate(face_detections):
            # Get face center
            fx1, fy1, fx2, fy2 = face_bbox[:4]
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2
            
            best_body_idx = None
            best_distance = FACE_TO_BODY_MAX_DISTANCE
            
            for body_idx, body_bbox in enumerate(body_bboxes):
                if body_idx in used_bodies:
                    continue
                
                bx1, by1, bx2, by2 = body_bbox[:4]
                
                # Check if face is inside or near the body bbox
                # Face should be in upper portion of body
                if (bx1 <= face_cx <= bx2 and 
                    by1 <= face_cy <= by1 + (by2 - by1) * 0.7):  # Upper 70% of body (expanded from 60%)
                    
                    # Calculate distance from face center to body center
                    body_cx = (bx1 + bx2) / 2
                    body_cy = (by1 + by2) / 2
                    distance = np.sqrt((face_cx - body_cx)**2 + (face_cy - body_cy)**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_body_idx = body_idx
            
            if best_body_idx is not None:
                associations[face_idx] = body_bboxes[best_body_idx]
                used_bodies.add(best_body_idx)
            else:
                associations[face_idx] = None
        
        return associations
    
    def associate_bodies_to_tracks(self, body_bboxes: List[np.ndarray], used_body_indices: set) -> Dict[int, np.ndarray]:
        """
        Associate unassociated bodies to existing tracks for body-only tracking.
        This is used when a track's face is not detected but their body might still be visible.
        
        Args:
            body_bboxes: List of all detected body bounding boxes
            used_body_indices: Set of body indices already associated with faces
        
        Returns:
            Dict of track_id -> body_bbox for body-only associations
        """
        body_to_track = {}
        
        if not body_bboxes or not self.tracks:
            return body_to_track
        
        # Get unassociated bodies
        unassociated_bodies = []
        for idx, body_bbox in enumerate(body_bboxes):
            if idx not in used_body_indices:
                unassociated_bodies.append((idx, body_bbox))
        
        if not unassociated_bodies:
            return body_to_track
        
        # Get tracks that might need body-only tracking
        # (persistent tracks that haven't been updated with face this frame)
        candidate_tracks = []
        for track_id, track in self.tracks.items():
            # Only consider persistent tracks that weren't just updated
            if (track.is_persistent and 
                self.frame_idx - track.last_seen_frame > 0 and
                self.frame_idx - track.last_seen_frame < 30 and  # Lost within last second
                track.last_body_position is not None):
                candidate_tracks.append((track_id, track))
        
        if not candidate_tracks:
            return body_to_track
        
        # Match bodies to tracks using spatial proximity and size similarity
        for body_idx, body_bbox in unassociated_bodies:
            bx1, by1, bx2, by2 = body_bbox[:4]
            body_cx = (bx1 + bx2) / 2
            body_cy = by2  # Bottom of bbox (feet position)
            body_width = bx2 - bx1
            body_height = by2 - by1
            
            best_track_id = None
            best_score = float('inf')
            
            for track_id, track in candidate_tracks:
                # Skip if track already matched
                if track_id in body_to_track:
                    continue
                
                # Calculate spatial distance from last known body position
                if track.last_body_position:
                    tx, ty = track.last_body_position
                    spatial_dist = np.sqrt((body_cx - tx)**2 + (body_cy - ty)**2)
                    
                    # Calculate size similarity
                    size_similarity = 1.0  # Default if no history
                    if len(track.body_size_history) > 0:
                        avg_width = np.mean([w for w, h in track.body_size_history])
                        avg_height = np.mean([h for w, h in track.body_size_history])
                        
                        width_diff = abs(body_width - avg_width) / max(avg_width, 1)
                        height_diff = abs(body_height - avg_height) / max(avg_height, 1)
                        size_similarity = (width_diff + height_diff) / 2
                    
                    # Combined score (lower is better)
                    # Prioritize spatial proximity (70%) over size similarity (30%)
                    combined_score = 0.7 * spatial_dist + 0.3 * (size_similarity * 100)
                    
                    # Accept match if within reasonable distance and size
                    if spatial_dist < 150 and size_similarity < 0.3 and combined_score < best_score:
                        best_score = combined_score
                        best_track_id = track_id
            
            if best_track_id is not None:
                body_to_track[best_track_id] = body_bbox
                logging.debug(f"Associated body to track {best_track_id} for body-only tracking (score: {best_score:.2f})")
        
        return body_to_track
    
    def associate_detections_to_tracks(self, detections: List[Tuple]) -> Dict[int, Tuple]:

        """
        Associate detected faces to existing tracks or create new ones.
        Uses both spatial proximity and embedding similarity for robust matching.
        
        Returns:
            Dict of track_id -> (bbox, embedding, name, confidence)
        """
        associations = {}
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if len(detections) == 0:
            return associations
        
        # First pass: Spatial matching for recently seen tracks
        if len(self.tracks) > 0:
            det_centers = []
            det_embeddings = []
            for bbox, _, embedding, _, _ in detections:
                x1, y1, x2, y2 = bbox[:4]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                det_centers.append([cx, cy])
                det_embeddings.append(embedding)
            
            track_centers = []
            track_embeddings = []
            track_ids = []
            for tid, track in self.tracks.items():
                # Extended timeout for persistent tracks (60 frames), 30 for others
                timeout = 60 if track.is_persistent else 30
                if self.frame_idx - track.last_seen_frame < timeout:
                    track_centers.append(track.last_position)
                    # Use average embedding for better matching
                    avg_emb = track.get_average_embedding()
                    track_embeddings.append(avg_emb if avg_emb is not None else track.embedding)
                    track_ids.append(tid)
            
            if len(track_ids) > 0:
                det_centers = np.array(det_centers)
                track_centers = np.array(track_centers)
                
                # Compute spatial distances
                spatial_distances = np.linalg.norm(
                    det_centers[:, np.newaxis, :] - track_centers[np.newaxis, :, :],
                    axis=2
                )
                
                # Compute embedding similarities (cosine similarity)
                embedding_similarities = np.zeros((len(det_embeddings), len(track_embeddings)))
                for i, det_emb in enumerate(det_embeddings):
                    if det_emb is not None:
                        det_emb_norm = det_emb.flatten()
                        norm = np.linalg.norm(det_emb_norm)
                        if norm > 0:
                            det_emb_norm = det_emb_norm / norm
                        
                        for j, track_emb in enumerate(track_embeddings):
                            if track_emb is not None:
                                track_emb_norm = track_emb.flatten()
                                t_norm = np.linalg.norm(track_emb_norm)
                                if t_norm > 0:
                                    track_emb_norm = track_emb_norm / t_norm
                                
                                similarity = float(np.dot(det_emb_norm, track_emb_norm))
                                embedding_similarities[i, j] = similarity
                
                # Combined cost: spatial + embedding (lower is better)
                # Convert similarity to distance (1 - similarity)
                embedding_distances = 1.0 - embedding_similarities
                
                # Normalize both to 0-1 range
                if spatial_distances.max() > 0:
                    norm_spatial = spatial_distances / spatial_distances.max()
                else:
                    norm_spatial = spatial_distances
                
                # Combined cost (weighted average)
                combined_cost = 0.6 * norm_spatial + 0.4 * embedding_distances
                
                # Greedy matching with combined cost
                matched_pairs = []
                while combined_cost.size > 0:
                    min_idx = np.unravel_index(combined_cost.argmin(), combined_cost.shape)
                    det_idx, track_idx = min_idx
                    
                    # Check thresholds
                    if (spatial_distances[det_idx, track_idx] > MAX_TRACK_DISTANCE and 
                        embedding_similarities[det_idx, track_idx] < 0.3):
                        # Both spatial and embedding matching failed
                        break
                    
                    det_id = unmatched_detections[det_idx]
                    track_id = track_ids[track_idx]
                    
                    bbox, _, embedding, name, confidence = detections[det_id]
                    
                    # If embedding similarity is high, use it even if spatial distance is large
                    if embedding_similarities[det_idx, track_idx] > 0.6:
                        logging.debug(f"Re-identified track {track_id} by embedding (sim: {embedding_similarities[det_idx, track_idx]:.2f})")
                    
                    associations[track_id] = (bbox, embedding, name, confidence)
                    matched_pairs.append((det_id, track_id))
                    
                    # Remove matched
                    unmatched_detections.remove(det_id)
                    if track_id in unmatched_tracks:
                        unmatched_tracks.remove(track_id)
                    
                    # Remove from cost matrix
                    combined_cost = np.delete(combined_cost, det_idx, axis=0)
                    combined_cost = np.delete(combined_cost, track_idx, axis=1)
                    spatial_distances = np.delete(spatial_distances, det_idx, axis=0)
                    spatial_distances = np.delete(spatial_distances, track_idx, axis=1)
                    embedding_similarities = np.delete(embedding_similarities, det_idx, axis=0)
                    embedding_similarities = np.delete(embedding_similarities, track_idx, axis=1)
                    del track_ids[track_idx]
        
        # Second pass: Try to match unmatched detections with lost tracks using embeddings
        if len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            for det_idx in list(unmatched_detections):
                bbox, _, embedding, name, confidence = detections[det_idx]
                
                if embedding is None or name is None:
                    continue
                
                # Try to find a lost track with same identity
                best_match_track = None
                best_similarity = 0.0
                
                for track_id in unmatched_tracks:
                    track = self.tracks[track_id]
                    
                    # Only consider tracks that were identified and lost recently
                    if (track.is_persistent and 
                        track.confirmed_name == name and 
                        self.frame_idx - track.last_seen_frame < 90):  # 3 seconds
                        
                        # Check embedding similarity
                        avg_emb = track.get_average_embedding()
                        if avg_emb is not None:
                            emb_norm = embedding.flatten()
                            norm = np.linalg.norm(emb_norm)
                            if norm > 0:
                                emb_norm = emb_norm / norm
                            
                            similarity = float(np.dot(emb_norm, avg_emb))
                            
                            if similarity > best_similarity and similarity > 0.5:
                                best_similarity = similarity
                                best_match_track = track_id
                
                if best_match_track is not None:
                    logging.info(f"✨ Re-identified lost track {best_match_track} as {name} (sim: {best_similarity:.2f})")
                    associations[best_match_track] = (bbox, embedding, name, confidence)
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(best_match_track)
        
        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_detections:
            bbox, _, embedding, name, confidence = detections[det_idx]
            
            new_track_id = self.next_track_id
            self.next_track_id += 1
            
            logging.info(f"🆕 Created new track {new_track_id} for {name if name else 'Unknown'}")
            associations[new_track_id] = (bbox, embedding, name, confidence)
        
        return associations
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with face and body tracking."""
        
        # Detect persons (bodies) using YOLO
        body_bboxes = self.detect_persons(frame)
        
        # Detect and recognize faces
        face_detections = self.detect_and_recognize_faces(frame)
        
        # Associate faces to bodies
        face_to_body = self.associate_faces_to_bodies(face_detections, body_bboxes)
        
        # Create enhanced detections with body information
        enhanced_detections = []
        for idx, detection in enumerate(face_detections):
            body_bbox = face_to_body.get(idx)
            enhanced_detections.append((*detection, body_bbox))  # (bbox, kps, embedding, name, conf, body_bbox)
        
        # Associate to tracks
        associations = self.associate_detections_to_tracks(face_detections)
        
        # Update tracks with face and body information
        for track_id, (bbox, embedding, name, confidence) in associations.items():
            # Find corresponding body bbox
            body_bbox = None
            for idx, (det_bbox, _, _, _, _) in enumerate(face_detections):
                if np.array_equal(det_bbox, bbox):
                    body_bbox = face_to_body.get(idx)
                    break
            
            if track_id in self.tracks:
                # Update existing track
                track = self.tracks[track_id]
                track.update(bbox, embedding, name, confidence, self.frame_idx)
                track.update_body(body_bbox)  # Update body tracking
                
                # Debug: Log body tracking status
                if body_bbox is not None:
                    logging.debug(f"Track {track_id} ({name}): Face+Body tracking active, body centroid: {track.body_centroid}")
                else:
                    logging.debug(f"Track {track_id} ({name}): Face-only tracking (no body detected)")

                
                # Add to MEF
                if name is not None:
                    self.mef.add_embedding(track_id, embedding)
                    
                    # Try fusion
                    fused_emb = self.mef.get_fused_embedding(track_id)
                    if fused_emb is not None:
                        name, confidence = self.known_faces.identify_face(fused_emb, self.args.similarity_threshold)
                        if name is not None:
                            track.name = name
                            track.confidence = confidence
            else:
                # Create new track
                self.tracks[track_id] = FaceTrack(track_id, bbox, embedding, name, confidence)
                self.tracks[track_id].last_seen_frame = self.frame_idx
                self.tracks[track_id].detection_count = 1
                self.tracks[track_id].update_body(body_bbox)  # Set initial body
                
                if name is not None:
                    self.mef.add_embedding(track_id, embedding)
        
        # NEW: Body-only tracking for persistent tracks without face detection
        # Track which body indices were already used for face associations
        used_body_indices = set()
        for idx in range(len(face_detections)):
            if face_to_body.get(idx) is not None:
                # Find the index of this body bbox in body_bboxes
                body_bbox = face_to_body[idx]
                for body_idx, bbox in enumerate(body_bboxes):
                    if np.array_equal(bbox, body_bbox):
                        used_body_indices.add(body_idx)
                        break
        
        # Associate remaining bodies to tracks for body-only tracking
        body_only_associations = self.associate_bodies_to_tracks(body_bboxes, used_body_indices)
        
        # Update tracks with body-only information
        for track_id, body_bbox in body_only_associations.items():
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.update_body_only(body_bbox, self.frame_idx)
                logging.info(f"🔄 Track {track_id} ({track.confirmed_name}): Body-only tracking active ({track.frames_since_face_detected} frames without face)")
        
        # Line crossing detection

        if self.line_detector is not None:
            for track_id, track in list(self.tracks.items()):
                # Skip if not seen recently
                if self.frame_idx - track.last_seen_frame > 5:
                    continue
                
                # For persistent tracking: use confirmed_name if available
                person_name = track.confirmed_name if track.is_persistent else track.name
                
                # Skip if not recognized or not enough detections
                # Lowered threshold for persistent tracks and overall minimum
                min_detections = 1 if track.is_persistent else max(1, MIN_TRACK_CONFIDENCE)
                if person_name is None or track.detection_count < min_detections:
                    continue
                
                # Decrement cooldown
                if track.cooldown > 0:
                    track.cooldown -= 1
                
                # Check crossing - use body centroid if available for accurate line crossing
                curr_pos = track.get_center()
                position_source = "BODY" if (track.has_body and track.body_centroid is not None) else "FACE"
                curr_side = self.line_detector.get_side(curr_pos)
                
                # Debug: Log position being used for line crossing
                if track.last_side is not None and track.last_side != curr_side:
                    logging.debug(f"Track {track_id} ({person_name}): Position changed from side {track.last_side} to {curr_side} using {position_source} at {curr_pos}")
                
                if track.last_side is not None and track.cooldown == 0:
                    event = self.line_detector.check_crossing(track.last_side, curr_side)
                    
                    # STATE MACHINE: Only record valid state transitions
                    if event == "ENTRY" and track.state == "OUTSIDE":
                        # Person was outside, now entering
                        timestamp = datetime.now()
                        self.db.record_entry(person_name, track.confidence, track_id)
                        track.record_entry(timestamp, track.confidence)
                        self.entry_count += 1
                        track.state = "INSIDE"  # Update state
                        track.cooldown = COOLDOWN_FRAMES
                        logging.info(f"✅ {person_name} ENTERED (ID:{track_id}, Position: {position_source} {curr_pos}, Quality:{track.get_tracking_quality():.2f})")
                    
                    elif event == "ENTRY" and track.state == "INSIDE":
                        # Person already inside, ignore duplicate entry
                        logging.debug(f"⚠️  {person_name} already INSIDE - ignoring duplicate entry")
                    
                    elif event == "EXIT" and track.state == "INSIDE":
                        # Person was inside, now exiting
                        timestamp = datetime.now()
                        self.db.record_exit(person_name, track.confidence, track_id)
                        track.record_exit(timestamp, track.confidence)
                        self.exit_count += 1
                        track.state = "OUTSIDE"  # Update state
                        track.cooldown = COOLDOWN_FRAMES
                        
                        # Calculate duration inside
                        if len(track.entry_events) > 0:
                            last_entry_time = track.entry_events[-1][0]
                            duration = (timestamp - last_entry_time).total_seconds()
                            logging.info(f"❌ {person_name} EXITED (ID:{track_id}, Position: {position_source} {curr_pos}, Duration: {duration:.1f}s, Quality:{track.get_tracking_quality():.2f})")
                        else:
                            logging.info(f"❌ {person_name} EXITED (ID:{track_id}, Position: {position_source} {curr_pos}, Quality:{track.get_tracking_quality():.2f})")
                    
                    elif event == "EXIT" and track.state == "OUTSIDE":
                        # Person already outside, ignore exit without entry
                        logging.debug(f"⚠️  {person_name} already OUTSIDE - ignoring exit without entry")
                
                track.last_side = curr_side
        
        # Draw on frame
        for track_id, track in self.tracks.items():
            if self.frame_idx - track.last_seen_frame > 5:
                continue
            
            # Face bbox
            x1, y1, x2, y2 = map(int, track.bbox[:4])
            
            # Color based on recognition and persistence
            person_name = track.confirmed_name if track.is_persistent else track.name
            
            if person_name is not None:
                if track.is_persistent:
                    color = (0, 255, 0)  # Green for persistent tracks
                    persistence_marker = "★"
                else:
                    color = (0, 200, 200)  # Yellow-green for new identifications
                    persistence_marker = ""
                
                # Show state
                state_marker = "🏠" if track.state == "INSIDE" else "🚪"
                
                # Add body tracking indicator with mode
                if track.body_only_mode:
                    tracking_mode = "+BODY"  # Body-only tracking
                    # Add indicator for how long without face
                    if track.frames_since_face_detected > 0:
                        tracking_mode += f"({track.frames_since_face_detected}f)"
                elif track.has_body:
                    tracking_mode = "+FACE+BODY"  # Full tracking
                else:
                    tracking_mode = "+FACE"  # Face-only tracking
                
                label = f"{persistence_marker}ID:{track_id} {person_name} {tracking_mode} ({track.confidence:.2f})"
                quality = track.get_tracking_quality()

                
                # Draw quality bar
                bar_width = x2 - x1
                bar_height = 4
                quality_width = int(bar_width * quality)
                cv2.rectangle(frame, (x1, y2 + 2), (x1 + quality_width, y2 + 2 + bar_height), 
                            (0, 255, 0), -1)
                cv2.rectangle(frame, (x1 + quality_width, y2 + 2), (x2, y2 + 2 + bar_height), 
                            (128, 128, 128), -1)
            else:
                color = (0, 165, 255)  # Orange for unknown
                tracking_mode = "+BODY" if track.has_body else ""
                label = f"ID:{track_id} Unknown{tracking_mode}"
                state_marker = ""
            
            # Draw body bbox first (larger, behind face box)
            if track.has_body and track.body_bbox is not None:
                bx1, by1, bx2, by2 = map(int, track.body_bbox[:4])
                body_color = (255, 200, 0)  # Cyan for body
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), body_color, 2)
                
                # Draw body centroid
                if track.body_centroid:
                    cv2.circle(frame, track.body_centroid, 6, body_color, -1)
                    cv2.circle(frame, track.body_centroid, 8, body_color, 2)
            
            # Draw face bbox on top
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw state marker
            if state_marker:
                cv2.putText(frame, state_marker, (x2 - 25, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point (uses body centroid if available)
            cx, cy = track.get_center()
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # Draw movement trail
            if len(track.position_history) > 1:
                points = list(track.position_history)
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], color, 1)
        
        # Clean up old tracks
        old_tracks = [tid for tid, track in self.tracks.items() 
                     if self.frame_idx - track.last_seen_frame > 60]
        for tid in old_tracks:
            del self.tracks[tid]
            self.mef.clear_track(tid)
        
        # Draw red line
        if self.line_detector is not None:
            cv2.line(frame,
                    (self.line_detector.x1, self.line_detector.y1),
                    (self.line_detector.x2, self.line_detector.y2),
                    (0, 0, 255), 3)
        
        # Draw statistics
        cv2.putText(frame, f"Entry: {self.entry_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {self.exit_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Faces: {len([t for t in self.tracks.values() if self.frame_idx - t.last_seen_frame <= 5])}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main processing loop."""
        # Open video source
        source = self.args.source
        try:
            if source.isdigit():
                source = int(source)
        except AttributeError:
            pass
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logging.error(f"Failed to open video source: {source}")
            if source == 0 or source == "0":
                logging.error("Webcam not available. Try using a video file:")
                logging.error("  python entry_exit_attendance.py --source /path/to/video.mp4")
            else:
                logging.error(f"Video file not found: {source}")
            return
        
        # Draw line
        line_coords = self.draw_line_interactive(cap)
        if line_coords is None:
            logging.error("Line drawing cancelled")
            return
        
        self.line_detector = LineCrossDetector(line_coords)
        logging.info(f"Red line set at: {line_coords}")
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        writer = None
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))
            logging.info(f"Saving output to: {self.args.output}")
        
        logging.info("Starting face-based attendance tracking... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                
                if writer is not None:
                    writer.write(processed_frame)
                
                if not self.args.skip_display:
                    cv2.imshow("Face-Based Attendance", processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                self.frame_idx += 1
                
                if self.frame_idx % 100 == 0:
                    logging.info(f"Frame {self.frame_idx} - Entry: {self.entry_count}, Exit: {self.exit_count}")
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            self.db.close()
            
            # Print detailed tracking summary
            self.print_tracking_summary()
    
    def get_tracking_statistics(self) -> Dict:
        """Get detailed tracking statistics for all persons."""
        stats = {
            'total_tracks': len(self.tracks),
            'persistent_tracks': 0,
            'persons': {}
        }
        
        for track_id, track in self.tracks.items():
            if track.is_persistent:
                stats['persistent_tracks'] += 1
                
                person_name = track.confirmed_name
                if person_name not in stats['persons']:
                    stats['persons'][person_name] = {
                        'track_ids': [],
                        'total_entries': 0,
                        'total_exits': 0,
                        'total_duration': 0.0,
                        'current_state': track.state,
                        'best_quality': 0.0
                    }
                
                person_stats = stats['persons'][person_name]
                person_stats['track_ids'].append(track_id)
                person_stats['total_entries'] += len(track.entry_events)
                person_stats['total_exits'] += len(track.exit_events)
                
                # Calculate total duration
                for i, (entry_time, _) in enumerate(track.entry_events):
                    if i < len(track.exit_events):
                        exit_time, _ = track.exit_events[i]
                        duration = (exit_time - entry_time).total_seconds()
                        person_stats['total_duration'] += duration
                
                # Update quality
                quality = track.get_tracking_quality()
                if quality > person_stats['best_quality']:
                    person_stats['best_quality'] = quality
                    person_stats['current_state'] = track.state
        
        return stats
    
    def print_tracking_summary(self):
        """Print detailed tracking summary."""
        stats = self.get_tracking_statistics()
        
        logging.info("\n" + "=" * 60)
        logging.info("DETAILED TRACKING SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total Tracks Created: {self.next_track_id - 1}")
        logging.info(f"Active Tracks: {stats['total_tracks']}")
        logging.info(f"Persistent Tracks: {stats['persistent_tracks']}")
        logging.info(f"Total Entries: {self.entry_count}")
        logging.info(f"Total Exits: {self.exit_count}")
        logging.info(f"Total Frames: {self.frame_idx}")
        logging.info("-" * 60)
        
        if stats['persons']:
            logging.info("PER-PERSON STATISTICS:")
            logging.info("-" * 60)
            for person_name, person_stats in sorted(stats['persons'].items()):
                logging.info(f"\n👤 {person_name}:")
                logging.info(f"   Track IDs: {person_stats['track_ids']}")
                logging.info(f"   Entries: {person_stats['total_entries']}")
                logging.info(f"   Exits: {person_stats['total_exits']}")
                if person_stats['total_duration'] > 0:
                    logging.info(f"   Total Time Inside: {person_stats['total_duration']:.1f}s")
                logging.info(f"   Current State: {person_stats['current_state']}")
                logging.info(f"   Tracking Quality: {person_stats['best_quality']:.2f}")
        
        logging.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face-Based Entry/Exit Attendance System with Body Tracking")
    
    parser.add_argument("--source", type=str, default="0",
                       help="Video source: 0 for webcam or video file path")
    parser.add_argument("--face-det-model", type=str, default=FACE_DETECTION_MODEL,
                       help="Path to face detection model (SCRFD)")
    parser.add_argument("--face-rec-model", type=str, default=FACE_RECOGNITION_MODEL,
                       help="Path to face recognition model (ArcFace)")
    parser.add_argument("--yolo-model-path", type=str, default=YOLO_MODEL_PATH,
                       help="Path to YOLO model for person detection")
    parser.add_argument("--known-faces-dir", type=str, default=KNOWN_FACES_DIR,
                       help="Directory with known face embeddings")
    parser.add_argument("--database-path", type=str, default=DATABASE_PATH,
                       help="Path to SQLite database")
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE,
                       help="Face detection confidence threshold")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONFIDENCE,
                       help="YOLO person detection confidence threshold")
    parser.add_argument("--similarity-threshold", type=float, default=FACE_SIMILARITY_THRESHOLD,
                       help="Face similarity threshold")
    parser.add_argument("--use-body-tracking", action="store_true", default=USE_BODY_TRACKING,
                       help="Enable YOLO body tracking (default: enabled)")
    parser.add_argument("--no-body-tracking", dest="use_body_tracking", action="store_false",
                       help="Disable YOLO body tracking (face-only mode)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path (optional)")
    parser.add_argument("--skip-display", action="store_true",
                       help="Skip display (headless mode)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    system = FaceBasedAttendanceSystem(args)
    system.run()


if __name__ == "__main__":
    main()
