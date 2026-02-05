"""
Entry/Exit Attendance System - SCRFD -> DeepSORT -> ArcFace
===========================================================
Pipeline per frame:
1) SCRFD for face detection and landmarks
2) DeepSORT assigns track IDs using face embeddings
3) ArcFace recognition per track (MEF fused)

Notes:
- No YOLO body detector is used.
- Once a person name is matched, the same name and stable ID are preserved.
- A heuristic "full body" box is drawn by expanding the face box.
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

from models.SCRFD import SCRFD
from models.ArcFace import ArcFace
from utils.deep_sort_tracker import DeepSORTTracker

# Check GPU availability
try:
    import onnxruntime as ort
    GPU_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()
except:
    GPU_AVAILABLE = False

# =====================
# CONFIGURATION
# =====================

# Model paths
FACE_DETECTION_MODEL = "./weights/face_detection/det_10g.onnx"
FACE_RECOGNITION_MODEL = "./weights/face_recognition/w600k_r50.onnx"
KNOWN_FACES_DIR = "./known_faces"

# Detection parameters
FACE_CONFIDENCE = 0.5  # Lower threshold for better detection
FACE_SIMILARITY_THRESHOLD = 0.40  # Lower threshold for easier matching

# Line crossing parameters
MIN_MOVEMENT = 3.0  # Minimum pixels to consider movement
COOLDOWN_FRAMES = 30  # Frames to wait before allowing another entry/exit for same person

# Multi-Embedding Fusion parameters
MEF_BUFFER_SIZE = 5  # Number of embeddings to keep for fusion
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]  # Weights for recent embeddings

# Tracking parameters
MAX_TRACK_DISTANCE = 100  # Maximum distance to associate faces across frames
MIN_TRACK_CONFIDENCE = 2  # Minimum detections before recording attendance (increased for stability)
MIN_FACE_SIZE = 20  # Minimum face box size in pixels (lower for distant faces)
ACTIVE_TRACK_GRACE = 10  # Frames to keep showing/using recent tracks (unknowns) - reduced to remove ghosts
RECOGNIZED_TRACK_GRACE = 60  # Longer grace for recognized names (face may disappear) - reduced
TRACKER_MAX_AGE = 30  # DeepSORT prediction window without detections - reduced to remove ghosts faster

# Full-body box heuristic (face-only mode)
BODY_WIDTH_FACTOR = 2.0
BODY_HEIGHT_FACTOR = 5.0
BODY_TOP_OFFSET = 0.2  # Extend a bit above face for head

# Performance optimization
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for maximum accuracy (1=all frames, 2=every other frame)
RESIZE_WIDTH = 800  # Resize frame width for processing (None=no resize, 800=good balance)
MAX_FRAME_WIDTH = 1280  # Maximum width before forcing resize

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
        
        # Use smaller threshold for more sensitive detection
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


class FaceBasedAttendanceSystem:
    """Main face-based attendance system (NO YOLO - direct face detection)."""
    
    def __init__(self, args):
        self.args = args
        
        # Performance settings
        self.process_every_n = getattr(args, 'process_every_n', PROCESS_EVERY_N_FRAMES)
        self.resize_width = getattr(args, 'resize_width', RESIZE_WIDTH)
        
        # Initialize models
        logging.info("Loading face detection and recognition models...")
        
        # Log GPU status
        if GPU_AVAILABLE:
            logging.info("⚡ GPU: CUDA available - using GPU acceleration")
        else:
            logging.warning("⚠️  GPU: Not available - using CPU (slower)")
            logging.warning("   To enable GPU: pip uninstall onnxruntime && pip install onnxruntime-gpu")
        
        self.face_detector = SCRFD(
            model_path=args.face_det_model,
            conf_thres=args.face_conf
        )
        self.face_recognizer = ArcFace(model_path=args.face_rec_model)
        logging.info("✅ Models loaded")
        logging.info(f"⚡ Performance: Processing every {self.process_every_n} frame(s), resize width: {self.resize_width}")
        
        # Initialize known faces
        self.known_faces = KnownFaceLoader(args.known_faces_dir)
        
        # Initialize MEF
        self.mef = MultiEmbeddingFusion()
        
        # Initialize database
        self.db = AttendanceDatabase(args.database_path)
        
        # Line detector (set interactively)
        self.line_detector = None
        
        # Initialize DeepSORT tracker for face tracking
        logging.info("Initializing DeepSORT tracker for face tracking...")
        self.tracker = DeepSORTTracker(
            max_age=TRACKER_MAX_AGE,  # Keep tracks alive without detections
            min_hits=2,  # Require 2 detections before confirming track (reduces ghosts)
            iou_threshold=0.3,
            embedding_distance_threshold=0.6,
            embedding_weight=0.8  # Stronger appearance matching
        )
        logging.info("✅ DeepSORT tracker initialized")
        
        # Track identity mapping (DeepSORT track_id -> person info)
        self.track_identities = {}  # track_id -> {name, confidence, last_seen, etc.}
        
        # Persistent identity mapping (once recognized, always keep same name + stable ID)
        self.name_to_stable_id = {}  # person_name -> stable_id
        self.track_to_stable_id = {}  # track_id -> stable_id
        self.stable_id_active_track = {}  # stable_id -> track_id
        self._stable_id_counter = 1
        
        # Line crossing state
        self.track_line_state = {}  # track_id -> {last_side, cooldown, state}
        
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
        Detect all faces in frame and compute embeddings.
        
        Returns:
            List of (bbox, kps, embedding)
        """
        results = []
        
        try:
            # Detect faces
            bboxes, kpss = self.face_detector.detect(frame, max_num=0)
            
            if bboxes is None or len(bboxes) == 0:
                return results
            
            # Embed each face (used for tracking and recognition later)
            for bbox, kps in zip(bboxes, kpss):
                x1, y1, x2, y2, score = bbox[:5]
                if score < self.args.face_conf:
                    continue
                w = x2 - x1
                h = y2 - y1
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue
                if x2 <= 0 or y2 <= 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                    continue
                embedding = self.face_recognizer(frame, kps)
                results.append((bbox, kps, embedding))
        
        except Exception as e:
            logging.debug(f"Face detection error: {e}")
        
        return results
    
    def process_frame(self, frame: np.ndarray, skip_detection: bool = False) -> np.ndarray:
        """Process a single frame using DeepSORT tracking."""
        
        # Log frame processing every 30 frames
        if self.frame_idx % 30 == 0:
            logging.info(f"\n{'='*60}\nProcessing Frame {self.frame_idx}\n{'='*60}")
        
        # Resize frame for faster processing if needed
        original_frame = frame
        scale_factor = 1.0
        
        if self.resize_width is not None and frame.shape[1] > self.resize_width:
            scale_factor = self.resize_width / frame.shape[1]
            new_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (self.resize_width, new_height))
        
        # Step 1: Detect faces and embeddings (process every frame for accuracy)
        face_detections = self.detect_and_recognize_faces(frame)
        
        # Step 2: Prepare detections for DeepSORT (bbox + embedding)
        deepsort_detections = []
        face_embeddings = {}  # index -> embedding
        
        for idx, (bbox, kps, embedding) in enumerate(face_detections):
            # DeepSORT expects (bbox, embedding) tuples
            deepsort_detections.append((bbox, embedding))
            face_embeddings[idx] = embedding
        
        # Step 3: Update DeepSORT tracker
        tracked_faces = self.tracker.update(deepsort_detections)
        
        # Log detection summary
        if len(face_detections) > 0 or len(tracked_faces) > 0:
            logging.info(f"Frame {self.frame_idx}: Detected {len(face_detections)} faces, Tracking {len(tracked_faces)} faces")
        
        # Step 4: Process tracked faces and assign identities
        for track_id, bbox in tracked_faces:
            # Find which detection matches this track (by bbox proximity)
            best_match_idx = None
            best_iou = 0.0
            
            for idx, (det_bbox, _, _) in enumerate(face_detections):
                iou = self._compute_iou(bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            # Get or create track identity
            if track_id not in self.track_identities:
                self.track_identities[track_id] = {
                    'name': None,
                    'confidence': 0.0,
                    'last_seen': self.frame_idx,
                    'detection_count': 0,
                    'bbox': bbox,
                    'body_bbox': None,
                    'stable_id': None,
                    'active': True
                }
            
            track_info = self.track_identities[track_id]
            track_info['last_seen'] = self.frame_idx
            track_info['detection_count'] += 1
            track_info['bbox'] = bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            track_info['body_bbox'] = self._estimate_body_bbox((x1, y1, x2, y2), frame.shape)
            
            # Update identity if we have a face match
            if best_match_idx is not None and best_iou > 0.3:
                embedding = face_embeddings[best_match_idx]
                
                # Add to MEF for better recognition
                if embedding is not None:
                    self.mef.add_embedding(track_id, embedding)
                
                # PERSISTENT IDENTITY LOGIC (name and stable ID never change once set)
                if track_info['name'] is not None:
                    pass
                else:
                    fused_emb = self.mef.get_fused_embedding(track_id)
                    if fused_emb is not None:
                        name, confidence = self.known_faces.identify_face(
                            fused_emb, self.args.similarity_threshold
                        )
                        if name is not None:
                            if name not in self.name_to_stable_id:
                                self.name_to_stable_id[name] = self._stable_id_counter
                                self._stable_id_counter += 1
                                logging.info(f"✨ New person tracked: {name} assigned Stable ID {self.name_to_stable_id[name]}")
                            stable_id = self.name_to_stable_id[name]
                            # If this stable_id is already active on another track, retire the old one
                            prev_track_id = self.stable_id_active_track.get(stable_id)
                            if prev_track_id is not None and prev_track_id != track_id:
                                prev_info = self.track_identities.get(prev_track_id)
                                if prev_info is not None:
                                    prev_info['active'] = False
                                    if stable_id in self.track_line_state:
                                        del self.track_line_state[stable_id]
                            self.track_to_stable_id[track_id] = stable_id
                            self.stable_id_active_track[stable_id] = track_id
                            track_info['name'] = name
                            track_info['confidence'] = confidence
                            track_info['stable_id'] = stable_id
                            logging.info(f"✨ Recognized: {name} (Stable ID:{stable_id}, Confidence:{confidence:.2f})")
        
        # Step 5: Line crossing detection
        if self.line_detector is not None:
            for track_id, track_info in list(self.track_identities.items()):
                # Skip if not seen recently (allow longer grace for recognized names)
                grace = RECOGNIZED_TRACK_GRACE if track_info['name'] is not None else ACTIVE_TRACK_GRACE
                if self.frame_idx - track_info['last_seen'] > grace:
                    continue
                
                # Skip inactive tracks
                if not track_info.get('active', True):
                    continue

                # Skip if not recognized or not enough detections
                if track_info['name'] is None:
                    continue
                    
                if track_info['detection_count'] < MIN_TRACK_CONFIDENCE:
                    logging.info(f"⏭️  Skipping {track_info['name']} (ID:{track_id}) - detection count {track_info['detection_count']} < {MIN_TRACK_CONFIDENCE}")
                    continue
                
                stable_id = track_info['stable_id']
                if stable_id is None:
                    logging.info(f"⏭️  Skipping {track_info['name']} (ID:{track_id}) - no stable ID")
                    continue

                # Require recent detection to avoid false crossings from predictions (relaxed from 3 to 5 frames)
                frames_since_seen = self.frame_idx - track_info['last_seen']
                if frames_since_seen > 5:
                    logging.info(f"⏭️  Skipping {track_info['name']} (ID:{stable_id}) - last seen {frames_since_seen} frames ago")
                    continue
                
                # Get current position (bottom center of body bbox if available)
                body_bbox = track_info.get('body_bbox')
                if body_bbox is not None:
                    bx1, by1, bx2, by2 = body_bbox
                    curr_pos = (int((bx1 + bx2) / 2), int(by2))
                else:
                    bbox = track_info['bbox']
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    curr_pos = (cx, cy)
                
                # Check which side of line
                curr_side = self.line_detector.get_side(curr_pos)
                
                # Log position and side for every tracked person
                logging.info(f"📍 {track_info['name']} (ID:{stable_id}) - Position: {curr_pos}, Side: {curr_side}")
                
                # Initialize line crossing state if needed
                if stable_id not in self.track_line_state:
                    # Determine initial state based on current position
                    initial_state = 'INSIDE' if curr_side > 0 else 'OUTSIDE'
                    
                    # CRITICAL FIX: Don't initialize with side=0, use a valid side
                    initial_side = curr_side if curr_side != 0 else (1 if initial_state == 'INSIDE' else -1)
                    
                    self.track_line_state[stable_id] = {
                        'last_side': initial_side,  # Use valid side, never 0
                        'cooldown': 0,
                        'state': initial_state
                    }
                    logging.info(f"🆕 Initialized {track_info['name']} (ID:{stable_id}) state: {initial_state}, side: {initial_side}, pos: {curr_pos}")
                
                line_state = self.track_line_state[stable_id]
                
                if line_state['cooldown'] > 0:
                    line_state['cooldown'] -= 1
                    if line_state['cooldown'] % 10 == 0:  # Log every 10 frames during cooldown
                        logging.info(f"⏳ {track_info['name']} (ID:{stable_id}) - Cooldown: {line_state['cooldown']} frames")
                
                if line_state['last_side'] is not None and line_state['cooldown'] == 0:
                    event = self.line_detector.check_crossing(line_state['last_side'], curr_side)
                    
                    if event is not None:
                        logging.info(f"🔍 {track_info['name']} (ID:{stable_id}) - Event: {event}, State: {line_state['state']}, Side: {line_state['last_side']}→{curr_side}, Pos: {curr_pos}")
                    
                    # STATE MACHINE: Only record valid state transitions
                    if event == "ENTRY" and line_state['state'] == "OUTSIDE":
                        # Person was outside, now entering
                        self.db.record_entry(track_info['name'], track_info['confidence'], stable_id)
                        self.entry_count += 1
                        line_state['state'] = "INSIDE"
                        line_state['cooldown'] = COOLDOWN_FRAMES
                        logging.info(f"✅ {track_info['name']} ENTERED (Stable ID:{stable_id})")
                    
                    elif event == "ENTRY" and line_state['state'] == "INSIDE":
                        logging.info(f"⚠️  {track_info['name']} already INSIDE - ignoring duplicate entry (Side: {line_state['last_side']}→{curr_side})")
                    
                    elif event == "EXIT" and line_state['state'] == "INSIDE":
                        # Person was inside, now exiting
                        self.db.record_exit(track_info['name'], track_info['confidence'], stable_id)
                        self.exit_count += 1
                        line_state['state'] = "OUTSIDE"
                        line_state['cooldown'] = COOLDOWN_FRAMES
                        logging.info(f"❌ {track_info['name']} EXITED (Stable ID:{stable_id})")
                    
                    elif event == "EXIT" and line_state['state'] == "OUTSIDE":
                        logging.info(f"⚠️  {track_info['name']} already OUTSIDE - ignoring exit without entry (Side: {line_state['last_side']}→{curr_side})")
                
                elif line_state['cooldown'] > 0:
                    logging.info(f"⏳ {track_info['name']} (ID:{stable_id}) - In cooldown ({line_state['cooldown']} frames remaining)")
                else:
                    logging.info(f"⚠️  {track_info['name']} (ID:{stable_id}) - No event detected (prev_side={line_state.get('last_side')}, curr_side={curr_side})")
                
                if curr_side != 0:
                    line_state['last_side'] = curr_side
                else:
                    logging.debug(f"⚖️  {track_info['name']} (ID:{stable_id}) - On line (Side 0), keeping last_side={line_state.get('last_side')}")
        
        # Scale coordinates back to original frame if resized
        if scale_factor != 1.0:
            frame = original_frame
        
        # Step 6: Draw visualizations
        for track_id, track_info in self.track_identities.items():
            grace = RECOGNIZED_TRACK_GRACE if track_info['name'] is not None else ACTIVE_TRACK_GRACE
            if self.frame_idx - track_info['last_seen'] > grace:
                continue
            if not track_info.get('active', True):
                continue
            
            bbox = track_info['bbox']
            # Scale bbox back to original size if frame was resized
            if scale_factor != 1.0:
                bbox = bbox.copy()
                bbox[:4] = bbox[:4] / scale_factor
            x1, y1, x2, y2 = map(int, bbox[:4])
            body_bbox = track_info.get('body_bbox')
            if body_bbox is not None and scale_factor != 1.0:
                body_bbox = tuple(int(coord / scale_factor) for coord in body_bbox)
            
            # Color based on recognition
            if track_info['name'] is not None:
                color = (0, 255, 0)  # Green
                stable_id = track_info.get('stable_id')
                label = f"ID:{stable_id} {track_info['name']} ({track_info['confidence']:.2f})"
            else:
                color = (0, 165, 255)  # Orange
                label = f"ID:{track_id} Unknown"
            
            if track_info['name'] is not None and body_bbox is not None:
                bx1, by1, bx2, by2 = map(int, body_bbox)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                cv2.putText(frame, label, (bx1, max(10, by1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
        # Step 7: Clean up old track identities
        old_tracks = []
        for track_id, track_info in self.track_identities.items():
            frames_since_seen = self.frame_idx - track_info['last_seen']
            
            # Keep recognized persons for longer, unknown for shorter
            max_age = 1800 if track_info['name'] is not None else 120
            
            if frames_since_seen > max_age:
                old_tracks.append(track_id)
        
        for track_id in old_tracks:
            track_info = self.track_identities[track_id]
            if track_info['name'] is not None:
                logging.info(f"🗑️  Removing old track {track_id} for {track_info['name']} (not seen for {self.frame_idx - track_info['last_seen']} frames)")
                stable_id = track_info.get('stable_id')
                if stable_id is not None and self.stable_id_active_track.get(stable_id) == track_id:
                    del self.stable_id_active_track[stable_id]
            del self.track_identities[track_id]
            self.mef.clear_track(track_id)
            # Note: We keep stable_id mapping permanently for re-acquisition
        
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
        active_tracks = len([
            t for t in self.track_identities.values()
            if self.frame_idx - t['last_seen'] <= (
                RECOGNIZED_TRACK_GRACE if t['name'] is not None else ACTIVE_TRACK_GRACE
            )
        ])
        cv2.putText(frame, f"Faces: {active_tracks}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
    
    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union

    @staticmethod
    def _estimate_body_bbox(face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Estimate a full-body box from a face box (heuristic, face-only mode)."""
        x1, y1, x2, y2 = face_bbox
        face_w = max(x2 - x1, 1)
        face_h = max(y2 - y1, 1)
        cx = (x1 + x2) / 2.0

        body_w = face_w * BODY_WIDTH_FACTOR
        body_h = face_h * BODY_HEIGHT_FACTOR
        body_x1 = int(cx - body_w / 2)
        body_y1 = int(y1 - face_h * BODY_TOP_OFFSET)
        body_x2 = int(cx + body_w / 2)
        body_y2 = int(body_y1 + body_h)

        h, w = frame_shape[:2]
        body_x1 = max(0, min(body_x1, w - 1))
        body_y1 = max(0, min(body_y1, h - 1))
        body_x2 = max(0, min(body_x2, w - 1))
        body_y2 = max(0, min(body_y2, h - 1))

        if body_x2 <= body_x1 or body_y2 <= body_y1:
            return None
        return (body_x1, body_y1, body_x2, body_y2)
    
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
                    if isinstance(source, str) and self.args.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                # Process every frame for maximum accuracy
                processed_frame = self.process_frame(frame, skip_detection=False)
                
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
            
            logging.info("\n" + "="*60)
            logging.info("ATTENDANCE SUMMARY")
            logging.info("="*60)
            logging.info(f"Total Entries: {self.entry_count}")
            logging.info(f"Total Exits: {self.exit_count}")
            logging.info(f"Total Frames: {self.frame_idx}")
            logging.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Face-Based Entry/Exit Attendance System (No YOLO)")
    
    parser.add_argument("--source", type=str, default="0",
                       help="Video source: 0 for webcam or video file path")
    parser.add_argument("--face-det-model", type=str, default=FACE_DETECTION_MODEL,
                       help="Path to face detection model (SCRFD)")
    parser.add_argument("--face-rec-model", type=str, default=FACE_RECOGNITION_MODEL,
                       help="Path to face recognition model (ArcFace)")
    parser.add_argument("--known-faces-dir", type=str, default=KNOWN_FACES_DIR,
                       help="Directory with known face embeddings")
    parser.add_argument("--database-path", type=str, default=DATABASE_PATH,
                       help="Path to SQLite database")
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE,
                       help="Face detection confidence threshold")
    parser.add_argument("--similarity-threshold", type=float, default=FACE_SIMILARITY_THRESHOLD,
                       help="Face similarity threshold")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path (optional)")
    parser.add_argument("--skip-display", action="store_true",
                       help="Skip display (headless mode)")
    parser.add_argument("--loop-video", action="store_true",
                       help="Loop video file instead of stopping at EOF")
    parser.add_argument("--process-every-n", type=int, default=PROCESS_EVERY_N_FRAMES,
                       help=f"Process every Nth frame (default: {PROCESS_EVERY_N_FRAMES})")
    parser.add_argument("--resize-width", type=int, default=RESIZE_WIDTH,
                       help=f"Resize frame width for processing (default: {RESIZE_WIDTH}, use 0 for no resize)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    system = FaceBasedAttendanceSystem(args)
    system.run()


if __name__ == "__main__":
    main()
