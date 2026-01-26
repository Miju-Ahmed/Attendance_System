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

from models.SCRFD import SCRFD
from models.ArcFace import ArcFace

# =====================
# CONFIGURATION
# =====================

# Model paths
FACE_DETECTION_MODEL = "./weights/face_detection/det_10g.onnx"
FACE_RECOGNITION_MODEL = "./weights/face_recognition/w600k_r50.onnx"
KNOWN_FACES_DIR = "./known_faces"

# Detection parameters
FACE_CONFIDENCE = 0.5
FACE_SIMILARITY_THRESHOLD = 0.45  # Threshold for face matching

# Line crossing parameters
MIN_MOVEMENT = 3.0  # Minimum pixels to consider movement
COOLDOWN_FRAMES = 30  # Frames to wait before allowing another entry/exit for same person

# Multi-Embedding Fusion parameters
MEF_BUFFER_SIZE = 5  # Number of embeddings to keep for fusion
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]  # Weights for recent embeddings

# Tracking parameters
MAX_TRACK_DISTANCE = 100  # Maximum distance to associate faces across frames
MIN_TRACK_CONFIDENCE = 3  # Minimum detections before recording attendance

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
        
        if cross > 1.0:
            return 1
        elif cross < -1.0:
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
    """Represents a tracked face across frames."""
    
    def __init__(self, track_id: int, bbox: np.ndarray, embedding: np.ndarray, name: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.name = name
        self.confidence = confidence
        self.last_seen_frame = 0
        self.detection_count = 0
        self.last_position = self.get_center()
        self.last_side = None
        self.cooldown = 0
        self.state = "OUTSIDE"  # Track if person is INSIDE or OUTSIDE
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of face bbox."""
        x1, y1, x2, y2 = self.bbox[:4]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def update(self, bbox: np.ndarray, embedding: np.ndarray, name: str, confidence: float, frame_idx: int):
        """Update track with new detection."""
        self.bbox = bbox
        self.embedding = embedding
        self.name = name
        self.confidence = confidence
        self.last_seen_frame = frame_idx
        self.detection_count += 1
        self.last_position = self.get_center()


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
        logging.info("✅ Models loaded")
        
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
    
    def associate_detections_to_tracks(self, detections: List[Tuple]) -> Dict[int, Tuple]:
        """
        Associate detected faces to existing tracks or create new ones.
        
        Returns:
            Dict of track_id -> (bbox, embedding, name, confidence)
        """
        associations = {}
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if len(detections) == 0:
            return associations
        
        # Compute distance matrix
        if len(self.tracks) > 0:
            det_centers = []
            for bbox, _, _, _, _ in detections:
                x1, y1, x2, y2 = bbox[:4]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                det_centers.append([cx, cy])
            
            track_centers = []
            track_ids = []
            for tid, track in self.tracks.items():
                if self.frame_idx - track.last_seen_frame < 30:  # Only recent tracks
                    track_centers.append(track.last_position)
                    track_ids.append(tid)
            
            if len(track_ids) > 0:
                det_centers = np.array(det_centers)
                track_centers = np.array(track_centers)
                
                # Compute distances
                distances = np.linalg.norm(
                    det_centers[:, np.newaxis, :] - track_centers[np.newaxis, :, :],
                    axis=2
                )
                
                # Hungarian matching (simple greedy for now)
                while distances.size > 0:
                    min_idx = np.unravel_index(distances.argmin(), distances.shape)
                    det_idx, track_idx = min_idx
                    
                    if distances[det_idx, track_idx] > MAX_TRACK_DISTANCE:
                        break
                    
                    det_id = unmatched_detections[det_idx]
                    track_id = track_ids[track_idx]
                    
                    bbox, _, embedding, name, confidence = detections[det_id]
                    associations[track_id] = (bbox, embedding, name, confidence)
                    
                    # Remove matched
                    unmatched_detections.remove(det_id)
                    if track_id in unmatched_tracks:
                        unmatched_tracks.remove(track_id)
                    
                    # Remove from distance matrix
                    distances = np.delete(distances, det_idx, axis=0)
                    distances = np.delete(distances, track_idx, axis=1)
                    del track_ids[track_idx]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox, _, embedding, name, confidence = detections[det_idx]
            
            new_track_id = self.next_track_id
            self.next_track_id += 1
            
            associations[new_track_id] = (bbox, embedding, name, confidence)
        
        return associations
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        
        # Detect and recognize faces
        detections = self.detect_and_recognize_faces(frame)
        
        # Associate to tracks
        associations = self.associate_detections_to_tracks(detections)
        
        # Update tracks
        for track_id, (bbox, embedding, name, confidence) in associations.items():
            if track_id in self.tracks:
                # Update existing track
                track = self.tracks[track_id]
                track.update(bbox, embedding, name, confidence, self.frame_idx)
                
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
                
                if name is not None:
                    self.mef.add_embedding(track_id, embedding)
        
        # Line crossing detection
        if self.line_detector is not None:
            for track_id, track in list(self.tracks.items()):
                # Skip if not seen recently
                if self.frame_idx - track.last_seen_frame > 5:
                    continue
                
                # Skip if not recognized or not enough detections
                if track.name is None or track.detection_count < MIN_TRACK_CONFIDENCE:
                    continue
                
                # Decrement cooldown
                if track.cooldown > 0:
                    track.cooldown -= 1
                
                # Check crossing
                curr_pos = track.get_center()
                curr_side = self.line_detector.get_side(curr_pos)
                
                if track.last_side is not None and track.cooldown == 0:
                    event = self.line_detector.check_crossing(track.last_side, curr_side)
                    
                    # STATE MACHINE: Only record valid state transitions
                    if event == "ENTRY" and track.state == "OUTSIDE":
                        # Person was outside, now entering
                        self.db.record_entry(track.name, track.confidence, track_id)
                        self.entry_count += 1
                        track.state = "INSIDE"  # Update state
                        track.cooldown = COOLDOWN_FRAMES
                        logging.info(f"✅ {track.name} ENTERED (ID:{track_id})")
                    
                    elif event == "ENTRY" and track.state == "INSIDE":
                        # Person already inside, ignore duplicate entry
                        logging.debug(f"⚠️  {track.name} already INSIDE - ignoring duplicate entry")
                    
                    elif event == "EXIT" and track.state == "INSIDE":
                        # Person was inside, now exiting
                        self.db.record_exit(track.name, track.confidence, track_id)
                        self.exit_count += 1
                        track.state = "OUTSIDE"  # Update state
                        track.cooldown = COOLDOWN_FRAMES
                        logging.info(f"❌ {track.name} EXITED (ID:{track_id})")
                    
                    elif event == "EXIT" and track.state == "OUTSIDE":
                        # Person already outside, ignore exit without entry
                        logging.debug(f"⚠️  {track.name} already OUTSIDE - ignoring exit without entry")
                
                track.last_side = curr_side
        
        # Draw on frame
        for track_id, track in self.tracks.items():
            if self.frame_idx - track.last_seen_frame > 5:
                continue
            
            x1, y1, x2, y2 = map(int, track.bbox[:4])
            
            # Color based on recognition
            if track.name is not None:
                color = (0, 255, 0)  # Green
                label = f"ID:{track_id} {track.name} ({track.confidence:.2f})"
            else:
                color = (0, 165, 255)  # Orange
                label = f"ID:{track_id} Unknown"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cx, cy = track.get_center()
            cv2.circle(frame, (cx, cy), 4, color, -1)
        
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    system = FaceBasedAttendanceSystem(args)
    system.run()


if __name__ == "__main__":
    main()
