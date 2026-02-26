"""
Real-Time Attendance System (EfficientDet-D0 + SCRFD + ArcFace + SQLite)
==========================================================================
Requirements:
- EfficientDet-D0: person detection only (tracking by body boxes).
- SCRFD + ArcFace: face recognition used ONLY for authorization.
- No face-based body heuristics for tracking (face used only to authorize).
- No motion prediction: tracks update only when EfficientDet-D0 detects a body.
- Detection-driven tracking: tracks freeze when no detection exists.
"""

import argparse
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from models.SCRFD import SCRFD
from models.ArcFace import ArcFace
from models.EfficientDet import EfficientDet

# API Integration for C# Backend
try:
    from api_integration import AttendanceAPIClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    import onnxruntime as ort
    GPU_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()
except Exception:
    ort = None
    GPU_AVAILABLE = False

# =====================
# DEFAULT CONFIG
# =====================

FACE_DETECTION_MODEL = "./weights/face_detection/det_10g.onnx"
FACE_RECOGNITION_MODEL = "./weights/face_recognition/w600k_r50.onnx"
EFFICIENTDET_MODEL_PATH = "./weights/efficientdet-d0.onnx"

DATABASE_PATH = "attendance.db"
KNOWN_FACES_TABLE = "face_embeddings"
KNOWN_FACES_DIR = "./known_faces"

FACE_CONFIDENCE = 0.5
FACE_SIMILARITY_THRESHOLD = 0.40

DETECTION_CONFIDENCE = 0.3
NMS_THRESHOLD = 0.5

IOU_MATCH_THRESHOLD = 0.3
TRACK_MAX_AGE = 10  # frames without a match before track removal
MIN_FACE_SIZE = 20

MIN_MOVEMENT = 3.0
COOLDOWN_FRAMES = 30
LINE_TOUCH_THRESHOLD = 20.0  # pixels; distance to line to count as touch

# API Configuration
API_BASE_URL = "http://localhost:5000"
DEFAULT_CAMERA_ID = "CAM-01"

RESIZE_WIDTH = 960

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# =====================
# DATABASE
# =====================

class AttendanceDatabase:
    """Handles SQLite database operations for attendance records and face embeddings."""

    def __init__(self, db_path: str, known_faces_table: str):
        self.db_path = db_path
        self.known_faces_table = known_faces_table
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._initialize_attendance_table()

    def _initialize_attendance_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                stable_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence REAL
            )
            """
        )
        self.conn.commit()
        logging.info("Attendance table ready.")

    def load_known_faces(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load known face embeddings from the database.
        Table schema expected:
          face_embeddings(person_name TEXT, stable_id INTEGER, embedding BLOB)
        Returns:
          stable_id -> {"name": str, "embeddings": np.ndarray}
        """
        table = self.known_faces_table
        # Ensure the known-faces table exists
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                person_name TEXT NOT NULL,
                stable_id INTEGER NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

        self.cursor.execute(f"SELECT person_name, stable_id, embedding FROM {table}")
        rows = self.cursor.fetchall()
        if not rows:
            logging.warning("No known faces found in database.")
        data: Dict[int, Dict[str, np.ndarray]] = {}
        for name, stable_id, emb_blob in rows:
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if stable_id not in data:
                data[stable_id] = {"name": name, "embeddings": emb}
            else:
                data[stable_id]["embeddings"] = np.vstack([data[stable_id]["embeddings"], emb])
        logging.info("Loaded %d known persons.", len(data))
        return data

    def import_known_faces_from_dir(self, known_faces_dir: str, rebuild: bool = False) -> int:
        """
        Import .npy embeddings from a directory into the known-faces table.
        Each .npy filename is treated as the person_name.
        Returns the number of embeddings inserted.
        """
        table = self.known_faces_table
        if rebuild:
            self.cursor.execute(f"DELETE FROM {table}")
            self.conn.commit()

        faces_dir = Path(known_faces_dir)
        if not faces_dir.exists():
            logging.warning("Known faces dir not found: %s", faces_dir)
            return 0

        # Determine next stable_id
        self.cursor.execute(f"SELECT MAX(stable_id) FROM {table}")
        row = self.cursor.fetchone()
        next_stable_id = (row[0] or 0) + 1

        inserted = 0
        for npy_file in faces_dir.glob("*.npy"):
            name = npy_file.stem
            try:
                embeddings = np.load(str(npy_file))
            except Exception as exc:
                logging.warning("Failed to load %s: %s", npy_file.name, exc)
                continue

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Check if this name already exists
            self.cursor.execute(
                f"SELECT stable_id FROM {table} WHERE person_name = ? LIMIT 1",
                (name,),
            )
            existing = self.cursor.fetchone()
            if existing is None:
                stable_id = next_stable_id
                next_stable_id += 1
            else:
                stable_id = int(existing[0])

            for emb in embeddings:
                if emb.dtype != np.float32:
                    emb = emb.astype(np.float32)
                emb_blob = emb.tobytes()
                self.cursor.execute(
                    f"INSERT INTO {table} (person_name, stable_id, embedding) VALUES (?, ?, ?)",
                    (name, stable_id, emb_blob),
                )
                inserted += 1

        self.conn.commit()
        logging.info("Imported %d embeddings from %s", inserted, faces_dir)
        return inserted

    def record_event(self, name: str, stable_id: int, event_type: str, confidence: float = 0.0) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            """
            INSERT INTO attendance (person_name, stable_id, event_type, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, stable_id, event_type, timestamp, confidence),
        )
        self.conn.commit()
        logging.info("%s: %s (ID:%s) at %s [conf: %.2f]", event_type, name, stable_id, timestamp, confidence)

    def close(self) -> None:
        self.conn.close()


# =====================
# FACE AUTHORIZATION
# =====================

class FaceAuthorizer:
    """Face recognition used only to authorize identities."""

    def __init__(self, known_faces: Dict[int, Dict[str, np.ndarray]], similarity_threshold: float):
        self.known_faces = known_faces
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], Optional[int], float]:
        if not self.known_faces:
            return None, None, 0.0

        embedding = self._normalize(embedding.flatten())
        best_similarity = 0.0
        best_name = None
        best_stable_id = None

        for stable_id, payload in self.known_faces.items():
            name = payload["name"]
            embeddings = payload["embeddings"]
            for known_emb in embeddings:
                known_emb = self._normalize(known_emb.flatten())
                similarity = float(np.dot(embedding, known_emb))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = name
                    best_stable_id = stable_id

        if best_similarity >= self.similarity_threshold:
            return best_name, best_stable_id, best_similarity
        return None, None, best_similarity


# =====================
# SIMPLE IOU TRACKER (NO PREDICTION)
# =====================

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    last_seen: int
    age: int = 0
    authorized: bool = False
    name: Optional[str] = None
    stable_id: Optional[int] = None
    confidence: float = 0.0


class IoUTracker:
    """Body-only tracking with IoU matching, no motion prediction."""

    def __init__(self, iou_threshold: float, max_age: int):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, detections: List[np.ndarray], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections.
        CRITICAL: Tracks update ONLY when matched to a detection.
        No motion prediction - tracks freeze without detections.
        """
        matched_track_ids = set()
        unmatched_dets = set(range(len(detections)))

        # Greedy IoU matching
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for det_idx in unmatched_dets:
                iou = self._iou(track.bbox, detections[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            if best_det_idx is not None and best_iou >= self.iou_threshold:
                # ONLY UPDATE POSITION WHEN DETECTION MATCHES
                track.bbox = detections[best_det_idx]
                track.last_seen = frame_idx
                track.age = 0
                matched_track_ids.add(track_id)
                unmatched_dets.remove(best_det_idx)
            else:
                # NO DETECTION MATCH - TRACK FREEZES (no position update)
                track.age += 1

        # Create tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = Track(
                track_id=self.next_id,
                bbox=detections[det_idx],
                last_seen=frame_idx,
            )
            self.tracks[self.next_id] = track
            matched_track_ids.add(self.next_id)
            self.next_id += 1

        # Remove stale tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].age > self.max_age:
                del self.tracks[track_id]

        # Return tracks updated this frame (only those with detections)
        return [self.tracks[tid] for tid in matched_track_ids if tid in self.tracks]


# =====================
# LINE CROSSING
# =====================

class LineCrossDetector:
    def __init__(self, line_coords: Tuple[int, int, int, int]):
        self.x1, self.y1, self.x2, self.y2 = line_coords

    def get_side(self, point: Tuple[int, int]) -> int:
        px, py = point
        cross = (self.x2 - self.x1) * (py - self.y1) - (self.y2 - self.y1) * (px - self.x1)
        if cross > 1.0:
            return 1
        if cross < -1.0:
            return -1
        return 0

    def check_crossing(self, prev_side: int, curr_side: int) -> Optional[str]:
        if prev_side == 0 or curr_side == 0:
            return None
        if prev_side < 0 and curr_side > 0:
            return "ENTRY"
        if prev_side > 0 and curr_side < 0:
            return "EXIT"
        return None

    def distance_to_line(self, point: Tuple[int, int]) -> float:
        """Perpendicular distance from point to the line segment (in pixels)."""
        px, py = point
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return float(np.hypot(px - x1, py - y1))
        # Project point onto the infinite line, clamp to segment
        t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))


# =====================
# MAIN SYSTEM
# =====================

class AttendanceEfficientDet:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.frame_idx = 0
        self.entry_count = 0
        self.exit_count = 0

        self.resize_width = args.resize_width if args.resize_width > 0 else None

        self.db = AttendanceDatabase(args.database_path, args.known_faces_table)
        # Prepare known faces from directory if table is empty or rebuild requested
        if args.rebuild_known_faces:
            self.db.import_known_faces_from_dir(args.known_faces_dir, rebuild=True)
        else:
            known = self.db.load_known_faces()
            if not known:
                self.db.import_known_faces_from_dir(args.known_faces_dir, rebuild=False)

        known_faces = self.db.load_known_faces()
        self.authorizer = FaceAuthorizer(known_faces, args.similarity_threshold)

        providers = self._select_providers(args.device)
        if "CUDAExecutionProvider" in providers:
            logging.info("CUDA enabled for ONNX Runtime.")
        else:
            logging.warning("CUDA not available, using CPU.")

        self.face_detector = SCRFD(
            model_path=args.face_det_model,
            conf_thres=args.face_conf,
        )
        self.face_recognizer = ArcFace(model_path=args.face_rec_model)

        # Initialize EfficientDet-D0 (NO YOLO!)
        self.efficientdet_model = EfficientDet(
            model_path=args.efficientdet_model_path,
            conf_thres=args.detection_conf,
            nms_thres=args.nms_threshold,
            person_class_id=args.person_class_id,
            providers=tuple(providers),
        )

        self.tracker = IoUTracker(
            iou_threshold=args.iou_threshold,
            max_age=args.track_max_age,
        )

        self.entry_line_detector: Optional[LineCrossDetector] = None
        self.exit_line_detector: Optional[LineCrossDetector] = None
        self.track_line_state: Dict[int, Dict[str, object]] = {}
        self.stable_id_active_track: Dict[int, int] = {}

        # Initialize API client for C# backend integration
        self.api_client = None
        if API_AVAILABLE:
            api_url = getattr(args, 'api_url', API_BASE_URL)
            self.camera_id = getattr(args, 'camera_id', DEFAULT_CAMERA_ID)
            self.api_client = AttendanceAPIClient(base_url=api_url)
            logging.info(f"API client initialized: {api_url}")

    @staticmethod
    def _select_providers(device: str) -> List[str]:
        if ort is None:
            return ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device == "cpu":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def draw_line_interactive(
        self,
        cap,
        window_name: str,
        prompt: str,
        color: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        points: List[Tuple[int, int]] = []

        def mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            display = frame.copy()
            cv2.putText(display, prompt,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, point in enumerate(points):
                cv2.circle(display, point, 8, color, -1)
                cv2.putText(display, f"P{i+1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], color, 3)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(points) == 2:
                break
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
        cv2.destroyWindow(window_name)
        return (*points[0], *points[1]) if len(points) == 2 else None

    def draw_two_lines_interactive(self, cap) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        entry_line = self.draw_line_interactive(
            cap,
            window_name="Draw Green Line (ENTRY)",
            prompt="Draw GREEN line (ENTRY) - press 's' to save",
            color=(0, 255, 0),
        )
        if entry_line is None:
            return None, None
        exit_line = self.draw_line_interactive(
            cap,
            window_name="Draw Red Line (EXIT)",
            prompt="Draw RED line (EXIT) - press 's' to save",
            color=(0, 0, 255),
        )
        if exit_line is None:
            return entry_line, None
        return entry_line, exit_line

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        faces: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        try:
            bboxes, kpss = self.face_detector.detect(frame, max_num=0)
            if bboxes is None or len(bboxes) == 0:
                return faces
            
            valid_faces = 0
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
                faces.append((bbox, kps, embedding))
                valid_faces += 1
            
            if valid_faces > 0:
                logging.debug(f"Detected {valid_faces} valid faces")
        except Exception as exc:
            logging.warning(f"Face detection error: {exc}")
        return faces

    def detect_persons(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect persons using EfficientDet-D0 ONLY.
        NO YOLO! NO motion prediction!
        """
        bodies = self.efficientdet_model.detect(frame)
        return bodies

    @staticmethod
    def associate_faces_to_bodies(
        face_detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        body_bboxes: List[np.ndarray],
    ) -> Dict[int, Optional[int]]:
        """
        Associate detected faces to body bounding boxes.
        CRITICAL: Face boxes are NEVER used to estimate/extend body boxes.
        Face is used ONLY for authorization.
        """
        if not body_bboxes or not face_detections:
            return {i: None for i in range(len(face_detections))}
        associations: Dict[int, Optional[int]] = {}
        used_bodies = set()
        for face_idx, (face_bbox, _kps, _emb) in enumerate(face_detections):
            fx1, fy1, fx2, fy2 = face_bbox[:4]
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2
            best_body_idx = None
            best_distance = 200.0
            for body_idx, body_bbox in enumerate(body_bboxes):
                if body_idx in used_bodies:
                    continue
                bx1, by1, bx2, by2 = body_bbox[:4]
                # Check if face center is inside upper 70% of body box
                if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by1 + (by2 - by1) * 0.7:
                    body_cx = (bx1 + bx2) / 2
                    body_cy = (by1 + by2) / 2
                    distance = np.hypot(face_cx - body_cx, face_cy - body_cy)
                    if distance < best_distance:
                        best_distance = distance
                        best_body_idx = body_idx
            associations[face_idx] = best_body_idx
            if best_body_idx is not None:
                used_bodies.add(best_body_idx)
        return associations

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        original_frame = frame
        scale_factor = 1.0
        if self.resize_width is not None and frame.shape[1] > self.resize_width:
            scale_factor = self.resize_width / frame.shape[1]
            new_height = int(frame.shape[0] * scale_factor)
            frame = cv2.resize(frame, (self.resize_width, new_height))

        # DETECTION: EfficientDet-D0 ONLY (NO YOLO!)
        body_bboxes = self.detect_persons(frame)
        face_detections = self.detect_faces(frame)
        face_to_body = self.associate_faces_to_bodies(face_detections, body_bboxes)

        # TRACKING: Detection-driven, NO prediction
        updated_tracks = self.tracker.update(body_bboxes, self.frame_idx)

        # AUTHORIZATION: Face recognition only, attach to body tracks
        # Build mapping from body detection index to face recognition results
        body_auth: Dict[int, Tuple[str, int, float]] = {}
        for face_idx, body_idx in face_to_body.items():
            if body_idx is None:
                continue
            _bbox, _kps, emb = face_detections[face_idx]
            name, stable_id, similarity = self.authorizer.identify(emb)
            if name is not None and stable_id is not None:
                body_auth[body_idx] = (name, stable_id, similarity)

        # Update track identities based on this frame's detections
        for track in updated_tracks:
            # Try to find the detection index by IoU (exact match in this frame)
            best_det_idx = None
            best_iou = 0.0
            for idx, det_bbox in enumerate(body_bboxes):
                iou = self.tracker._iou(track.bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            
            # AUTHORIZATION: Attach identity if face recognized
            # Allow re-authorization if face is detected (for better accuracy)
            if best_det_idx is not None and best_det_idx in body_auth:
                name, stable_id, confidence = body_auth[best_det_idx]
                
                # Check if this is a new identity or better confidence
                if not track.authorized:
                    # First-time authorization
                    track.authorized = True
                    track.name = name
                    track.stable_id = stable_id
                    track.confidence = confidence
                    logging.info(f"✓ Authorized: {name} (ID:{stable_id}, conf:{confidence:.2f})")
                    
                    # Ensure a stable_id points to the most recent track_id
                    prev_track_id = self.stable_id_active_track.get(stable_id)
                    if prev_track_id is not None and prev_track_id != track.track_id:
                        if prev_track_id in self.tracker.tracks:
                            self.tracker.tracks[prev_track_id].authorized = False
                    self.stable_id_active_track[stable_id] = track.track_id
                elif track.stable_id != stable_id:
                    # Different person detected - update if confidence is higher
                    if confidence > track.confidence:
                        logging.info(f"⚠ Identity changed: {track.name} -> {name} (better conf: {confidence:.2f} > {track.confidence:.2f})")
                        track.name = name
                        track.stable_id = stable_id
                        track.confidence = confidence
                        self.stable_id_active_track[stable_id] = track.track_id
                else:
                    # Same person, update confidence if better
                    if confidence > track.confidence:
                        track.confidence = confidence
            # PERSISTENT TRACKING: Identity persists even after face loss

        # LINE CROSSING: Authorized only, updated this frame
        if self.entry_line_detector is not None and self.exit_line_detector is not None:
            for track in updated_tracks:
                if not track.authorized or track.stable_id is None or track.name is None:
                    continue
                bx1, by1, bx2, by2 = track.bbox[:4]
                curr_pos = (int((bx1 + bx2) / 2), int(by2))
                if scale_factor != 1.0:
                    curr_pos = (int(curr_pos[0] / scale_factor), int(curr_pos[1] / scale_factor))
                entry_side = self.entry_line_detector.get_side(curr_pos)
                exit_side = self.exit_line_detector.get_side(curr_pos)

                state = self.track_line_state.get(track.stable_id)
                if state is None:
                    self.track_line_state[track.stable_id] = {
                        "entry_last_side": entry_side,
                        "exit_last_side": exit_side,
                        "cooldown": 0,
                        "state": "UNKNOWN",
                        "last_pos": curr_pos,
                        "entry_on_line": False,
                        "exit_on_line": False,
                    }
                    continue

                if state["cooldown"] > 0:
                    state["cooldown"] -= 1

                last_pos = state.get("last_pos")
                if last_pos is not None:
                    dx = curr_pos[0] - last_pos[0]
                    dy = curr_pos[1] - last_pos[1]
                    if (dx * dx + dy * dy) < (MIN_MOVEMENT * MIN_MOVEMENT):
                        state["last_pos"] = curr_pos
                        state["entry_last_side"] = entry_side
                        state["exit_last_side"] = exit_side
                        continue

                if state["cooldown"] == 0:
                    entry_touch = self.entry_line_detector.distance_to_line(curr_pos) <= LINE_TOUCH_THRESHOLD
                    exit_touch = self.exit_line_detector.distance_to_line(curr_pos) <= LINE_TOUCH_THRESHOLD

                    entry_trigger = entry_touch and not state["entry_on_line"]
                    exit_trigger = exit_touch and not state["exit_on_line"]

                    if entry_trigger and state["state"] in ("UNKNOWN", "OUTSIDE"):
                        self.db.record_event(track.name, track.stable_id, "ENTRY", track.confidence)
                        # Send to C# backend
                        if self.api_client:
                            self.api_client.send_attendance(
                                employee_id=track.stable_id, name=track.name,
                                confidence=track.confidence, camera_id=self.camera_id)
                        self.entry_count += 1
                        state["state"] = "INSIDE"
                        state["cooldown"] = COOLDOWN_FRAMES
                    elif exit_trigger and state["state"] in ("UNKNOWN", "INSIDE"):
                        self.db.record_event(track.name, track.stable_id, "EXIT", track.confidence)
                        # Send to C# backend
                        if self.api_client:
                            self.api_client.send_attendance(
                                employee_id=track.stable_id, name=track.name,
                                confidence=track.confidence, camera_id=self.camera_id)
                        self.exit_count += 1
                        state["state"] = "OUTSIDE"
                        state["cooldown"] = COOLDOWN_FRAMES

                state["entry_last_side"] = entry_side
                state["exit_last_side"] = exit_side
                state["last_pos"] = curr_pos
                state["entry_on_line"] = self.entry_line_detector.distance_to_line(curr_pos) <= LINE_TOUCH_THRESHOLD
                state["exit_on_line"] = self.exit_line_detector.distance_to_line(curr_pos) <= LINE_TOUCH_THRESHOLD

        if scale_factor != 1.0:
            frame = original_frame

        # VISUALIZATION
        if self.entry_line_detector is not None:
            cv2.line(
                frame,
                (self.entry_line_detector.x1, self.entry_line_detector.y1),
                (self.entry_line_detector.x2, self.entry_line_detector.y2),
                (0, 255, 0),
                3,
            )
        if self.exit_line_detector is not None:
            cv2.line(
                frame,
                (self.exit_line_detector.x1, self.exit_line_detector.y1),
                (self.exit_line_detector.x2, self.exit_line_detector.y2),
                (0, 0, 255),
                3,
            )

        for track in updated_tracks:
            bx1, by1, bx2, by2 = map(int, track.bbox[:4])
            if scale_factor != 1.0:
                bx1 = int(bx1 / scale_factor)
                by1 = int(by1 / scale_factor)
                bx2 = int(bx2 / scale_factor)
                by2 = int(by2 / scale_factor)

            if track.authorized and track.name and track.stable_id is not None:
                # Recognized person - GREEN with prominent display
                color = (0, 255, 0)
                label = f"{track.name}"
                id_label = f"ID:{track.stable_id}"
                conf_label = f"Conf:{track.confidence:.2f}"
                
                # Draw thick green box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
                
                # Draw filled background for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (bx1, by1 - 30), (bx1 + text_size[0] + 10, by1), color, -1)
                
                # Draw name in white on green background
                cv2.putText(frame, label, (bx1 + 5, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw ID and confidence below name
                cv2.putText(frame, f"{id_label} | {conf_label}", (bx1, by2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Unknown person - ORANGE
                color = (0, 165, 255)
                label = f"Unknown (ID:{track.track_id})"
                
                # Draw thin orange box
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, label, (bx1, max(10, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Entry: {self.entry_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {self.exit_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "EfficientDet-D0", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame

    def run(self) -> None:
        source = self.args.source
        try:
            if source.isdigit():
                source = int(source)
        except AttributeError:
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        if self.args.no_interaction:
            # Default: Horizontal lines in middle of frame
            y_mid = height // 2
            entry_line = (0, y_mid, width, y_mid)
            exit_line = (0, y_mid + 100, width, y_mid + 100)
            logging.info("Using default lines (No Interaction Mode)")
        else:
            entry_line, exit_line = self.draw_two_lines_interactive(cap)
            
        if entry_line is None or exit_line is None:
            logging.error("Line drawing cancelled.")
            return
        self.entry_line_detector = LineCrossDetector(entry_line)
        self.exit_line_detector = LineCrossDetector(exit_line)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))
            logging.info("Saving output to: %s", self.args.output)

        logging.info("Starting attendance system (EfficientDet-D0). Press 'q' to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and self.args.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                processed = self.process_frame(frame)
                if writer is not None:
                    writer.write(processed)
                if not self.args.skip_display:
                    cv2.imshow("Attendance EfficientDet-D0", processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                self.frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            self.db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attendance System (EfficientDet-D0 + SCRFD + ArcFace + SQLite)")
    parser.add_argument("--source", type=str, default="rtsp://admin:deepminD1@192.168.0.3:554/Streaming/Channels/101",
                        help="Video source: 0 for webcam, RTSP URL, or video file path")
    parser.add_argument("--face-det-model", type=str, default=FACE_DETECTION_MODEL,
                        help="Path to face detection model (SCRFD)")
    parser.add_argument("--face-rec-model", type=str, default=FACE_RECOGNITION_MODEL,
                        help="Path to face recognition model (ArcFace)")
    parser.add_argument("--efficientdet-model-path", type=str, default=EFFICIENTDET_MODEL_PATH,
                        help="Path to EfficientDet-D0 ONNX model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="ONNX Runtime device")
    parser.add_argument("--database-path", type=str, default=DATABASE_PATH,
                        help="Path to SQLite database")
    parser.add_argument("--known-faces-table", type=str, default=KNOWN_FACES_TABLE,
                        help="Table storing known face embeddings")
    parser.add_argument("--known-faces-dir", type=str, default=KNOWN_FACES_DIR,
                        help="Directory with .npy face embeddings")
    parser.add_argument("--rebuild-known-faces", action="store_true",
                        help="Rebuild known faces table from --known-faces-dir")
    parser.add_argument("--face-conf", type=float, default=FACE_CONFIDENCE,
                        help="Face detection confidence threshold")
    parser.add_argument("--similarity-threshold", type=float, default=FACE_SIMILARITY_THRESHOLD,
                        help="Face similarity threshold")
    parser.add_argument("--detection-conf", type=float, default=DETECTION_CONFIDENCE,
                        help="EfficientDet person detection confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=NMS_THRESHOLD,
                        help="NMS IoU threshold for EfficientDet")
    parser.add_argument("--person-class-id", type=int, default=0,
                        help="Class id for 'person' in the EfficientDet model label space")
    parser.add_argument("--iou-threshold", type=float, default=IOU_MATCH_THRESHOLD,
                        help="IoU threshold for body tracking")
    parser.add_argument("--track-max-age", type=int, default=TRACK_MAX_AGE,
                        help="Max frames to keep an unmatched track")
    parser.add_argument("--resize-width", type=int, default=RESIZE_WIDTH,
                        help="Resize frame width for processing (0 to disable)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--skip-display", action="store_true", help="Run without UI")
    parser.add_argument("--loop-video", action="store_true", help="Loop video at EOF")
    parser.add_argument("--no-interaction", action="store_true", help="Skip interactive line drawing (use defaults)")
    # API Integration arguments
    parser.add_argument("--api-url", type=str, default=API_BASE_URL,
                        help="C# backend API base URL (default: http://localhost:5000)")
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID,
                        help="Camera identifier sent to backend (default: CAM-01)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = AttendanceEfficientDet(args)
    system.run()


if __name__ == "__main__":
    main()
