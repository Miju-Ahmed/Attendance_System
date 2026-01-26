"""
Person Counter using YOLO + ByteTrack
======================================
This script counts people entering/exiting through a door using:
- YOLO for person detection
- ByteTrack for tracking with persistent IDs
- Dual-line system (outer + inner) for reliable entry/exit detection
- Optional ROI zone to filter detections
- Centroid movement direction tracking
- Count once per ID to avoid duplicates
"""

import os
import cv2
import numpy as np
import time
import json
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# =====================
# CONFIGURATION
# =====================
VIDEO_PATH = "/home/dml/Desktop/Miju/Apex_Customer_Count/trimmed_videos/merged_trimmed_D01_713.mp4"
OUTPUT_PATH = "output/face_custom_merge_713.mp4"
MODEL_PATH = "yolov8n.pt"  # YOLO model

# Detection parameters
MIN_CONFIDENCE = 0.4  # Minimum detection confidence
PERSON_CLASS_ID = 0  # YOLO class ID for person

# Tracking parameters
COOLDOWN_FRAMES = 15  # Prevent double-counting within this many frames
MIN_MOVEMENT = 1.0  # Minimum pixels of movement to consider purposeful

# ROI (Region of Interest) - set to True to enable ROI zone drawing
USE_ROI = True  # Set to False to skip ROI drawing

# Face recognition configuration
FACES_DIR = "Excluded_Persons"  # Folder structure: Excluded_Persons/<person_name>/*.jpg|png
ENABLE_FACE_SKIP = True  # If True: skip entry/exit counting for recognized faces
FACE_RECOGNIZE_EVERY_N_FRAMES = 10  # Run recognition every N frames per track (until recognized)
FACE_MATCH_THRESHOLD = 0.48  # Lower = stricter (face_recognition distance threshold)
LBPH_CONFIDENCE_THRESHOLD = 60.0  # Lower = stricter (OpenCV LBPH confidence threshold)
ARCFACE_SIMILARITY_THRESHOLD = 0.62  # Higher = stricter (ArcFace cosine similarity threshold)
ARCFACE_SIMILARITY_MARGIN = 0.07  # Require best - second_best >= margin to accept match
FACE_CONFIRM_HITS = 3  # Require N consecutive matches before locking identity
ARCFACE_DET_MODEL_PATH = "weights/face_detection/det_10g.onnx"
ARCFACE_REC_MODEL_PATH = "weights/face_recognition/w600k_r50.onnx"

# =====================
# GLOBALS FOR DRAWING
# =====================
drawing_points = []
drawing_complete = False
current_drawing_mode = None  # 'outer', 'inner', or 'roi'


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for drawing lines and ROI."""
    global drawing_points, drawing_complete
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_drawing_mode == 'roi':
            # ROI needs 4 points for polygon
            if len(drawing_points) < 4:
                drawing_points.append((x, y))
                print(f"ROI Point {len(drawing_points)}: ({x}, {y})")
                if len(drawing_points) == 4:
                    drawing_complete = True
                    print("ROI zone complete! Press 's' to continue.")
        else:
            # Lines need 2 points
            if len(drawing_points) < 2:
                drawing_points.append((x, y))
                print(f"Point {len(drawing_points)}: ({x}, {y})")
                if len(drawing_points) == 2:
                    drawing_complete = True
                    print(f"{current_drawing_mode.upper()} line complete! Press 's' to continue.")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click to remove last point
        if drawing_points:
            removed = drawing_points.pop()
            print(f"Removed point: {removed}")
            drawing_complete = False


def draw_roi_zone(cap):
    """
    Interactive drawing of ROI zone (4-point polygon).
    Returns: numpy array of 4 points or None
    """
    global drawing_points, drawing_complete, current_drawing_mode
    
    drawing_points = []
    drawing_complete = False
    current_drawing_mode = 'roi'
    
    cv2.namedWindow("Draw ROI Zone")
    cv2.setMouseCallback("Draw ROI Zone", mouse_callback)
    
    print("\n" + "="*60)
    print("DRAW ROI ZONE (Region of Interest)")
    print("="*60)
    print("Click 4 points to define the detection zone.")
    print("Only people inside this zone will be counted.")
    print("Order: top-left, top-right, bottom-right, bottom-left")
    print("RIGHT CLICK to undo last point")
    print("Press 's' to continue after drawing.")
    print("="*60 + "\n")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        display = frame.copy()
        
        # Draw instruction
        cv2.putText(display, "Draw ROI Zone (4 points)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Points: {len(drawing_points)}/4", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw existing points
        for i, point in enumerate(drawing_points):
            cv2.circle(display, point, 8, (0, 255, 0), -1)
            cv2.putText(display, f"P{i+1}", (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw lines between points
        if len(drawing_points) >= 2:
            for i in range(len(drawing_points) - 1):
                cv2.line(display, drawing_points[i], drawing_points[i + 1], 
                        (0, 255, 255), 2)
        
        # Draw polygon if we have 4 points
        if len(drawing_points) == 4:
            pts = np.array(drawing_points, dtype=np.int32)
            # Draw filled polygon with transparency
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [pts], True, (0, 255, 255), 3)
            cv2.putText(display, "Press 's' to continue", 
                       (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Draw ROI Zone", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and drawing_complete:
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyWindow("Draw ROI Zone")
    
    if len(drawing_points) == 4:
        return np.array(drawing_points, dtype=np.int32)
    return None


def draw_line(cap, line_name, color):
    """
    Interactive drawing of a line (2 points).
    Returns: (x1, y1, x2, y2) coordinates or None
    """
    global drawing_points, drawing_complete, current_drawing_mode
    
    drawing_points = []
    drawing_complete = False
    current_drawing_mode = line_name.lower()
    
    cv2.namedWindow(f"Draw {line_name} Line")
    cv2.setMouseCallback(f"Draw {line_name} Line", mouse_callback)
    
    print(f"\n{'='*60}")
    print(f"DRAW {line_name.upper()} LINE")
    print(f"{'='*60}")
    print(f"Click 2 points to define the {line_name.lower()} line.")
    if line_name.lower() == 'outer':
        print("This represents the OUTSIDE boundary.")
    else:
        print("This represents the INSIDE boundary.")
    print("RIGHT CLICK to undo last point")
    print("Press 's' to continue after drawing.")
    print(f"{'='*60}\n")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        display = frame.copy()
        
        # Draw instruction
        cv2.putText(display, f"Draw {line_name} Line (2 points)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Points: {len(drawing_points)}/2", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw existing points
        for i, point in enumerate(drawing_points):
            cv2.circle(display, point, 8, color, -1)
            cv2.putText(display, f"P{i+1}", (point[0] + 10, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line if we have 2 points
        if len(drawing_points) == 2:
            cv2.line(display, drawing_points[0], drawing_points[1], color, 3)
            cv2.putText(display, "Press 's' to continue", 
                       (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(f"Draw {line_name} Line", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and drawing_complete:
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyWindow(f"Draw {line_name} Line")
    
    if len(drawing_points) == 2:
        return (*drawing_points[0], *drawing_points[1])
    return None


def point_side_of_line(line, point):
    """
    Calculate which side of the line a point is on.
    Returns: positive value if on one side, negative if on other side
    """
    x1, y1, x2, y2 = line
    px, py = point
    # Cross product to determine side
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def sign(value, epsilon=1e-3):
    """
    Return the sign of a value with a small epsilon for stability.
    Returns: 1, -1, or 0
    """
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def unit_vector(p1, p2):
    """Return a normalized direction vector from p1 -> p2."""
    vx, vy = p2[0] - p1[0], p2[1] - p1[1]
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-6:
        return (0.0, 1.0)
    return (vx / norm, vy / norm)


def get_region(outer_sign, inner_sign, inside_sign):
    """
    Classify point relative to outer/inner lines.
    Returns: 'inside', 'between', or 'outside'
    """
    if inner_sign == inside_sign:
        return "inside"
    if outer_sign == inside_sign:
        return "between"
    return "outside"


def get_centroid(bbox):
    """
    Get the centroid (bottom-center point) of a bounding box.
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    bottom_y = int(y2)  # Use bottom of bbox (feet position)
    return (center_x, bottom_y)


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon."""
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


def _iter_face_image_paths(faces_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not faces_dir.exists():
        return []
    paths = []
    for person_dir in sorted([p for p in faces_dir.iterdir() if p.is_dir()]):
        for img_path in sorted(person_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                paths.append(img_path)
    return paths


def _latest_mtime(paths):
    latest = 0.0
    for p in paths:
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


class FaceMatcher:
    """
    Loads known faces from `faces/<name>/` and recognizes faces in person bboxes.
    Backend preference:
      1) ArcFace (onnxruntime) with cached npz
      2) `face_recognition` (embeddings) with cached npz
      3) OpenCV LBPH (`cv2.face`) with cached model
      3) Disabled (no backend available)
    """

    def __init__(self, faces_dir: str):
        self.faces_dir = Path(faces_dir)
        self.backend = None

        # Used for some backends / training images.
        self._face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        if self._face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade for face detection")

        self._face_reid = None
        self._known_arc_labels = []  # person names (one per identity)
        self._known_arc_emb_norm = None  # shape: (N_people, 512) normalized

        self._fr = None
        self._known_labels = []
        self._known_encodings = None

        self._lbph = None
        self._lbph_id_to_name = {}

        self._init_backend_and_load()

    def _init_backend_and_load(self):
        # Try ArcFace + SCRFD first (no extra pip deps if onnxruntime is present).
        try:
            det_path = Path(ARCFACE_DET_MODEL_PATH)
            rec_path = Path(ARCFACE_REC_MODEL_PATH)
            if det_path.exists() and rec_path.exists():
                from reid.face_reid import FaceReID  # local module

                self._face_reid = FaceReID(str(det_path), str(rec_path))
                self.backend = "arcface_onnx"
                self._load_or_build_arcface_cache()
                if self._known_arc_emb_norm is not None and len(self._known_arc_labels) > 0:
                    return
        except Exception:
            self._face_reid = None

        # Try face_recognition first
        try:
            import face_recognition  # type: ignore
            self._fr = face_recognition
            self.backend = "face_recognition"
            self._load_or_build_face_recognition_cache()
            return
        except Exception:
            self._fr = None

        # Fallback: OpenCV LBPH (requires opencv-contrib)
        try:
            if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
                self.backend = "lbph"
                self._load_or_train_lbph()
                return
        except Exception:
            pass

        self.backend = None

    def _load_or_build_arcface_cache(self):
        cache_path = self.faces_dir / "arcface_index.npz"
        img_paths = _iter_face_image_paths(self.faces_dir)
        if not img_paths:
            self._known_arc_labels = []
            self._known_arc_emb_norm = None
            return

        need_rebuild = True
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            images_mtime = _latest_mtime(img_paths)
            need_rebuild = cache_mtime < images_mtime

        if not need_rebuild:
            try:
                data = np.load(str(cache_path), allow_pickle=False)
                # New format (per-identity)
                if "person_labels" in data and "person_embeddings_norm" in data:
                    labels = data["person_labels"].astype(str).tolist()
                    emb_norm = data["person_embeddings_norm"].astype(np.float32)
                    if emb_norm.ndim == 2 and emb_norm.shape[0] == len(labels):
                        self._known_arc_labels = labels
                        self._known_arc_emb_norm = emb_norm
                        return
                # Old format (per-image) - keep backward compatibility
                if "labels" in data and "embeddings" in data:
                    labels = data["labels"].astype(str).tolist()
                    emb = data["embeddings"].astype(np.float32)
                    if emb.ndim == 2 and emb.shape[0] == len(labels):
                        # Convert to per-identity and overwrite cache
                        per_person = {}
                        for lab, vec in zip(labels, emb):
                            vec_norm = vec.astype(np.float32)
                            denom = float(np.linalg.norm(vec_norm))
                            if denom <= 1e-12:
                                continue
                            vec_norm = vec_norm / denom
                            per_person.setdefault(lab, []).append(vec_norm)
                        self._write_arcface_cache(cache_path, per_person)
                        self._known_arc_labels = sorted(per_person.keys())
                        self._known_arc_emb_norm = np.stack(
                            [np.mean(per_person[n], axis=0) for n in self._known_arc_labels],
                            axis=0
                        ).astype(np.float32)
                        self._known_arc_emb_norm = self._l2_normalize_rows(self._known_arc_emb_norm)
                        return
            except Exception:
                need_rebuild = True

        if self._face_reid is None:
            self._known_arc_labels = []
            self._known_arc_emb_norm = None
            return

        per_person = {}
        for img_path in img_paths:
            name = img_path.parent.name
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]
            emb, _ = self._face_reid.detect_embed_in_crop(image_bgr, (0, 0, w, h))
            if emb is None:
                continue
            vec = emb.astype(np.float32)
            denom = float(np.linalg.norm(vec))
            if denom <= 1e-12:
                continue
            vec = vec / denom
            per_person.setdefault(name, []).append(vec)

        if not per_person:
            self._known_arc_labels = []
            self._known_arc_emb_norm = None
            return

        self._write_arcface_cache(cache_path, per_person)
        self._known_arc_labels = sorted(per_person.keys())
        person_means = np.stack(
            [np.mean(per_person[n], axis=0) for n in self._known_arc_labels],
            axis=0
        ).astype(np.float32)
        self._known_arc_emb_norm = self._l2_normalize_rows(person_means)

    def _write_arcface_cache(self, cache_path: Path, per_person: dict):
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        person_labels = sorted(per_person.keys())
        person_means = np.stack([np.mean(per_person[n], axis=0) for n in person_labels], axis=0).astype(np.float32)
        person_means_norm = self._l2_normalize_rows(person_means)
        np.savez_compressed(
            str(cache_path),
            person_labels=np.array(person_labels, dtype=object),
            person_embeddings_norm=person_means_norm,
        )

    @staticmethod
    def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return x / norms

    def _load_or_build_face_recognition_cache(self):
        cache_path = self.faces_dir / "encodings.npz"
        img_paths = _iter_face_image_paths(self.faces_dir)
        if not img_paths:
            self._known_labels = []
            self._known_encodings = None
            return

        need_rebuild = True
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            images_mtime = _latest_mtime(img_paths)
            need_rebuild = cache_mtime < images_mtime

        if not need_rebuild:
            try:
                data = np.load(str(cache_path), allow_pickle=False)
                labels = data["labels"].astype(str).tolist()
                encodings = data["encodings"].astype(np.float32)
                if encodings.ndim == 2 and encodings.shape[0] == len(labels):
                    self._known_labels = labels
                    self._known_encodings = encodings
                    return
            except Exception:
                need_rebuild = True

        labels = []
        encodings = []
        for img_path in img_paths:
            name = img_path.parent.name
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue

            face = self._extract_largest_face_bgr(image_bgr)
            if face is None:
                continue

            encoding = self._encode_face_bgr(face)
            if encoding is None:
                continue

            labels.append(name)
            encodings.append(encoding.astype(np.float32))

        if not encodings:
            self._known_labels = []
            self._known_encodings = None
            return

        encodings_arr = np.stack(encodings, axis=0)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), labels=np.array(labels, dtype=object), encodings=encodings_arr)
        self._known_labels = labels
        self._known_encodings = encodings_arr

    def _load_or_train_lbph(self):
        model_path = self.faces_dir / "lbph_model.yml"
        labels_path = self.faces_dir / "lbph_labels.json"
        img_paths = _iter_face_image_paths(self.faces_dir)
        if not img_paths:
            self._lbph = None
            self._lbph_id_to_name = {}
            return

        need_train = True
        if model_path.exists() and labels_path.exists():
            model_mtime = model_path.stat().st_mtime
            images_mtime = _latest_mtime(img_paths)
            need_train = model_mtime < images_mtime

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not need_train:
            try:
                recognizer.read(str(model_path))
                with open(labels_path, "r", encoding="utf-8") as f:
                    id_to_name = {int(k): str(v) for k, v in json.load(f).items()}
                self._lbph = recognizer
                self._lbph_id_to_name = id_to_name
                return
            except Exception:
                need_train = True

        name_to_id = {}
        faces = []
        ids = []

        next_id = 0
        for img_path in img_paths:
            name = img_path.parent.name
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue

            face_bgr = self._extract_largest_face_bgr(image_bgr)
            if face_bgr is None:
                continue

            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)
            faces.append(face_gray)
            ids.append(name_to_id[name])

        if len(faces) < 2:
            self._lbph = None
            self._lbph_id_to_name = {}
            return

        recognizer.train(faces, np.array(ids, dtype=np.int32))
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        recognizer.write(str(model_path))
        id_to_name = {v: k for k, v in name_to_id.items()}
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in id_to_name.items()}, f, ensure_ascii=False, indent=2)

        self._lbph = recognizer
        self._lbph_id_to_name = id_to_name

    def _extract_largest_face_bgr(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)
        return image_bgr[y1:y2, x1:x2]

    def _encode_face_bgr(self, face_bgr: np.ndarray):
        if self._fr is None:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        h, w = face_rgb.shape[:2]
        # Provide a single face location covering the crop to avoid running another detector.
        encs = self._fr.face_encodings(face_rgb, known_face_locations=[(0, w, h, 0)])
        if not encs:
            return None
        return np.array(encs[0], dtype=np.float32)

    def recognize_person_bbox(self, frame_bgr: np.ndarray, bbox_xyxy):
        """
        Returns (name, score) if recognized else (None, None).
        score is distance (face_recognition) or confidence (LBPH).
        """
        if self.backend is None:
            return None, None

        x1, y1, x2, y2 = map(int, bbox_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None, None

        if self.backend == "arcface_onnx":
            if (
                self._face_reid is None
                or self._known_arc_emb_norm is None
                or not self._known_arc_labels
            ):
                return None, None
            emb, _ = self._face_reid.detect_embed_in_crop(frame_bgr, (x1, y1, x2, y2))
            if emb is None:
                return None, None
            emb_norm = emb.astype(np.float32)
            denom = float(np.linalg.norm(emb_norm))
            if denom <= 1e-12:
                return None, None
            emb_norm = emb_norm / denom
            sims = (self._known_arc_emb_norm @ emb_norm).astype(np.float32)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if sims.shape[0] >= 2:
                second_best = float(np.partition(sims, -2)[-2])
            else:
                second_best = -1.0
            margin = best_sim - second_best
            if best_sim >= ARCFACE_SIMILARITY_THRESHOLD and margin >= ARCFACE_SIMILARITY_MARGIN:
                return self._known_arc_labels[best_idx], best_sim
            return None, best_sim

        # Other backends use top portion of the person bbox to find a face.
        h = y2 - y1
        face_roi_y2 = y1 + int(h * 0.6)
        roi = frame_bgr[y1:face_roi_y2, x1:x2]
        if roi.size == 0:
            return None, None

        face_bgr = self._extract_largest_face_bgr(roi)
        if face_bgr is None:
            return None, None

        if self.backend == "face_recognition":
            if self._known_encodings is None or not self._known_labels:
                return None, None
            encoding = self._encode_face_bgr(face_bgr)
            if encoding is None:
                return None, None

            diffs = self._known_encodings - encoding[None, :]
            dists = np.linalg.norm(diffs, axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist <= FACE_MATCH_THRESHOLD:
                return self._known_labels[best_idx], best_dist
            return None, best_dist

        if self.backend == "lbph":
            if self._lbph is None or not self._lbph_id_to_name:
                return None, None
            face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)
            label_id, confidence = self._lbph.predict(face_gray)
            confidence = float(confidence)
            if confidence <= LBPH_CONFIDENCE_THRESHOLD and label_id in self._lbph_id_to_name:
                return self._lbph_id_to_name[label_id], confidence
            return None, confidence

        return None, None


class PersonCounter:
    """Main person counting class with dual-line system."""
    
    def __init__(self, video_path, output_path, outer_line, inner_line, roi_zone=None):
        self.video_path = video_path
        self.output_path = output_path
        self.outer_line = outer_line
        self.inner_line = inner_line
        self.roi_zone = roi_zone
        
        # Calculate the "inside" direction
        outer_mid = ((outer_line[0] + outer_line[2]) * 0.5, 
                     (outer_line[1] + outer_line[3]) * 0.5)
        inner_mid = ((inner_line[0] + inner_line[2]) * 0.5, 
                     (inner_line[1] + inner_line[3]) * 0.5)
        
        in_dir = unit_vector(outer_mid, inner_mid)
        inside_probe = (inner_mid[0] + in_dir[0] * 8.0, inner_mid[1] + in_dir[1] * 8.0)
        self.inside_sign = sign(point_side_of_line(inner_line, inside_probe))
        
        # Initialize YOLO model with ByteTrack
        print("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded")

        # Face matcher (optional)
        self.face_matcher = None
        if ENABLE_FACE_SKIP:
            try:
                self.face_matcher = FaceMatcher(FACES_DIR)
                if self.face_matcher.backend is None:
                    print("⚠️  Face skip enabled, but no recognition backend available (install `face_recognition` or opencv-contrib).")
                else:
                    print(f"✅ Face recognition enabled (backend: {self.face_matcher.backend})")
            except Exception as e:
                print(f"⚠️  Face recognition disabled due to error: {e}")
                self.face_matcher = None
        
        # Counting statistics
        self.enter_count = 0
        self.exit_count = 0
        
        # Track states: track_id -> state dict
        self.track_states = {}
        
        # Track IDs that have been counted
        self.counted_enter_ids = set()
        self.counted_exit_ids = set()
        
        # Frame counter
        self.frame_idx = 0

        # Interactive selection state
        self.current_tracks = []  # list of dicts: {track_id, bbox, centroid}
        self.selected_track_id = None
        self._mouse_callback_set = False

    def _main_mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to select a tracked person by clicking inside their bbox."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Find the first track whose bbox contains the click point (prefers nearest by centroid)
        clicked = (x, y)
        best_id = None
        best_dist = 1e9
        for t in self.current_tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            if x1 <= x <= x2 and y1 <= y <= y2:
                cx, cy = t["centroid"]
                d = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                if d < best_dist:
                    best_dist = d
                    best_id = t["track_id"]
        if best_id is not None:
            self.selected_track_id = best_id
            print(f"🎯 Selected ID-{best_id}. Press 'E' to exclude from counting.")

    def _ensure_mouse_callback(self):
        if not self._mouse_callback_set:
            try:
                cv2.namedWindow("Person Counter")
            except Exception:
                pass
            cv2.setMouseCallback("Person Counter", self._main_mouse_callback)
            self._mouse_callback_set = True

    def _exclude_selected_track(self, frame):
        """Mark the selected track ID as excluded (skip counting) and optionally save a snapshot."""
        tid = self.selected_track_id
        if tid is None:
            return
        state = self.track_states.get(tid)
        if not state:
            print("⚠️ No state found for selected track.")
            return
        # Assign a stable manual name and mark excluded
        manual_name = f"Excluded_{tid}"
        # Call the same handler to rollback any counts if needed
        self._on_face_recognized(tid, state, manual_name, None)

        # Save a cropped snapshot to aid future recognition (optional)
        try:
            # Find the bbox from current tracks
            bbox = None
            for t in self.current_tracks:
                if t["track_id"] == tid:
                    bbox = t["bbox"]
                    break
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    out_dir = Path(FACES_DIR) / manual_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"snap_{self.frame_idx}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    print(f"🖼️  Saved snapshot to {out_path}")
        except Exception as e:
            print(f"⚠️  Failed to save snapshot: {e}")
    
    def process_video(self):
        """Main video processing loop."""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Cannot open video")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30.0
        frame_interval_s = 1.0 / fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video: {frame_w}x{frame_h} @ {fps:.1f} fps, {total_frames} frames")
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        
        # Video writer
        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_w, frame_h)
        )
        
        print(f"🎬 Processing video...")
        if self.roi_zone is not None:
            print(f"📍 ROI zone enabled: {len(self.roi_zone)} points")
        print(f"📊 Outer line: {self.outer_line}")
        print(f"📊 Inner line: {self.inner_line}\n")

        # Prepare interactive selection
        self._ensure_mouse_callback()
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.perf_counter()
            self.frame_idx += 1
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Write and display
            writer.write(annotated_frame)
            cv2.imshow("Person Counter", annotated_frame)
            
            # Progress
            if self.frame_idx % 30 == 0:
                progress = (self.frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Enter: {self.enter_count} | Exit: {self.exit_count}")

            # Play back at (approximately) the source FPS.
            elapsed_s = time.perf_counter() - frame_start
            delay_ms = max(1, int((frame_interval_s - elapsed_s) * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Exclude currently selected track
                self._exclude_selected_track(annotated_frame)
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Final Results:")
        print(f"   Entered: {self.enter_count}")
        print(f"   Exited:  {self.exit_count}")
        print(f"   Net:     {self.enter_count - self.exit_count}")
        print(f"   Unique IDs tracked: {len(self.track_states)}")
        print(f"📁 Output saved to: {self.output_path}")
    
    def process_frame(self, frame):
        """Process a single frame: detect, track, count."""
        # Draw zones
        self.draw_zones(frame)
        
        # Run YOLO detection + ByteTrack tracking
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[PERSON_CLASS_ID],
            conf=MIN_CONFIDENCE,
            tracker="bytetrack.yaml"
        )
        
        # Process detections
        self.current_tracks = []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Get centroid
                centroid = get_centroid(box)
                
                # Check if in ROI (if enabled)
                if self.roi_zone is not None:
                    if not point_in_polygon(centroid, self.roi_zone):
                        continue  # Skip detections outside ROI
                
                # Keep track for interactive selection
                self.current_tracks.append({
                    "track_id": int(track_id),
                    "bbox": box.copy(),
                    "centroid": centroid,
                })

                # Process this tracked person
                self.process_person(frame, box, track_id, centroid, conf)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        return frame
    
    def process_person(self, frame, bbox, track_id, centroid, confidence):
        """Process a single tracked person."""
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = centroid
        
        # Calculate which side of each line the centroid is on
        outer_side_value = point_side_of_line(self.outer_line, centroid)
        inner_side_value = point_side_of_line(self.inner_line, centroid)
        
        outer_sign = sign(outer_side_value)
        inner_sign = sign(inner_side_value)
        
        # Determine current region
        current_region = get_region(outer_sign, inner_sign, self.inside_sign)
        
        # Initialize tracking state if new
        if track_id not in self.track_states:
            # Determine origin based on first region seen
            origin = "unknown"
            if current_region == "outside":
                origin = "outside"
            elif current_region == "inside":
                origin = "inside"
            
            self.track_states[track_id] = {
                "region": current_region,
                "origin": origin,
                "outer_sign": outer_sign,
                "inner_sign": inner_sign,
                "prev_centroid": centroid,
                "last_update": self.frame_idx,
                "has_entered": False,
                "has_exited": False,
                "name": None,
                "skip_counting": False,
                "face_score": None,
                "face_candidate": None,
                "face_candidate_hits": 0
            }
            state = self.track_states[track_id]  # Define state for new tracks too

            # Try recognition immediately for new tracks to avoid counting known faces.
            if self.face_matcher is not None and self.face_matcher.backend is not None:
                name, score = self.face_matcher.recognize_person_bbox(frame, (x1, y1, x2, y2))
                if name:
                    self._update_face_candidate(track_id, state, name, score)

            color = (255, 200, 0)
        else:
            state = self.track_states[track_id]
            prev_region = state["region"]

            # Try to recognize the person (until recognized), then skip counting for that track.
            if (
                self.face_matcher is not None
                and self.face_matcher.backend is not None
                and not state.get("skip_counting", False)
                and state.get("name") is None
                and (self.frame_idx % max(1, FACE_RECOGNIZE_EVERY_N_FRAMES) == 0)
            ):
                name, score = self.face_matcher.recognize_person_bbox(frame, (x1, y1, x2, y2))
                if name:
                    self._update_face_candidate(track_id, state, name, score)
            
            # Update origin if unknown
            if state["origin"] == "unknown":
                if current_region == "outside":
                    state["origin"] = "outside"
                elif current_region == "inside":
                    state["origin"] = "inside"
            
            # Check for region change
            if current_region != prev_region:
                self.check_crossing(track_id, prev_region, current_region, centroid, state)
            
            # Update state
            state["region"] = current_region
            state["outer_sign"] = outer_sign
            state["inner_sign"] = inner_sign
            state["prev_centroid"] = centroid
            state["last_update"] = self.frame_idx
            
            if state["has_entered"]:
                color = (0, 255, 0)
            elif state["has_exited"]:
                color = (0, 0, 255)
            else:
                color = (255, 200, 0)

        if state.get("skip_counting", False) and state.get("name"):
            color = (255, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, centroid, 6, (0, 0, 255), -1)
        
        # Debug info
        origin_str = f"[{state['origin'][0].upper()}]" if state['origin'] != "unknown" else "[?]"
        name_str = f" {state['name']}" if state.get("name") else ""
        cv2.putText(frame, f"ID{track_id}{origin_str}{name_str}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, current_region.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Highlight selected track
        if self.selected_track_id is not None and int(track_id) == int(self.selected_track_id):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, "SELECTED (press E)", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    def _on_face_recognized(self, track_id, state, name, score):
        """
        Mark this track as a known person: label it and ensure it never affects counts.
        If this track was already counted before recognition, undo that count.
        """
        state["name"] = name
        state["skip_counting"] = True
        state["face_score"] = score

        # If we already counted this track, roll back to satisfy "known faces do not count".
        if state.get("has_entered") and track_id in self.counted_enter_ids:
            self.enter_count = max(0, self.enter_count - 1)
            self.counted_enter_ids.discard(track_id)
            state["has_entered"] = False

        if state.get("has_exited") and track_id in self.counted_exit_ids:
            self.exit_count = max(0, self.exit_count - 1)
            self.counted_exit_ids.discard(track_id)
            state["has_exited"] = False

        print(f"🙅 Known face (no count): {name} (ID-{track_id})")

    def _update_face_candidate(self, track_id, state, name, score):
        """
        Reduce false positives by requiring consecutive matches before confirming.
        """
        if state.get("skip_counting", False):
            return

        candidate = state.get("face_candidate")
        hits = int(state.get("face_candidate_hits") or 0)

        if candidate == name:
            hits += 1
        else:
            candidate = name
            hits = 1

        state["face_candidate"] = candidate
        state["face_candidate_hits"] = hits

        if hits >= max(1, int(FACE_CONFIRM_HITS)):
            self._on_face_recognized(track_id, state, name, score)

    def check_crossing(self, track_id, prev_region, current_region, centroid, state):
        """
        Check transitions including skipped regions (fast movement).
        """
        if state.get("skip_counting", False):
            return
        origin = state["origin"]
        
        # --- ENTRY LOGIC ---
        # Normal: Outside -> Between -> Inside
        # Fast:   Outside -> Inside
        # Skip:   Between -> Inside
        is_entry_path = False
        
        if current_region == "inside":
            if prev_region == "between" or prev_region == "outside":
                is_entry_path = True
        
        if is_entry_path:
            # Enforce Origin: Must NOT be 'inside'
            if origin != "inside":
                if not state["has_entered"] and track_id not in self.counted_enter_ids:
                    self.enter_count += 1
                    self.counted_enter_ids.add(track_id)
                    state["has_entered"] = True
                    print(f"✅ ENTER: ID-{track_id} (Origin: {origin})")
            else:
                # Debug log for ignored entry
                # print(f"Ignored Entry ID-{track_id}: Origin is Inside")
                pass

        # --- EXIT LOGIC ---
        # Normal: Inside -> Between -> Outside
        # Fast:   Inside -> Outside
        # Skip:   Between -> Outside
        is_exit_path = False
        
        if current_region == "outside":
            if prev_region == "between" or prev_region == "inside":
                is_exit_path = True
        
        if is_exit_path:
            # Enforce Origin: Must NOT be 'outside'
            if origin != "outside":
                if not state["has_exited"] and track_id not in self.counted_exit_ids:
                    self.exit_count += 1
                    self.counted_exit_ids.add(track_id)
                    state["has_exited"] = True
                    print(f"⬅️  EXIT:  ID-{track_id} (Origin: {origin})")
            else:
                # Debug log for ignored exit
                # print(f"Ignored Exit ID-{track_id}: Origin is Outside")
                pass

    def draw_zones(self, frame):
        """Draw lines and visualization of 'IN' direction."""
        if self.roi_zone is not None:
            cv2.polylines(frame, [self.roi_zone], True, (100, 100, 100), 1)
        
        # Draw lines
        cv2.line(frame, self.outer_line[:2], self.outer_line[2:], (0, 255, 0), 2)
        cv2.line(frame, self.inner_line[:2], self.inner_line[2:], (0, 0, 255), 2)
        
        # Calculate midpoints for arrow
        om = ((self.outer_line[0]+self.outer_line[2])//2, (self.outer_line[1]+self.outer_line[3])//2)
        im = ((self.inner_line[0]+self.inner_line[2])//2, (self.inner_line[1]+self.inner_line[3])//2)
        
        # Draw arrow showing "ENTRY" direction (Green -> Red)
        cv2.arrowedLine(frame, om, im, (255, 255, 0), 2, tipLength=0.2)
        cv2.putText(frame, "IN DIRECTION", ((om[0]+im[0])//2, (om[1]+im[1])//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def draw_statistics(self, frame):
        """Draw counting statistics on the frame."""
        # Background for stats
        cv2.rectangle(frame, (10, 10), (380, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (380, 140), (255, 255, 255), 2)
        
        # Enter count
        cv2.putText(frame, f"ENTERED: {self.enter_count}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Exit count
        cv2.putText(frame, f"EXITED:  {self.exit_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Current inside
        net = self.enter_count - self.exit_count
        cv2.putText(frame, f"INSIDE:  {net}", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Help
        cv2.putText(frame, "Click a person, press E to exclude", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Person Counter - YOLO + ByteTrack")
    print("Dual-Line System with Optional ROI")
    print("="*60)
    
    # Open video for drawing
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    # Draw ROI zone (optional)
    roi_zone = None
    if USE_ROI:
        roi_zone = draw_roi_zone(cap)
        if roi_zone is None:
            print("⚠️  ROI drawing cancelled, will count all detections")
    
    # Draw outer line (green - outside boundary)
    outer_line = draw_line(cap, "Outer", (0, 255, 0))
    if outer_line is None:
        print("❌ Outer line drawing cancelled")
        cap.release()
        return
    
    # Draw inner line (red - inside boundary)
    inner_line = draw_line(cap, "Inner", (0, 0, 255))
    if inner_line is None:
        print("❌ Inner line drawing cancelled")
        cap.release()
        return
    
    cap.release()
    
    # Create counter and process video
    counter = PersonCounter(VIDEO_PATH, OUTPUT_PATH, outer_line, inner_line, roi_zone)
    counter.process_video()


if __name__ == "__main__":
    main()
