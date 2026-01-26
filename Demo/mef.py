import os
import cv2
import warnings
import argparse
import logging
import numpy as np
import threading
import queue
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import faiss
from ultralytics import YOLO

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox

warnings.filterwarnings("ignore")


class FaceJob:
    """Job structure for face detection/recognition worker queue"""
    def __init__(self, track_id: int, crop: np.ndarray, max_num: int = 1):
        self.track_id = track_id
        self.crop = crop
        self.max_num = max_num


class FaceResult:
    """Result structure from face detection/recognition worker"""
    def __init__(self, track_id: int, name: str, similarity: float, embedding: Optional[np.ndarray] = None):
        self.track_id = track_id
        self.name = name
        self.similarity = similarity
        self.embedding = embedding


class PersistentIdentity:
    """Persistent identity cache for long-term ReID"""
    def __init__(
        self,
        name: str,
        embedding: np.ndarray,
        similarity: float,
        authorized: bool,
        first_seen: int,
        last_seen: int,
        body_features: Optional[np.ndarray] = None,
        display_id: Optional[int] = None,
    ):
        self.name = name
        self.embedding = embedding
        self.similarity = similarity
        self.authorized = authorized
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.appearances = 1  # Number of times this identity has appeared
        self.body_features = body_features  # Body appearance features for ReID when face is hidden
        self.display_id = display_id


def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition + YOLOv8 Multi-Object Tracking Pipeline")
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLOv8/YOLO11 model for person detection",
    )
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to face detection model (SCRFD)",
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to face recognition model (ArcFace)",
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.5,
        help="Similarity threshold for face recognition (cosine similarity)",
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLO person detection",
    )
    parser.add_argument(
        "--face-db",
        type=str,
        default="face_database",
        help="Path to FAISS face database (without extension, created by build_face_db.py)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        # default="rtsp://admin:deepminD1@192.168.0.2:554/Streaming/Channels/101",
        help="Video source: 0 for webcam or path to video file",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="botsort.yaml",
        help="Path to tracker configuration file (botsort.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on (cuda or cpu)",
    )
    parser.add_argument(
        "--max-face-num",
        type=int,
        default=0,
        help="Max number of face detections per person crop (0 = unlimited)",
    )
    parser.add_argument(
        "--identity-cache-frames",
        type=int,
        default=90,
        help="Number of frames to keep identity cached after track is lost (default: 90, ~3 seconds at 30fps)",
    )
    parser.add_argument(
        "--persistent-cache-seconds",
        type=int,
        default=300,
        help="Seconds to keep authorized identities in persistent cache for ReID (default: 300, 5 minutes)",
    )
    parser.add_argument(
        "--identity-confidence-decay-frames",
        type=int,
        default=150,
        help="Number of frames without face detection before identity confidence starts decaying (default: 150, ~5 seconds at 30fps)",
    )
    parser.add_argument(
        "--enable-reid",
        action="store_true",
        default=True,
        help="Enable Re-Identification to match returning people with cached identities",
    )
    parser.add_argument(
        "--reid-similarity-thresh",
        type=float,
        default=0.75,
        help="Similarity threshold for ReID matching (default: 0.75, higher = stricter matching to prevent false positives)",
    )
    parser.add_argument(
        "--enable-body-reid",
        action="store_true",
        default=True,
        help="Enable body-based Re-Identification when face is not visible (uses appearance features like clothing)",
    )
    parser.add_argument(
        "--body-reid-similarity-thresh",
        type=float,
        default=0.5,
        help="Similarity threshold for body-based ReID matching (default: 0.65, lower than face since body features are less distinctive)",
    )
    parser.add_argument(
        "--face-detection-interval",
        type=int,
        default=3,
        help="Run face detection every N frames to reduce latency (default: 3, more frequent for better reappearance detection)",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip displaying frames (faster processing, for saving video only)",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Path to save output video (optional)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--num-worker-threads",
        type=int,
        default=4,
        help="Number of worker threads for face detection/recognition (default: 4, increased for better throughput)",
    )
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def extract_body_features(crop: np.ndarray, grid_size: int = 4) -> Optional[np.ndarray]:
    """
    Extract appearance features from person crop for body-based ReID.
    Uses color histograms in HSV space with spatial grid for better discrimination.
    
    Args:
        crop: Person crop image (BGR format)
        grid_size: Divide image into grid_size x grid_size regions
    
    Returns:
        Feature vector combining color histograms from different body regions
    """
    if crop is None or crop.size == 0:
        return None
    
    h, w = crop.shape[:2]
    if h < 32 or w < 32:  # Too small to extract reliable features
        return None
    
    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Extract features from spatial grid (focus on torso/upper body)
    features = []
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Focus more on upper body (where clothing is more visible)
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_size - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_size - 1 else w
            
            cell = hsv[y1:y2, x1:x2]
            
            # Compute color histogram for this cell
            # H: 0-180, S: 0-255, V: 0-255
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
    
    # Concatenate all features
    feature_vector = np.concatenate(features)
    
    # Normalize to unit length
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    return feature_vector


def compare_body_features(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compare two body feature vectors using cosine similarity.
    
    Returns:
        Similarity score between 0 and 1
    """
    if feat1 is None or feat2 is None:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    return float(max(0.0, similarity))


class FaissGallery:
    """
    Face gallery using FAISS for efficient similarity search.
    Automatically uses GPU if available, otherwise falls back to CPU.
    """
    
    def __init__(self, database_path: str = "face_database"):
        """
        Load FAISS database created by build_face_db.py
        
        Args:
            database_path: Path to database files (without extension)
        """
        self.names = []
        self.stats = {}
        self.index = None
        self.use_gpu = False
        self.embedding_dim = 512
        
        faiss_path = f"{database_path}.faiss"
        names_path = f"{database_path}_names.pkl"
        
        # Check if database files exist
        if not os.path.exists(faiss_path):
            logging.warning(f"FAISS database not found: {faiss_path}")
            logging.warning("Please run build_face_db.py first to create the database.")
            return
        
        if not os.path.exists(names_path):
            logging.warning(f"Names file not found: {names_path}")
            return
        
        # Load FAISS index
        logging.info(f"Loading FAISS database from: {faiss_path}")
        cpu_index = faiss.read_index(faiss_path)
        
        # Try to use GPU if available
        try:
            # Check if GPU resources are available
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                logging.info(f"Found {ngpus} GPU(s), using GPU for FAISS")
                # Convert to GPU index
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                self.use_gpu = True
            else:
                logging.info("No GPU available, using CPU for FAISS")
                self.index = cpu_index
        except Exception as e:
            logging.warning(f"Could not initialize GPU FAISS: {e}")
            logging.info("Falling back to CPU FAISS")
            self.index = cpu_index
        
        # Load names and stats
        with open(names_path, "rb") as f:
            data = pickle.load(f)
            self.names = data.get("names", [])
            self.stats = data.get("stats", {})
        
        logging.info(f"Loaded FAISS gallery with {len(self.names)} identities (GPU: {self.use_gpu})")
        logging.info(f"Registered persons: {', '.join(self.names)}")
    
    def search(self, embedding: np.ndarray, k: int = 1) -> Tuple[str, float]:
        """
        Search for the closest match in the gallery.
        
        Args:
            embedding: Face embedding to search for
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (name, similarity_score)
        """
        if self.index is None or len(self.names) == 0:
            return "Unknown", 0.0
        
        # Normalize embedding for cosine similarity
        embedding = embedding.flatten().astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding = embedding.reshape(1, -1)
        
        # Search in FAISS (returns inner product which equals cosine similarity for normalized vectors)
        distances, indices = self.index.search(embedding, k)
        
        if len(indices) == 0 or indices[0][0] < 0:
            return "Unknown", 0.0
        
        best_idx = indices[0][0]
        best_similarity = float(distances[0][0])
        
        if best_idx < len(self.names):
            return self.names[best_idx], best_similarity
        
        return "Unknown", 0.0
    
    def __len__(self):
        return len(self.names)
    
    def is_loaded(self) -> bool:
        """Check if gallery is properly loaded."""
        return self.index is not None and len(self.names) > 0


def identify_face(
    embedding: np.ndarray, 
    gallery: FaissGallery, 
    threshold: float
) -> Tuple[str, float]:
    """
    Compare embedding against the FAISS gallery and return best match.
    Returns (name, similarity_score).
    """
    if gallery is None or not gallery.is_loaded():
        return "Unknown", 0.0
    
    name, similarity = gallery.search(embedding, k=1)
    return name, similarity


def detect_face_in_crop(
    crop: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    max_num: int = 1,
    min_crop_size: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect face in a cropped person image and return embedding.
    Returns (bbox, kps, embedding) if face found, None otherwise.
    """
    # Skip if crop is too small
    if crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
        return None
    
    bboxes, kpss = detector.detect(crop, max_num=max_num)
    
    if bboxes is None or len(bboxes) == 0 or kpss is None or len(kpss) == 0:
        return None
    
    # Use the first (most confident) detection
    bbox = bboxes[0]
    kps = kpss[0]
    
    # Generate embedding
    embedding = recognizer(crop, kps)
    
    return bbox, kps, embedding


# Worker thread that runs face detection + recognition
def face_worker_loop(job_q: queue.Queue, result_q: queue.Queue, face_detector, face_recognizer, face_gallery: FaissGallery, params):
    """Worker thread that processes face detection and recognition jobs."""
    while True:
        job: FaceJob = job_q.get()
        if job is None:
            break
        try:
            face_res = detect_face_in_crop(job.crop, face_detector, face_recognizer, max_num=job.max_num)
            if face_res is None:
                result_q.put(FaceResult(job.track_id, "Pending...", 0.0, None))
            else:
                _, _, embedding = face_res
                name, similarity = identify_face(embedding, face_gallery, params.similarity_thresh)
                # Apply threshold check
                if similarity < params.similarity_thresh:
                    name = "Unknown"
                result_q.put(FaceResult(job.track_id, name, float(similarity), embedding))
        except Exception as e:
            logging.exception("Face worker error")
            result_q.put(FaceResult(job.track_id, "Pending...", 0.0, None))
        finally:
            job_q.task_done()


def start_face_workers(n_threads: int, face_detector, face_recognizer, face_gallery: FaissGallery, params):
    """Start worker threads for face detection and recognition."""
    job_q = queue.Queue(maxsize=256)
    result_q = queue.Queue(maxsize=1024)
    threads = []
    for _ in range(n_threads):
        t = threading.Thread(
            target=face_worker_loop, 
            args=(job_q, result_q, face_detector, face_recognizer, face_gallery, params), 
            daemon=True
        )
        t.start()
        threads.append(t)
    return job_q, result_q, threads


def process_frame(
    frame: np.ndarray,
    yolo_model: YOLO,
    face_detector: SCRFD,
    face_recognizer: ArcFace,
    face_gallery: FaissGallery,
    track_identities: Dict[int, Dict],
    params: argparse.Namespace,
    frame_index: int,
    job_q: Optional[queue.Queue] = None,
    result_q: Optional[queue.Queue] = None,
    face_detection_interval: int = 5,
    pending_jobs: Optional[set] = None,
    persistent_cache: Optional[Dict[str, PersistentIdentity]] = None,
    fps: float = 30.0,
    name_to_display_id: Optional[Dict[str, int]] = None,
    yolo_to_display_id: Optional[Dict[int, int]] = None,
    next_display_id: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Process a single frame (CORE FUNCTIONALITY from face_yolo_pipeline.py with threading):
    1. Collect results from worker threads
    2. Run YOLO for person detection and tracking
    3. For each tracked person, submit face detection jobs to worker threads  
    4. Maintain identity persistence across frames
    5. Draw bounding boxes and labels
    """
    def resolve_display_id(track_id: int, name: Optional[str], authorized: bool, current_display_id: Optional[int]) -> Optional[int]:
        display_id = current_display_id
        valid_name = name not in (None, "Unknown", "Pending...")
        if authorized and valid_name and name_to_display_id is not None:
            if name in name_to_display_id:
                display_id = name_to_display_id[name]
            elif next_display_id is not None:
                display_id = next_display_id[0]
                next_display_id[0] += 1
                name_to_display_id[name] = display_id
        if display_id is None and yolo_to_display_id is not None:
            display_id = yolo_to_display_id.get(track_id)
        if display_id is None and next_display_id is not None:
            display_id = next_display_id[0]
            next_display_id[0] += 1
        if authorized and valid_name and name_to_display_id is not None and display_id is not None:
            name_to_display_id[name] = display_id
        if yolo_to_display_id is not None and display_id is not None:
            yolo_to_display_id[track_id] = display_id
        return display_id
    
    # STEP 1: Collect all available results from worker threads (non-blocking)
    if result_q is not None and pending_jobs is not None:
        while not result_q.empty():
            try:
                result = result_q.get_nowait()
                track_id = result.track_id
                identity = track_identities.get(track_id)
                
                # Remove from pending jobs
                pending_jobs.discard(track_id)
                
                if result.embedding is not None:
                    # Valid face detection result
                    authorized = result.similarity >= params.similarity_thresh
                    
                    # Update last face detection timestamp
                    if identity is None:
                        identity = track_identities.get(track_id, {})
                    identity["last_face_detected"] = frame_index
                    
                    # Check if this is a new track that might match a cached identity (ReID)
                    if params.enable_reid:
                        # Look for matching cached identity (even if identity exists, check for better match)
                        best_match_id = None
                        best_match_sim = 0.0
                        reid_thresh = getattr(params, "reid_similarity_thresh", 0.75)
                        
                        for cached_id, cached_info in track_identities.items():
                            # Skip current track
                            if cached_id == track_id:
                                continue
                            
                            # CRITICAL FIX: Only match against LOST/INACTIVE tracks
                            # Check if this cached track was recently seen (within last 10 frames)
                            frames_since_seen = frame_index - cached_info.get("last_seen", frame_index)
                            
                            # Only match if:
                            # 1. Track was lost recently (< identity_cache_frames/3, ~1 second)
                            # 2. Has valid embedding
                            # 3. Was authorized (don't match against random unauthorized people)
                            max_reid_gap = min(30, params.identity_cache_frames // 3)  # Max 1 second at 30fps
                            if frames_since_seen > max_reid_gap or frames_since_seen <= 0:
                                continue
                                
                            if cached_info.get("embedding") is None:
                                continue
                            
                            # Only match authorized identities to prevent merging different unauthorized people
                            if not cached_info.get("authorized", False):
                                continue
                            
                            # Compute similarity between new embedding and cached embedding
                            cached_emb = cached_info["embedding"]
                            sim = compute_similarity(result.embedding, cached_emb)
                            
                            # Use ReID threshold for matching
                            if sim >= reid_thresh and sim > best_match_sim:
                                best_match_sim = sim
                                best_match_id = cached_id
                        
                        # If we found a good match with a different track, merge identities
                        if best_match_id is not None and best_match_id != track_id:
                            cached_identity = track_identities[best_match_id]
                            
                            # Only merge if cached identity is better (authorized or higher similarity)
                            should_merge = (
                                cached_identity.get("authorized", False) or
                                cached_identity.get("similarity", 0.0) > result.similarity
                            )
                            
                            if should_merge:
                                frames_gap = frame_index - cached_identity.get("last_seen", frame_index)
                                logging.info(
                                    f"ReID: Track {track_id} matched with track {best_match_id} "
                                    f"'{cached_identity['name']}' (similarity: {best_match_sim:.2f}, "
                                    f"gap: {frames_gap} frames) - Merging identities"
                                )
                                # Copy the cached identity to new track with updated embedding
                                display_id = cached_identity.get("display_id")
                                track_identities[track_id] = {
                                    "name": cached_identity["name"],
                                    "similarity": max(cached_identity["similarity"], result.similarity),
                                    "authorized": cached_identity["authorized"],
                                    "last_seen": frame_index,
                                    "last_face_detected": frame_index,
                                    "embedding": result.embedding,  # Use new embedding
                                    "reid_matched": True,
                                    "merged_from": best_match_id,
                                    "confidence": 1.0,
                                    "display_id": display_id
                                }
                                if yolo_to_display_id is not None and display_id is not None:
                                    yolo_to_display_id[track_id] = display_id
                                # IMMEDIATELY remove the old track to prevent duplicate display
                                logging.info(f"ReID: Removing old track {best_match_id} to prevent duplicate display")
                                track_identities.pop(best_match_id, None)
                                if yolo_to_display_id is not None:
                                    yolo_to_display_id.pop(best_match_id, None)
                                continue
                    
                    # ENHANCED ReID: Check persistent cache for matches (for people returning after leaving completely)
                    if params.enable_reid and persistent_cache and authorized:
                        # Look for matching persistent identity
                        best_persistent_name = None
                        best_persistent_sim = 0.0
                        reid_thresh = getattr(params, "reid_similarity_thresh", 0.75)
                        
                        for cached_name, cached_identity in persistent_cache.items():
                            # Only match against authorized identities
                            if not cached_identity.authorized:
                                continue
                            
                            # Compute similarity with persistent cache
                            sim = compute_similarity(result.embedding, cached_identity.embedding)
                            
                            if sim >= reid_thresh and sim > best_persistent_sim:
                                best_persistent_sim = sim
                                best_persistent_name = cached_name
                        
                        # If we found a match in persistent cache, use that identity
                        if best_persistent_name is not None and result.name != best_persistent_name:
                            frames_gap = frame_index - persistent_cache[best_persistent_name].last_seen
                            time_gap_seconds = frames_gap / fps
                            logging.info(
                                f"ReID: Track {track_id} matched with persistent identity "
                                f"'{best_persistent_name}' (similarity: {best_persistent_sim:.2f}, "
                                f"gap: {time_gap_seconds:.1f}s) - Restoring identity"
                            )
                            result.name = best_persistent_name
                            # Update persistent cache appearance count
                            persistent_cache[best_persistent_name].appearances += 1
                            persistent_cache[best_persistent_name].last_seen = frame_index
                    
                    # Update identity if:
                    # 1. No existing identity, OR
                    # 2. New recognition is authorized, OR
                    # 3. Similarity is significantly better
                    should_update = (
                        identity is None or 
                        authorized or 
                        (identity is not None and result.similarity > identity.get("similarity", 0.0) + 0.1)
                    )
                    
                    if should_update:
                        resolved_name = result.name if authorized else result.name
                        current_display = identity.get("display_id") if identity else None
                        display_id = resolve_display_id(track_id, resolved_name, authorized, current_display)
                        track_identities[track_id] = {
                            "name": result.name if authorized else "Unknown",
                            "similarity": result.similarity,
                            "authorized": authorized,
                            "last_seen": frame_index,
                            "last_face_detected": frame_index,
                            "embedding": result.embedding,
                            "body_features": None,  # Will be updated separately
                            "reid_matched": False,
                            "confidence": 1.0,
                            "display_id": display_id
                        }
                        logging.debug(
                            f"Track {track_id}: Updated -> {result.name} "
                            f"(similarity: {result.similarity:.2f}, authorized: {authorized})"
                        )
                        identity = track_identities[track_id]
                        
                        # Add to persistent cache if authorized
                        if authorized and persistent_cache is not None:
                            if result.name not in persistent_cache:
                                persistent_cache[result.name] = PersistentIdentity(
                                    name=result.name,
                                    embedding=result.embedding,
                                    similarity=result.similarity,
                                    authorized=True,
                                    first_seen=frame_index,
                                    last_seen=frame_index,
                                    body_features=None,  # Will be updated with body features
                                    display_id=display_id
                                )
                                logging.info(f"Added '{result.name}' to persistent cache")
                            else:
                                # Update existing persistent cache with latest embedding
                                persistent_cache[result.name].embedding = result.embedding
                                persistent_cache[result.name].similarity = max(
                                    persistent_cache[result.name].similarity, 
                                    result.similarity
                                )
                                persistent_cache[result.name].last_seen = frame_index
                                if persistent_cache[result.name].display_id is None and display_id is not None:
                                    persistent_cache[result.name].display_id = display_id
                    elif identity is not None:
                        # Keep existing identity but update last_seen and embedding
                        identity["last_seen"] = frame_index
                        identity["last_face_detected"] = frame_index
                        identity["embedding"] = result.embedding  # Update with latest embedding
                        identity["confidence"] = 1.0  # Reset confidence when face is detected
                else:
                    # No face detected, but maintain existing identity with confidence decay
                    if identity is not None:
                        identity["last_seen"] = frame_index
                        
                        # Calculate frames since last face detection (only if we had a face before)
                        last_face_frame = identity.get("last_face_detected")
                        if last_face_frame is not None:
                            frames_without_face = frame_index - last_face_frame
                            
                            # Decay confidence after threshold
                            decay_threshold = getattr(params, "identity_confidence_decay_frames", 150)
                            if frames_without_face > decay_threshold:
                                decay_rate = 0.995  # Slow decay
                                current_confidence = identity.get("confidence", 1.0)
                                identity["confidence"] = max(0.3, current_confidence * decay_rate)
                                
                                if frames_without_face % 30 == 0:  # Log every second
                                    logging.debug(
                                        f"Track {track_id}: No face for {frames_without_face} frames, "
                                        f"confidence: {identity['confidence']:.2f}"
                                    )
                
                result_q.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Error processing face result: {e}")

    # STEP 2: Run YOLO tracking
    results = yolo_model.track(
        source=frame,
        persist=True,
        tracker=params.tracker_config,
        conf=params.yolo_conf,
        classes=[0],  # 0 = person class in COCO
        device=params.device,
        verbose=False,
        stream=False,
        half=True
    )
    result = results[0]

    # Check if there are any detections
    if result.boxes is None or len(result.boxes) == 0:
        # No detections, remove stale identities
        stale_ids = [
            track_id
            for track_id, info in track_identities.items()
            if frame_index - info.get("last_seen", frame_index) > params.identity_cache_frames
        ]
        for track_id in stale_ids:
            track_identities.pop(track_id, None)
            if yolo_to_display_id is not None:
                yolo_to_display_id.pop(track_id, None)
        return frame

    boxes = result.boxes
    
    # Get current active track IDs from YOLO
    current_track_ids = set()
    for box in boxes:
        if box.id is not None:
            current_track_ids.add(int(box.id[0].cpu().numpy()))
    
    # Remove identities for tracks that YOLO no longer sees (except recent ReID merges)
    tracks_to_remove = []
    for track_id in list(track_identities.keys()):
        if track_id not in current_track_ids:
            identity = track_identities[track_id]
            frames_since_seen = frame_index - identity.get("last_seen", frame_index)
            # Remove if not seen by YOLO and cache time expired (unless just merged)
            if frames_since_seen > 5 and not identity.get("reid_matched", False):
                tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        logging.debug(f"Removing track {track_id} - no longer detected by YOLO")
        track_identities.pop(track_id, None)
        if yolo_to_display_id is not None:
            yolo_to_display_id.pop(track_id, None)
    
    # STEP 3: Process each tracked person
    for i, box in enumerate(boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        
        # Get track ID
        track_id = None
        if box.id is not None:
            track_id = int(box.id[0].cpu().numpy())
        
        if track_id is None:
            continue
        
        # Get existing identity for this track
        identity = track_identities.get(track_id)
        base_name = identity.get("name") if identity else None
        base_authorized = bool(identity.get("authorized")) if identity else False
        base_display = identity.get("display_id") if identity else None
        display_id = resolve_display_id(track_id, base_name, base_authorized, base_display)
        if identity is not None:
            identity["display_id"] = display_id
        
        # Extract body features from person crop for ReID
        crop = frame[y1:y2, x1:x2]
        body_features = None
        if crop.size > 0 and params.enable_body_reid:
            body_features = extract_body_features(crop)
            
            # Store body features in current identity
            if identity is not None and body_features is not None:
                identity["body_features"] = body_features
                
                # Update persistent cache body features if authorized
                if identity.get("authorized") and persistent_cache is not None:
                    cached_name = identity.get("name")
                    if cached_name and cached_name in persistent_cache:
                        # Update with exponential moving average for stability
                        if persistent_cache[cached_name].body_features is None:
                            persistent_cache[cached_name].body_features = body_features
                        else:
                            alpha = 0.3  # Smoothing factor
                            old_features = persistent_cache[cached_name].body_features
                            if old_features is not None:
                                persistent_cache[cached_name].body_features = (
                                    alpha * body_features + (1 - alpha) * old_features
                                )
                        if (
                            persistent_cache[cached_name].display_id is None
                            and identity.get("display_id") is not None
                        ):
                            persistent_cache[cached_name].display_id = identity.get("display_id")
            
            # BODY-BASED ReID: If no identity or pending, try to match using body features
            if params.enable_body_reid and body_features is not None:
                if identity is None or identity.get("name") == "Pending..." or not identity.get("authorized", False):
                    # Try to match against persistent cache using body features
                    best_body_match_name = None
                    best_body_match_sim = 0.0
                    body_reid_thresh = getattr(params, "body_reid_similarity_thresh", 0.65)
                    
                    if persistent_cache:
                        for cached_name, cached_identity in persistent_cache.items():
                            if not cached_identity.authorized or cached_identity.body_features is None:
                                continue
                            
                            # Check if this identity was recently lost (within 30 seconds)
                            frames_gap = frame_index - cached_identity.last_seen
                            if frames_gap > 30 * fps:  # More than 30 seconds
                                continue
                            
                            # Compare body features
                            body_sim = compare_body_features(body_features, cached_identity.body_features)
                            
                            if body_sim >= body_reid_thresh and body_sim > best_body_match_sim:
                                best_body_match_sim = body_sim
                                best_body_match_name = cached_name
                        
                        # If we found a good body match, assign that identity
                        if best_body_match_name is not None:
                            time_gap = (frame_index - persistent_cache[best_body_match_name].last_seen) / fps
                            logging.info(
                                f"Body ReID: Track {track_id} matched with '{best_body_match_name}' "
                                f"(body similarity: {best_body_match_sim:.2f}, gap: {time_gap:.1f}s) - "
                                f"Face not visible but body recognized"
                            )
                            
                            # Assign the matched identity
                            cached = persistent_cache[best_body_match_name]
                            stable_display_id = resolve_display_id(
                                track_id,
                                best_body_match_name,
                                True,
                                cached.display_id,
                            )
                            cached.display_id = stable_display_id
                            track_identities[track_id] = {
                                "name": best_body_match_name,
                                "similarity": cached.similarity,  # Use original face similarity
                                "authorized": True,
                                "last_seen": frame_index,
                                "last_face_detected": None,  # Face not detected, using body
                                "embedding": cached.embedding,
                                "body_features": body_features,
                                "reid_matched": True,
                                "body_reid": True,  # Flag to indicate this was body-based match
                                "confidence": 0.85,  # Slightly lower confidence for body-only match
                                "display_id": stable_display_id
                            }
                            identity = track_identities[track_id]
                            cached.last_seen = frame_index
                            cached.appearances += 1
        
        # AGGRESSIVE face detection for new tracks and periodic checks for all tracks
        should_detect_face = (
            identity is None or  # New track - ALWAYS detect
            identity.get("name") == "Pending..." or  # Still pending - ALWAYS detect
            not identity.get("authorized", False) or  # Not yet authorized - ALWAYS detect
            identity.get("reid_matched", False) or  # ReID matched - verify with fresh detection
            frame_index % face_detection_interval == 0  # Periodic check for all tracks
        )
        
        if should_detect_face:
            # Crop person region
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Submit job to worker thread (non-blocking) - but skip if already pending
            if job_q is not None and pending_jobs is not None:
                # Check if this track already has a pending job
                if track_id not in pending_jobs:
                    try:
                        max_face = params.max_face_num if params.max_face_num > 0 else 1
                        job = FaceJob(track_id, crop.copy(), max_num=max_face)
                        job_q.put_nowait(job)
                        pending_jobs.add(track_id)
                        logging.debug(f"Submitted face job for track {track_id}")
                    except queue.Full:
                        # Queue is full - this is the problem!
                        # For critical new/pending tracks, try harder
                        if identity is None or identity.get("name") == "Pending...":
                            logging.warning(
                                f"Job queue FULL ({job_q.qsize()}/{job_q.maxsize}) - "
                                f"Cannot process NEW track {track_id}. Consider increasing worker threads!"
                            )
                        else:
                            logging.debug(f"Job queue full, skipping track {track_id}")
                else:
                    logging.debug(f"Track {track_id} already has pending face detection job")
            
            # If no existing identity, create a pending one
            if identity is None:
                identity = {
                    "name": "Pending...",
                    "similarity": 0.0,
                    "authorized": False,
                    "last_seen": frame_index,
                    "last_face_detected": None,
                    "embedding": None,
                    "body_features": None,
                    "confidence": 1.0,
                    "display_id": display_id
                }
                track_identities[track_id] = identity
                logging.debug(f"Track {track_id}: New track, waiting for face detection")
        else:
            # Skip face detection this frame, just maintain existing identity
            if identity is not None:
                identity["last_seen"] = frame_index
                
                # Apply confidence decay if no face detected for a while (only if we had a face before)
                last_face_frame = identity.get("last_face_detected")
                if last_face_frame is not None:
                    frames_without_face = frame_index - last_face_frame
                    decay_threshold = getattr(params, "identity_confidence_decay_frames", 150)
                    if frames_without_face > decay_threshold:
                        decay_rate = 0.995
                        current_confidence = identity.get("confidence", 1.0)
                        identity["confidence"] = max(0.3, current_confidence * decay_rate)
            else:
                # New track, will detect face on next interval
                identity = {
                    "name": "Pending...",
                    "similarity": 0.0,
                    "authorized": False,
                    "last_seen": frame_index,
                    "last_face_detected": None,
                    "embedding": None,
                    "body_features": None,
                    "confidence": 1.0,
                    "display_id": display_id
                }
                track_identities[track_id] = identity
        
        # STEP 4: Draw bounding box and label
        color = (0, 255, 0) if identity["authorized"] else (0, 0, 255)
        status = "Authorized" if identity["authorized"] else "Unauthorized"
        
        # Special indicator for body-based ReID
        if identity.get("body_reid", False):
            status = "Authorized (Body)"
        
        # Adjust color based on confidence (dim if low confidence)
        confidence = identity.get("confidence", 1.0)
        if confidence < 0.7:
            # Blend with gray for low confidence
            color = tuple(int(c * confidence + 128 * (1 - confidence)) for c in color)
        
        # Draw box
        draw_bbox(frame, (x1, y1, x2, y2), color)
        
        display_label_id = identity.get("display_id", track_id)
        # Draw label with confidence indicator if not full confidence
        if confidence < 0.95 and identity["name"] != "Pending...":
            label = f"{status} | ID:{display_label_id} {identity['name']} [{confidence:.0%}]"
        else:
            label = f"{status} | ID:{display_label_id} {identity['name']}"
        
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    
    # STEP 5: Remove stale identities (tracks lost for too long)
    stale_ids = [
        track_id
        for track_id, info in track_identities.items()
        if frame_index - info.get("last_seen", frame_index) > params.identity_cache_frames
    ]
    for track_id in stale_ids:
        logging.info(f"Removing stale track {track_id} ({track_identities[track_id]['name']})")
        track_identities.pop(track_id, None)
        if yolo_to_display_id is not None:
            yolo_to_display_id.pop(track_id, None)
    
    return frame


def main(params):
    setup_logging(params.log_level)
    
    logging.info("=" * 60)
    logging.info("Face Recognition + YOLOv8 Multi-Object Tracking Pipeline")
    logging.info("=" * 60)
    
    # Load YOLO model for person detection and tracking
    logging.info(f"Loading YOLO model: {params.yolo_model}")
    yolo_model = YOLO(params.yolo_model)
    
    # Load face detection model (SCRFD)
    logging.info(f"Loading face detection model: {params.det_weight}")
    face_detector = SCRFD(
        params.det_weight,
        input_size=(640, 640),
        conf_thres=params.confidence_thresh
    )
    
    # Load face recognition model (ArcFace)
    logging.info(f"Loading face recognition model: {params.rec_weight}")
    face_recognizer = ArcFace(params.rec_weight)
    
    # Load FAISS face gallery (created by build_face_db.py)
    logging.info(f"Loading FAISS face gallery from: {params.face_db}")
    face_gallery = FaissGallery(database_path=params.face_db)
    
    if not face_gallery.is_loaded():
        logging.error("Failed to load face gallery! Please run build_face_db.py first.")
        logging.error("Example: python build_face_db.py --faces-folder face_photos --output face_database")
        return
    
    # Start face worker threads
    logging.info(f"Starting {params.num_worker_threads} face worker threads")
    job_q, result_q, worker_threads = start_face_workers(
        params.num_worker_threads, 
        face_detector, 
        face_recognizer, 
        face_gallery, 
        params
    )
    
    # Open video source
    logging.info(f"Opening video source: {params.source}")
    source = int(params.source) if params.source.isdigit() else params.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {params.source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        width, height = 640, 480
    
    logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
    
    # Initialize video writer - always save output with unique timestamp
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tracking_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Use user-specified path if provided, otherwise use auto-generated path
    if params.save_output:
        output_path = params.save_output
        logging.info(f"Saving output to: {output_path}")
    else:
        logging.info(f"Auto-saving output to: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Dictionary to maintain track identities
    # Format: {track_id: {"name": str, "similarity": float, "authorized": bool, "last_seen": int, "embedding": np.ndarray}}
    track_identities: Dict[int, Dict] = {}
    
    # Persistent identity cache for long-term ReID (keyed by name)
    persistent_cache: Dict[str, PersistentIdentity] = {}
    # Stable ID mappings
    name_to_display_id: Dict[str, int] = {}
    yolo_to_display_id: Dict[int, int] = {}
    next_display_id: List[int] = [1]
    
    # Set to track pending face detection jobs (prevent duplicate submissions)
    pending_jobs: set = set()
    
    frame_index = 0
    
    logging.info("Starting video processing... Press 'q' to quit")
    logging.info(f"Worker threads: {params.num_worker_threads}")
    logging.info(f"Face detection interval: {params.face_detection_interval} frames")
    logging.info("=" * 60)
    
    # Performance tracking
    import time
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video stream")
                break
            
            # Process frame
            processed_frame = process_frame(
                frame,
                yolo_model,
                face_detector,
                face_recognizer,
                face_gallery,
                track_identities,
                params,
                frame_index,
                job_q=job_q,
                result_q=result_q,
                face_detection_interval=params.face_detection_interval,
                pending_jobs=pending_jobs,
                persistent_cache=persistent_cache,
                fps=fps,
                name_to_display_id=name_to_display_id,
                yolo_to_display_id=yolo_to_display_id,
                next_display_id=next_display_id
            )
            
            # Write to output video (always enabled now)
            out.write(processed_frame)
            
            # Display frame (unless skipped for performance)
            if not params.skip_display:
                # Add FPS counter to frame
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    current_fps = fps_counter / max(0.001, time.time() - fps_start_time)
                
                # Draw FPS on frame
                cv2.putText(
                    processed_frame,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                # Draw queue status
                cv2.putText(
                    processed_frame,
                    f"Queue: {job_q.qsize()}/{job_q.maxsize}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                
                cv2.imshow("Face Recognition + YOLO Tracking (Optimized)", processed_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User requested quit")
                    break
            
            frame_index += 1
            
            # Clean up expired persistent cache entries
            if frame_index % 300 == 0:  # Check every 10 seconds at 30fps
                cache_timeout_frames = int(getattr(params, "persistent_cache_seconds", 300) * fps)
                expired_names = [
                    name for name, identity in persistent_cache.items()
                    if frame_index - identity.last_seen > cache_timeout_frames
                ]
                for name in expired_names:
                    logging.info(f"Removing expired identity '{name}' from persistent cache")
                    persistent_cache.pop(name)
            
            # Log progress every 100 frames
            if frame_index % 100 == 0:
                active_tracks = len(track_identities)
                authorized_tracks = sum(1 for info in track_identities.values() if info.get("authorized", False))
                pending_tracks = sum(1 for info in track_identities.values() if info.get("name") == "Pending...")
                queue_usage_pct = (job_q.qsize() / job_q.maxsize) * 100 if job_q.maxsize > 0 else 0
                cache_size = len(persistent_cache)
                
                log_msg = (
                    f"Frame {frame_index}: {active_tracks} tracks "
                    f"({authorized_tracks} authorized, {pending_tracks} pending) | "
                    f"Persistent cache: {cache_size} identities | "
                    f"Queue: {job_q.qsize()}/{job_q.maxsize} ({queue_usage_pct:.0f}% full)"
                )
                
                # Warning if queue is getting full
                if queue_usage_pct > 80:
                    logging.warning(f"{log_msg} - QUEUE NEARLY FULL! Increase --num-worker-threads")
                else:
                    logging.info(log_msg)
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    
    finally:
        # Cleanup
        logging.info("Cleaning up...")
        
        # Stop worker threads
        logging.info("Stopping worker threads...")
        for _ in range(params.num_worker_threads):
            job_q.put(None)
        for t in worker_threads:
            t.join(timeout=2.0)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logging.info(f"Video saved to: {output_path}")
        
        logging.info("=" * 60)
        logging.info(f"Processing complete. Total frames processed: {frame_index}")
        logging.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
 