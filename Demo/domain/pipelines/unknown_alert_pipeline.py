import base64
import queue
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import cv2
import imageio
import numpy as np
from asgiref.sync import async_to_sync
from celery import states
from celery.result import AsyncResult
from channels.layers import get_channel_layer
from django.conf import settings
from pydantic import BaseModel, InstanceOf

from domain.model import AlertEvent, StreamActivityData, StreamStatus
from domain.repository import (
    AlertEventRepository,
    EmployeeRepository,
    FaceVectorRepository,
    RecordedStreamRepository,
    StreamRepository,
)
from domain.service import HumanFaceDetection, HumanFaceEmbedding, HumanTracking
from utils import get_logger, open_video_capture, resolve_local_media_path
from utils.helpers import compute_similarity, draw_bbox

from .pipeline import Pipeline

logger = get_logger(__name__)


class FaceJob:
    """Work item for async face detection/recognition."""

    def __init__(self, track_id: int, crop: np.ndarray, origin: Tuple[int, int], max_num: int = 1):
        self.track_id = track_id
        self.crop = crop
        self.origin = origin
        self.max_num = max_num


class FaceResult:
    """Result produced by a face worker."""

    def __init__(
        self,
        track_id: int,
        employee_id: str,
        name: str,
        similarity: float,
        embedding: Optional[np.ndarray] = None,
        face_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.track_id = track_id
        self.employee_id = employee_id
        self.name = name
        self.similarity = similarity
        self.embedding = embedding
        self.face_box = face_box


class PersistentIdentity:
    """Longer-lived cache for ReID across track drops."""

    def __init__(
        self,
        name: str,
        employee_id: Optional[str],
        embedding: np.ndarray,
        similarity: float,
        authorized: bool,
        first_seen: int,
        last_seen: int,
        body_features: Optional[np.ndarray] = None,
        display_id: Optional[int] = None,
    ):
        self.name = name
        self.employee_id = employee_id
        self.embedding = embedding
        self.similarity = similarity
        self.authorized = authorized
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.appearances = 1
        self.body_features = body_features
        self.display_id = display_id


def extract_body_features(crop: np.ndarray, grid_size: int = 4) -> Optional[np.ndarray]:
    """Extract appearance features from a person crop for body-based ReID."""
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 32 or w < 32:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    features = []
    cell_h = h // grid_size
    cell_w = w // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_size - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_size - 1 else w

            cell = hsv[y1:y2, x1:x2]
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)

    feature_vector = np.concatenate(features)
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm

    return feature_vector


def compare_body_features(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Cosine similarity between two body feature vectors."""
    if feat1 is None or feat2 is None:
        return 0.0
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
    return float(max(0.0, similarity))


class UnknownAlertPipeline(BaseModel, Pipeline):
    face_detection: InstanceOf[HumanFaceDetection]
    face_embedding_generator: InstanceOf[HumanFaceEmbedding]
    human_tracking: InstanceOf[HumanTracking]
    face_vector_db: InstanceOf[FaceVectorRepository]
    employee_db: InstanceOf[EmployeeRepository]
    stream_db: InstanceOf[StreamRepository]
    recorded_stream_db: InstanceOf[RecordedStreamRepository]
    alert_event_db: InstanceOf[AlertEventRepository]
    is_debugging: bool = False
    count: int = 0
    unknown_presence_threshold: float = 10.0
    face_detection_interval: int = 3  # Run face detection every N frames
    identity_cache_frames: int = 30  # Keep identities for this many frames without updates
    similarity_threshold: float = 0.35  # Minimum similarity for positive match
    num_worker_threads: int = 4
    max_face_num: int = 0
    enable_reid: bool = True
    reid_similarity_thresh: float = 0.75
    enable_body_reid: bool = True
    body_reid_similarity_thresh: float = 0.5
    persistent_cache_seconds: int = 300
    identity_confidence_decay_frames: int = 150
    crop_expand_ratio: float = 0.35  # Expand person crop before face detection for robustness

    def process(self, stream_data: StreamActivityData, task_id: str | None = None):
        logger.info("Processing unknown alert pipeline for %s", stream_data.stream_id)
        global stream
        channel_layer = get_channel_layer()

        # Track identities: {track_id: {name, employee_id, similarity, authorized, last_seen, embedding}}
        track_identities: dict[int, dict] = {}
        # Track unknown person durations for alerts
        unknown_first_seen: dict[int, float] = {}
        unknown_alerted_tracks: set[int] = set()
        # Persistent identity cache for ReID
        persistent_cache: Dict[str, PersistentIdentity] = {}
        # Stable ID mappings for consistent labels
        name_to_display_id: Dict[str, int] = {}
        yolo_to_display_id: Dict[int, int] = {}
        next_display_id: List[int] = [1]
        track_history = deque(maxlen=200)
        pending_jobs: set[int] = set()
        worker_threads: List[threading.Thread] = []
        job_q: Optional[queue.Queue] = None
        result_q: Optional[queue.Queue] = None
        gallery: List[dict] = []

        # Load all employees
        try:
            all_employees = {
                str(emp.employee_id): f"{emp.first_name} {emp.last_name}".strip()
                for emp in self.employee_db.get_all_employees()
            }
        except Exception as exc:
            logger.error("Unable to fetch employees: %s", exc)
            all_employees = {}

        # Load stream metadata
        stream_info = None
        recorded_stream_info = None
        normalized_stream_id: UUID | None = None
        if stream_data.stream_id:
            normalized_stream_id = (
                stream_data.stream_id
                if isinstance(stream_data.stream_id, UUID)
                else UUID(str(stream_data.stream_id))
            )

        if normalized_stream_id:
            try:
                stream_info = self.stream_db.get_stream(normalized_stream_id)
            except Exception as exc:
                logger.error("Failed to load stream metadata: %s", exc)
                stream_info = None

        if not stream_info and normalized_stream_id:
            try:
                recorded_stream_info = self.recorded_stream_db.get_recorded_stream(normalized_stream_id)
                if recorded_stream_info and recorded_stream_info.stream_id:
                    linked_stream_id = (
                        recorded_stream_info.stream_id
                        if isinstance(recorded_stream_info.stream_id, UUID)
                        else UUID(str(recorded_stream_info.stream_id))
                    )
                    stream_info = self.stream_db.get_stream(linked_stream_id)
            except Exception as exc:
                logger.error("Unable to fetch recorded stream metadata: %s", exc)

        branch_id = getattr(stream_info, "branch_id", None)
        floor_id = getattr(stream_info, "floor_id", None)
        camera_label = getattr(stream_info, "stream_name", None) or getattr(recorded_stream_info, "camera_name", None)

        stream_source = resolve_local_media_path(stream_data.stream_url)
        if stream_source is None:
            logger.error("Stream path %s could not be resolved", stream_data.stream_url)
            return

        if not stream_data.processed_uri:
            logger.error("Processed URI missing for stream %s", stream_data.stream_id)
            return

        processed_output_path = Path(stream_data.processed_uri)
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load known face embeddings once and build gallery
        known_embeddings = self.face_vector_db.get_face_vectors()
        gallery = self._build_face_gallery(known_embeddings, all_employees)
        job_q, result_q, worker_threads = self._start_face_workers(gallery)

        try:
            stream = open_video_capture(stream_source)
            if not stream:
                raise FileNotFoundError(f"Failed to load video from {stream_source}")

            fps = max(stream.get(cv2.CAP_PROP_FPS), 1)
            writer = imageio.get_writer(str(processed_output_path), fps=fps)

            while True:
                ret, frame = stream.read()
                if not ret:
                    logger.info("Stream ended for %s", stream_source)
                    break

                # Check for task cancellation
                if task_id is not None:
                    task = AsyncResult(task_id)
                    if task.state == states.REVOKED:
                        logger.info("Task %s revoked. Stopping stream processing.", task_id)
                        stream_data.success = True
                        stream_data.status = StreamStatus.STOPPED
                        break

                frame = self._rescale_frame(frame)
                self.count += 1
                current_time = self.count / fps

                # Process frame
                try:
                    frame, metadata_presence, metadata_alerts = self._process_frame(
                        frame=frame,
                        frame_index=self.count,
                        fps=fps,
                        current_time=current_time,
                        track_identities=track_identities,
                        unknown_first_seen=unknown_first_seen,
                        unknown_alerted_tracks=unknown_alerted_tracks,
                        all_employees=all_employees,
                        stream_data=stream_data,
                        branch_id=branch_id,
                        floor_id=floor_id,
                        camera_label=camera_label,
                        job_q=job_q,
                        result_q=result_q,
                        pending_jobs=pending_jobs,
                    persistent_cache=persistent_cache,
                    name_to_display_id=name_to_display_id,
                    yolo_to_display_id=yolo_to_display_id,
                    next_display_id=next_display_id,
                    track_history=track_history,
                )
                except Exception:
                    logger.exception("Unexpected error while processing frame %s", self.count)
                    continue

                # Write frame to output (convert to RGB to avoid channel swapping)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Encode frame for broadcast
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

                # Prepare metadata payload
                metadata_payload = {}
                if metadata_presence:
                    metadata_payload["presence"] = metadata_presence
                if metadata_alerts:
                    metadata_payload["unknown_alerts"] = metadata_alerts

                # Count known and unknown faces
                known_faces = {info["name"] for info in track_identities.values() if info.get("authorized", False)}
                unknown_count = sum(1 for info in track_identities.values() if not info.get("authorized", False))

                # Broadcast to clients
                broadcast_targets = {str(stream_data.stream_id)}
                if stream_data.event_id:
                    broadcast_targets.add(str(stream_data.event_id))

                for target in broadcast_targets:
                    message = {
                        'type': 'send_frame',
                        'frame': frame_data,
                        'unknown': unknown_count,
                    }
                    if metadata_payload:
                        message['metadata'] = metadata_payload
                    if known_faces:
                        message['known'] = sorted(known_faces)

                    async_to_sync(channel_layer.group_send)(
                        f"video_stream_{target}",
                        message,
                    )

                if self.count % 300 == 0:
                    cache_timeout_frames = int(self.persistent_cache_seconds * fps)
                    expired_keys = [
                        key
                        for key, identity in persistent_cache.items()
                        if self.count - identity.last_seen > cache_timeout_frames
                    ]
                    for key in expired_keys:
                        logger.info("Removing expired identity %s from persistent cache", key)
                        persistent_cache.pop(key, None)

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            stream_data.success = False
            stream_data.errors = [str(fnf_error)]
            return stream_data
        except Exception as exc:
            logger.exception("Unexpected error while running unknown alert pipeline")
            stream_data.success = False
            stream_data.errors = ["Unexpected error while loading the stream."]
            return stream_data
        finally:
            if 'stream' in locals() and stream.isOpened():
                stream.release()
                logger.info("Released the video resource.")
            if 'writer' in locals():
                writer.close()
            if job_q is not None:
                for _ in range(self.num_worker_threads):
                    job_q.put(None)
            for t in worker_threads:
                t.join(timeout=2.0)

    def _build_face_gallery(self, known_embeddings, all_employees: dict[str, str]) -> List[dict]:
        """Decode stored embeddings into a gallery for fast similarity lookups."""
        gallery: List[dict] = []
        for vector in known_embeddings:
            if getattr(vector, "vector_data", None) is None:
                continue
            try:
                decoded_bytes = base64.b64decode(vector.vector_data)
                embedding = np.frombuffer(decoded_bytes, dtype=np.float32)
            except Exception as exc:
                logger.warning("Skipping invalid face vector: %s", exc)
                continue

            if embedding.size == 0:
                continue

            employee_id = str(vector.employee_id) if getattr(vector, "employee_id", None) else "Unknown"
            name = all_employees.get(employee_id, "Unknown")
            gallery.append({"embedding": embedding, "employee_id": employee_id, "name": name})

        logger.info("Loaded %s face embeddings into gallery", len(gallery))
        return gallery

    def _start_face_workers(self, gallery: List[dict]):
        job_q: queue.Queue = queue.Queue(maxsize=256)
        result_q: queue.Queue = queue.Queue(maxsize=1024)
        threads: List[threading.Thread] = []
        for _ in range(self.num_worker_threads):
            t = threading.Thread(
                target=self._face_worker_loop,
                args=(job_q, result_q, gallery),
                daemon=True,
            )
            t.start()
            threads.append(t)
        return job_q, result_q, threads

    def _face_worker_loop(self, job_q: queue.Queue, result_q: queue.Queue, gallery: List[dict]):
        """Worker thread: detect a face in the crop and identify the employee."""
        while True:
            job: FaceJob = job_q.get()
            if job is None:
                break
            try:
                detection = self._detect_face_in_crop(job.crop, max_faces=job.max_num)
                if detection is None:
                    result_q.put(FaceResult(job.track_id, "Unknown", "Pending...", 0.0, None))
                else:
                    bbox, embedding = detection
                    employee_id, name, similarity = self._identify_employee_from_embedding(embedding, gallery)
                    face_box = (
                        job.origin[0] + int(bbox.x),
                        job.origin[1] + int(bbox.y),
                        int(bbox.width),
                        int(bbox.height),
                    )
                    result_q.put(
                        FaceResult(
                            job.track_id,
                            employee_id,
                            name,
                            float(similarity),
                            embedding,
                            face_box,
                        )
                    )
            except Exception as exc:
                logger.exception("Face worker error")
                result_q.put(FaceResult(job.track_id, "Unknown", "Pending...", 0.0, None))
            finally:
                job_q.task_done()

    def _detect_face_in_crop(self, crop: np.ndarray, max_faces: int = 1):
        """Detect face within a person crop and return bbox with embedding."""
        if crop is None or crop.size == 0:
            return None

        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return None

        # Pad crop to increase detection robustness (aligns closer to pipeline_v3 full-frame usage)
        h, w = crop.shape[:2]
        pad_y = int(h * 0.3)
        pad_x = int(w * 0.3)
        padded = cv2.copyMakeBorder(
            crop,
            pad_y,
            pad_y,
            pad_x,
            pad_x,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        try:
            face_detections = self.face_detection.predict(padded, max_faces=0 if max_faces == 0 else max_faces)
        except TypeError:
            face_detections = self.face_detection.predict(padded)

        if not face_detections:
            return None

        face = face_detections[0]
        embedding = self.face_embedding_generator.predict(padded, face.key_points)
        if embedding is None or len(embedding) == 0:
            return None

        # Adjust bbox back to original crop coordinates
        bbox = face.bounding_boxes
        bbox.x = max(0, bbox.x - pad_x)
        bbox.y = max(0, bbox.y - pad_y)
        return bbox, embedding

    def _identify_employee_from_embedding(
        self, embedding: np.ndarray, gallery: List[dict]
    ) -> tuple[str, str, float]:
        """Identify employee based on embedding similarity against the gallery."""
        if embedding is None or embedding.size == 0 or not gallery:
            return "Unknown", "Unknown", 0.0

        best_similarity = 0.0
        matched_employee_id = "Unknown"
        matched_name = "Unknown"

        for candidate in gallery:
            similarity = self._compute_similarity(candidate["embedding"], embedding)
            if similarity > best_similarity and similarity >= 0.35:
                best_similarity = similarity
                matched_employee_id = candidate["employee_id"]
                matched_name = candidate["name"]

        return matched_employee_id, matched_name, float(best_similarity)

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        fps: float,
        current_time: float,
        track_identities: dict[int, dict],
        unknown_first_seen: dict[int, float],
        unknown_alerted_tracks: set[int],
        all_employees: dict[str, str],
        stream_data: StreamActivityData,
        branch_id,
        floor_id,
        camera_label,
        job_q: Optional[queue.Queue],
        result_q: Optional[queue.Queue],
        pending_jobs: set[int],
        persistent_cache: Dict[str, PersistentIdentity],
        name_to_display_id: Dict[str, int],
        yolo_to_display_id: Dict[int, int],
        next_display_id: List[int],
        track_history,
    ) -> tuple[np.ndarray, list, list]:
        """
        Process a single frame:
        1. Drain worker results (face detection/recognition)
        2. Run human tracking
        3. Submit new face jobs when needed
        4. Handle ReID/persistence and alerts
        5. Draw overlays and return metadata
        """
        metadata_presence = []
        metadata_alerts = []
        frame_for_crops = frame.copy()

        def resolve_display_id(track_id: int, name: Optional[str], authorized: bool, current_display_id: Optional[int]):
            display_id = current_display_id
            valid_name = name not in (None, "Unknown", "Pending...")
            if authorized and valid_name:
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

        def reuse_display_id_from_iou(track_id: int, bbox: tuple[int, int, int, int], default_display: Optional[int]):
            """Reuse an existing display id when tracker IDs flicker, based on IoU with prior boxes."""
            if default_display is not None:
                return default_display
            if not track_identities:
                return default_display
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            best_disp = None
            best_iou = 0.0
            for tid, info in track_identities.items():
                ib = info.get("bbox")
                if not ib:
                    continue
                ix, iy, iw, ih = ib
                ix2 = ix + iw
                iy2 = iy + ih
                inter_x1 = max(x, ix)
                inter_y1 = max(y, iy)
                inter_x2 = min(x2, ix2)
                inter_y2 = min(y2, iy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area == 0:
                    continue
                area_a = w * h
                area_b = iw * ih
                union = area_a + area_b - inter_area
                if union <= 0:
                    continue
                iou = inter_area / union
                if iou > 0.6 and iou > best_iou:
                    best_iou = iou
                    best_disp = info.get("display_id")
            if best_disp is not None:
                yolo_to_display_id[track_id] = best_disp
                return best_disp
            return default_display

        def reuse_display_id_from_history(bbox: tuple[int, int, int, int], frame_idx: int):
            """Fallback to recent history when tracker emits a brand new ID."""
            if not track_history:
                return None
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            best_disp = None
            best_iou = 0.0
            for item in track_history:
                hx, hy, hw, hh, disp_id, seen_frame = item
                if frame_idx - seen_frame > max(15, self.identity_cache_frames // 2):
                    continue
                hx2 = hx + hw
                hy2 = hy + hh
                inter_x1 = max(x, hx)
                inter_y1 = max(y, hy)
                inter_x2 = min(x2, hx2)
                inter_y2 = min(y2, hy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area == 0:
                    continue
                area_a = w * h
                area_b = hw * hh
                union = area_a + area_b - inter_area
                if union <= 0:
                    continue
                iou = inter_area / union
                if iou > 0.55 and iou > best_iou:
                    best_iou = iou
                    best_disp = disp_id
            return best_disp

        # Step 1: collect worker results
        if result_q is not None and pending_jobs is not None:
            while not result_q.empty():
                try:
                    result = result_q.get_nowait()
                    track_id = result.track_id
                    identity = track_identities.get(track_id)

                    pending_jobs.discard(track_id)

                    # Reject detections whose face center falls outside the track's inner region;
                    # mitigates cross-assignment when person boxes overlap.
                    if identity and identity.get("bbox") and result.face_box:
                        face_cx = result.face_box[0] + result.face_box[2] / 2
                        face_cy = result.face_box[1] + result.face_box[3] / 2
                        bx, by, bw, bh = identity["bbox"]
                        margin_x = bw * 0.15
                        margin_y = bh * 0.15
                        if not (bx + margin_x <= face_cx <= bx + bw - margin_x and by + margin_y <= face_cy <= by + bh - margin_y):
                            result_q.task_done()
                            continue

                    if result.embedding is not None:
                        authorized = (
                            result.employee_id != "Unknown"
                            and result.similarity >= self.similarity_threshold
                        )

                        if identity is None:
                            identity = track_identities.get(track_id, {})
                        identity["last_face_detected"] = frame_index

                        # ReID with recently lost tracks
                        if self.enable_reid:
                            best_match_id = None
                            best_match_sim = 0.0
                            max_reid_gap = min(30, self.identity_cache_frames // 3)

                            for cached_id, cached_info in track_identities.items():
                                if cached_id == track_id:
                                    continue
                                frames_since_seen = frame_index - cached_info.get("last_seen", frame_index)
                                if frames_since_seen > max_reid_gap or frames_since_seen <= 0:
                                    continue
                                if cached_info.get("embedding") is None:
                                    continue
                                if not cached_info.get("authorized", False):
                                    continue

                                sim = compute_similarity(result.embedding, cached_info["embedding"])
                                if sim >= self.reid_similarity_thresh and sim > best_match_sim:
                                    best_match_sim = sim
                                    best_match_id = cached_id

                            if best_match_id is not None and best_match_id != track_id:
                                cached_identity = track_identities[best_match_id]
                                should_merge = (
                                    cached_identity.get("authorized", False)
                                    or cached_identity.get("similarity", 0.0) > result.similarity
                                )
                                if should_merge:
                                    display_id = cached_identity.get("display_id")
                                    track_identities[track_id] = {
                                        "name": cached_identity["name"],
                                        "employee_id": cached_identity.get("employee_id"),
                                        "similarity": max(cached_identity.get("similarity", 0.0), result.similarity),
                                        "authorized": cached_identity.get("authorized", False),
                                        "last_seen": frame_index,
                                        "last_face_detected": frame_index,
                                        "embedding": result.embedding,
                                        "reid_matched": True,
                                        "merged_from": best_match_id,
                                        "confidence": 1.0,
                                        "display_id": display_id,
                                        "presence_frames": cached_identity.get("presence_frames", 1),
                                        "face_box": result.face_box or cached_identity.get("face_box"),
                                        "body_features": cached_identity.get("body_features"),
                                    }
                                    if yolo_to_display_id is not None and display_id is not None:
                                        yolo_to_display_id[track_id] = display_id
                                    track_identities.pop(best_match_id, None)
                                    if yolo_to_display_id is not None:
                                        yolo_to_display_id.pop(best_match_id, None)
                                    result_q.task_done()
                                    continue

                        # Persistent cache ReID
                        if self.enable_reid and persistent_cache and authorized:
                            best_cache_key = None
                            best_cache_sim = 0.0
                            for cache_key, cached_identity in persistent_cache.items():
                                if not cached_identity.authorized:
                                    continue
                                sim = compute_similarity(result.embedding, cached_identity.embedding)
                                if sim >= self.reid_similarity_thresh and sim > best_cache_sim:
                                    best_cache_sim = sim
                                    best_cache_key = cache_key
                            if best_cache_key is not None:
                                cached_identity = persistent_cache[best_cache_key]
                                result.name = cached_identity.name
                                result.employee_id = cached_identity.employee_id or result.employee_id
                                cached_identity.appearances += 1
                                cached_identity.last_seen = frame_index

                        # Guard against identity flips when boxes overlap: only switch to a different
                        # employee if the new similarity is materially higher than the current one.
                        same_person = (
                            identity
                            and identity.get("employee_id") is not None
                            and identity.get("employee_id") == result.employee_id
                        )
                        identity_flip_blocked = False
                        if (
                            identity
                            and identity.get("authorized")
                            and identity.get("employee_id")
                            and result.employee_id != identity.get("employee_id")
                            and result.similarity < identity.get("similarity", 0.0) + 0.25
                        ):
                            identity_flip_blocked = True

                        claim_allowed = True
                        if authorized:
                            identity_key = str(result.employee_id or result.name)
                            claim_allowed = self._resolve_identity_claim(
                                track_identities,
                                identity_key,
                                track_id,
                                (result.similarity, 1.0, frame_index),
                            )
                            if not claim_allowed:
                                authorized = False
                                result.employee_id = "Unknown"
                                result.name = "Unknown"

                        should_update = (
                            identity is None
                            or authorized
                            or (
                                identity is not None
                                and result.similarity > identity.get("similarity", 0.0) + 0.12
                                and not identity_flip_blocked
                            )
                        )

                        base_presence = identity.get("presence_frames", 0) if identity else 0
                        face_box = result.face_box if result.face_box else (identity.get("face_box") if identity else None)
                        resolved_name = result.name if authorized else "Unknown"
                        display_id = resolve_display_id(
                            track_id,
                            resolved_name,
                            authorized,
                            identity.get("display_id") if identity else None,
                        )

                    if should_update:
                        track_identities[track_id] = {
                            "name": resolved_name if authorized else "Unknown",
                            "employee_id": result.employee_id if authorized else None,
                            "similarity": result.similarity,
                            "authorized": authorized,
                            "last_seen": frame_index,
                            "last_face_detected": frame_index,
                            "embedding": result.embedding,
                            "body_features": identity.get("body_features") if identity else None,
                            "reid_matched": False,
                            "confidence": 1.0,
                            "display_id": display_id,
                            "presence_frames": base_presence if base_presence else 0,
                            "face_box": face_box,
                        }
                        identity = track_identities[track_id]
                        if authorized and identity.get("employee_id"):
                            self._enforce_unique_identity(track_identities, track_id)

                        if authorized:
                            cache_key = result.employee_id or result.name
                            if cache_key and cache_key != "Unknown":
                                if cache_key not in persistent_cache:
                                    persistent_cache[cache_key] = PersistentIdentity(
                                        name=result.name,
                                        employee_id=result.employee_id if result.employee_id != "Unknown" else None,
                                        embedding=result.embedding,
                                        similarity=result.similarity,
                                        authorized=True,
                                        first_seen=frame_index,
                                        last_seen=frame_index,
                                        body_features=identity.get("body_features"),
                                        display_id=display_id,
                                    )
                                else:
                                    cached = persistent_cache[cache_key]
                                    cached.embedding = result.embedding
                                    cached.similarity = max(cached.similarity, result.similarity)
                                    cached.last_seen = frame_index
                                    if cached.display_id is None and display_id is not None:
                                        cached.display_id = display_id
                        else:
                            if identity is not None:
                                identity["last_seen"] = frame_index
                                identity["last_face_detected"] = frame_index
                                identity["embedding"] = result.embedding
                                identity["face_box"] = result.face_box or identity.get("face_box")
                                identity["confidence"] = 1.0
                    else:
                        if identity is not None:
                            identity["last_seen"] = frame_index
                            last_face_frame = identity.get("last_face_detected")
                            if last_face_frame is not None:
                                frames_without_face = frame_index - last_face_frame
                                if frames_without_face > self.identity_confidence_decay_frames:
                                    decay_rate = 0.995
                                    current_confidence = identity.get("confidence", 1.0)
                                    identity["confidence"] = max(0.3, current_confidence * decay_rate)
                    result_q.task_done()
                except queue.Empty:
                    break
                except Exception as exc:
                    logger.error("Error processing face result: %s", exc)

        # Step 2: Run human tracking
        human_tracks = self._dedupe_tracks(self.human_tracking.track(frame))
        if not human_tracks:
            self._remove_stale_identities(track_identities, frame_index, yolo_to_display_id)
            return frame, metadata_presence, metadata_alerts

        # Current active track IDs
        current_track_ids = {track.track_id for track in human_tracks if track.track_id is not None}

        # Remove identities for tracks no longer detected
        tracks_to_remove: list[int] = []
        for track_id, identity in list(track_identities.items()):
            if track_id not in current_track_ids:
                frames_since_seen = frame_index - identity.get("last_seen", frame_index)
                if frames_since_seen > max(10, self.identity_cache_frames // 2) and not identity.get("reid_matched", False):
                    tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            info = track_identities.get(track_id)
            if info and info.get("bbox") and info.get("display_id") is not None:
                bx, by, bw, bh = info["bbox"]
                track_history.append((bx, by, bw, bh, info["display_id"], frame_index))
            track_identities.pop(track_id, None)
            if yolo_to_display_id is not None:
                yolo_to_display_id.pop(track_id, None)

        # Step 3: Process each tracked person
        for track in human_tracks:
            track_id = track.track_id
            if track_id is None:
                continue

            x, y, w, h = (
                int(track.x),
                int(track.y),
                int(track.width),
                int(track.height),
            )
            x2 = x + w
            y2 = y + h
            # Expand the crop for face detection robustness (align with pipeline_v3 accuracy)
            expand_x = int(w * self.crop_expand_ratio)
            expand_y = int(h * self.crop_expand_ratio)
            x1e = max(0, x - expand_x)
            y1e = max(0, y - expand_y)
            x2e = min(frame.shape[1], x + w + expand_x)
            y2e = min(frame.shape[0], y + h + expand_y)

            identity = track_identities.get(track_id)
            base_name = identity.get("name") if identity else None
            base_authorized = bool(identity.get("authorized")) if identity else False
            base_display = identity.get("display_id") if identity else None
            display_id = resolve_display_id(track_id, base_name, base_authorized, base_display)
            display_id = reuse_display_id_from_iou(track_id, (x, y, w, h), display_id)
            if display_id is None:
                hist_disp = reuse_display_id_from_history((x, y, w, h), frame_index)
                if hist_disp is not None:
                    display_id = hist_disp
                    yolo_to_display_id[track_id] = hist_disp
            if identity is not None:
                identity["display_id"] = display_id
                identity["bbox"] = (x, y, w, h)

            crop = frame_for_crops[y1e:y2e, x1e:x2e]
            body_features = None
            if crop.size > 0 and self.enable_body_reid:
                body_features = extract_body_features(crop)

                if identity is not None and body_features is not None:
                    identity["body_features"] = body_features
                    if identity.get("authorized") and persistent_cache is not None:
                        cache_key = identity.get("employee_id") or identity.get("name")
                        if cache_key and cache_key in persistent_cache:
                            cached = persistent_cache[cache_key]
                            if cached.body_features is None:
                                cached.body_features = body_features
                            else:
                                alpha = 0.3
                                cached.body_features = alpha * body_features + (1 - alpha) * cached.body_features
                            if cached.display_id is None and identity.get("display_id") is not None:
                                cached.display_id = identity.get("display_id")

                if self.enable_body_reid and body_features is not None:
                    if identity is None or identity.get("name") == "Pending..." or not identity.get("authorized", False):
                        best_body_match_key = None
                        best_body_match_sim = 0.0
                        for cached_key, cached_identity in persistent_cache.items():
                            if not cached_identity.authorized or cached_identity.body_features is None:
                                continue
                            frames_gap = frame_index - cached_identity.last_seen
                            if frames_gap > 30 * fps:
                                continue
                            body_sim = compare_body_features(body_features, cached_identity.body_features)
                            if body_sim >= self.body_reid_similarity_thresh and body_sim > best_body_match_sim:
                                best_body_match_sim = body_sim
                                best_body_match_key = cached_key

                        if best_body_match_key is not None:
                            cached = persistent_cache[best_body_match_key]
                            identity_key = str(cached.employee_id or cached.name)
                            candidate_score = (cached.similarity, 0.85, frame_index)
                            if not self._resolve_identity_claim(
                                track_identities, identity_key, track_id, candidate_score
                            ):
                                # Another track has a stronger claim; keep current identity as-is
                                pass
                            else:
                                stable_display_id = resolve_display_id(
                                    track_id,
                                    cached.name,
                                    True,
                                    cached.display_id,
                                )
                                cached.display_id = stable_display_id
                                track_identities[track_id] = {
                                    "name": cached.name,
                                    "employee_id": cached.employee_id,
                                    "similarity": cached.similarity,
                                    "authorized": True,
                                    "last_seen": frame_index,
                                    "last_face_detected": None,
                                    "embedding": cached.embedding,
                                    "body_features": body_features,
                                    "reid_matched": True,
                                    "body_reid": True,
                                    "confidence": 0.85,
                                    "display_id": stable_display_id,
                                    "presence_frames": identity.get("presence_frames", 0) + 1 if identity else 1,
                                    "face_box": identity.get("face_box") if identity else None,
                                }
                                identity = track_identities[track_id]
                                cached.last_seen = frame_index
                                cached.appearances += 1
                                if identity.get("employee_id"):
                                    self._enforce_unique_identity(track_identities, track_id)

            should_detect_face = (
                identity is None
                or identity.get("name") == "Pending..."
                or not identity.get("authorized", False)
                or identity.get("reid_matched", False)
                or frame_index % self.face_detection_interval == 0
            )

            if should_detect_face:
                if crop.size > 0 and job_q is not None and pending_jobs is not None:
                    if track_id not in pending_jobs:
                        try:
                            max_face = self.max_face_num if self.max_face_num > 0 else 1
                            job = FaceJob(track_id, crop.copy(), origin=(x, y), max_num=max_face)
                            job_q.put_nowait(job)
                            pending_jobs.add(track_id)
                        except queue.Full:
                            logger.warning(
                                "Job queue full (%s/%s) - cannot submit track %s",
                                job_q.qsize(),
                                job_q.maxsize,
                                track_id,
                            )
                    else:
                        logger.debug("Track %s already pending face detection", track_id)

                if identity is None:
                    identity = {
                        "name": "Pending...",
                        "employee_id": None,
                        "similarity": 0.0,
                        "authorized": False,
                        "last_seen": frame_index,
                        "last_face_detected": None,
                        "embedding": None,
                        "body_features": body_features,
                        "confidence": 1.0,
                        "display_id": display_id,
                        "presence_frames": 1,
                        "face_box": None,
                    }
                    track_identities[track_id] = identity
            else:
                if identity is not None:
                    identity["last_seen"] = frame_index
                    last_face_frame = identity.get("last_face_detected")
                    if last_face_frame is not None:
                        frames_without_face = frame_index - last_face_frame
                        if frames_without_face > self.identity_confidence_decay_frames:
                            decay_rate = 0.995
                            current_confidence = identity.get("confidence", 1.0)
                            identity["confidence"] = max(0.3, current_confidence * decay_rate)
                else:
                    identity = {
                        "name": "Pending...",
                        "employee_id": None,
                        "similarity": 0.0,
                        "authorized": False,
                        "last_seen": frame_index,
                        "last_face_detected": None,
                        "embedding": None,
                        "body_features": body_features,
                        "confidence": 1.0,
                        "display_id": display_id,
                        "presence_frames": 1,
                        "face_box": None,
                    }
                    track_identities[track_id] = identity

            if identity is not None:
                identity["presence_frames"] = identity.get("presence_frames", 0) + 1

            if identity and identity.get("authorized", False):
                employee_id = identity.get("employee_id")
                if employee_id:
                    seconds_present = identity.get("presence_frames", 0) / fps
                    metadata_presence.append(
                        {
                            "employee_id": employee_id,
                            "employee_name": identity["name"],
                            "presence_seconds": round(seconds_present, 1),
                        }
                    )

            if identity and (not identity.get("authorized", False)) and identity.get("name") != "Pending...":
                if track_id not in unknown_first_seen:
                    unknown_first_seen[track_id] = current_time
                duration = current_time - unknown_first_seen[track_id]
                if duration >= self.unknown_presence_threshold and track_id not in unknown_alerted_tracks:
                    face_box = identity.get("face_box", (x, y, w, h))
                    crop_face = self._crop_face(frame_for_crops, *face_box)
                    snapshot_path = self._store_snapshot(crop_face)

                    alert_event = AlertEvent(
                        stream_id=stream_data.stream_id,
                        event_id=stream_data.event_id,
                        branch_id=branch_id,
                        floor_id=floor_id,
                        camera_no=camera_label,
                        track_id=track_id,
                        duration_seconds=int(duration),
                        snapshot_path=snapshot_path,
                        metadata={"similarity": identity.get("similarity", 0.0)},
                    )

                    stored_alert = self.alert_event_db.add_alert_event(alert_event)
                    payload = stored_alert.to_json()
                    payload["snapshot_base64"] = self._encode_image(crop_face)
                    metadata_alerts.append(payload)
                    unknown_alerted_tracks.add(track_id)

            # Draw overlay
            color = (0, 255, 0) if identity and identity.get("authorized", False) else (0, 0, 255)
            status = "Authorized" if identity and identity.get("authorized", False) else "Unauthorized"
            if identity and identity.get("body_reid", False):
                status = "Authorized (Body)"

            confidence = identity.get("confidence", 1.0) if identity else 1.0
            if confidence < 0.7:
                color = tuple(int(c * confidence + 128 * (1 - confidence)) for c in color)

            draw_bbox(frame, (x, y, x2, y2), color)
            display_label_id = identity.get("display_id", track_id) if identity else track_id
            label_name = identity.get("name", "Unknown") if identity else "Unknown"
            if identity and identity.get("authorized", False):
                label = f"{status} | ID:{display_label_id} {label_name}"
            else:
                label = f"{status} | ID:{display_label_id} {label_name}"
            text_y = y - 10 if y - 10 > 20 else y + 25
            if confidence < 0.95 and identity and identity.get("name") not in (None, "Pending..."):
                label = f"{label} [{confidence:.0%}]"

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Remove stale identities
        self._remove_stale_identities(track_identities, frame_index, yolo_to_display_id)

        return frame, metadata_presence, metadata_alerts

    def _remove_stale_identities(
        self,
        track_identities: dict[int, dict],
        frame_index: int,
        yolo_to_display_id: Optional[Dict[int, int]] = None,
    ):
        """Remove tracks that haven't been seen for too long."""
        stale_ids = [
            track_id
            for track_id, info in track_identities.items()
            if frame_index - info.get("last_seen", frame_index) > self.identity_cache_frames
        ]
        for track_id in stale_ids:
            logger.info(f"Removing stale track {track_id} ({track_identities[track_id]['name']})")
            track_identities.pop(track_id, None)
            if yolo_to_display_id is not None:
                yolo_to_display_id.pop(track_id, None)

    def _demote_identity(self, track_identities: dict[int, dict], track_id: int):
        info = track_identities.get(track_id)
        if not info:
            return
        info["authorized"] = False
        info["name"] = "Unknown"
        info["employee_id"] = None
        info["reid_matched"] = False
        info["body_reid"] = False
        info["confidence"] = min(info.get("confidence", 1.0), 0.8)

    def _resolve_identity_claim(
        self,
        track_identities: dict[int, dict],
        identity_key: str | None,
        track_id: int,
        candidate_score: tuple[float, float, int],
    ) -> bool:
        """
        Ensure only one active track holds a given identity. If another track has
        an equal or stronger claim, block this assignment. Otherwise demote weaker duplicates.
        """
        if not identity_key or identity_key in {"Unknown", "Pending..."}:
            return True

        # Block if a stronger/equal holder exists
        for tid, info in track_identities.items():
            if tid == track_id or not info.get("authorized"):
                continue
            key = info.get("employee_id") or info.get("name")
            if key != identity_key:
                continue
            score = (
                info.get("similarity", 0.0),
                info.get("confidence", 1.0),
                info.get("last_seen", 0),
            )
            if score >= candidate_score:
                return False

        # Demote weaker duplicates
        for tid, info in list(track_identities.items()):
            if tid == track_id or not info.get("authorized"):
                continue
            key = info.get("employee_id") or info.get("name")
            if key != identity_key:
                continue
            self._demote_identity(track_identities, tid)

        return True

    def _dedupe_tracks(self, tracks: list) -> list:
        """
        Deduplicate tracker outputs by track_id, keeping the highest-confidence (or largest)
        box for each track to avoid double boxes in the same frame.
        """
        if not tracks:
            return []

        best: Dict[int, tuple] = {}
        for track in tracks:
            tid = getattr(track, "track_id", None)
            if tid is None:
                continue
            conf = getattr(track, "confidence", None) or 0.0
            area = (getattr(track, "width", 0) or 0) * (getattr(track, "height", 0) or 0)
            score = (conf, area)
            if tid not in best or score > best[tid][0]:
                best[tid] = (score, track)

        return [entry[1] for entry in best.values()]

    def _enforce_unique_identity(self, track_identities: dict[int, dict], current_track_id: int | None):
        """
        Ensure a single authorized track per employee_id. If duplicates exist, keep the one
        with the highest similarity/confidence and demote the others to Unknown.
        """
        if not track_identities:
            return

        keys_to_tracks: Dict[str, list[int]] = {}
        for tid, info in track_identities.items():
            if not info.get("authorized"):
                continue
            identity_key = info.get("employee_id") or info.get("name")
            if not identity_key or identity_key in ("Unknown", "Pending..."):
                continue
            keys_to_tracks.setdefault(str(identity_key), []).append(tid)

        target_keys = keys_to_tracks.keys() if current_track_id is None else [
            key
            for key, ids in keys_to_tracks.items()
            if current_track_id in ids
        ]

        for identity_key in list(target_keys):
            tids = keys_to_tracks.get(identity_key, [])
            if len(tids) <= 1:
                continue

            best_id = None
            best_score = (-1.0, -1.0, -1)
            for tid in tids:
                info = track_identities.get(tid, {})
                score = (
                    info.get("similarity", 0.0),
                    info.get("confidence", 1.0),
                    info.get("last_seen", 0),
                )
                if score > best_score:
                    best_score = score
                    best_id = tid

            for tid in tids:
                if tid == best_id:
                    continue
                info = track_identities.get(tid)
                if not info:
                    continue
                self._demote_identity(track_identities, tid)

    def _identify_employee_from_crop(self, crop, face, known_embeddings):
        """Identify employee from a cropped face region"""
        if not known_embeddings:
            return "Unknown", 0.0

        # Generate face embedding from the crop using detected keypoints
        face_embedding = self.face_embedding_generator.predict(crop, face.key_points)
        if face_embedding is None or len(face_embedding) == 0:
            return "Unknown", 0.0

        # Compare against all known embeddings
        best_similarity = 0.0
        matched_employee_id = "Unknown"
        for target_embedding in known_embeddings:
            decoded_bytes = base64.b64decode(target_embedding.vector_data)
            embedding_array = np.frombuffer(decoded_bytes, dtype=np.float32)
            similarity = self._compute_similarity(embedding_array, face_embedding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                matched_employee_id = str(target_embedding.employee_id)

        return matched_employee_id, best_similarity

    @staticmethod
    def _rescale_frame(frame, percent=100):
        height, width = frame.shape[:2]
        if (width / height) == (16 / 9):
            width = 1280
            height = 720
        else:
            height = height * percent // 100
            width = width * percent // 100
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _identify_employee(self, frame, face, known_embeddings, face_employee_map):
        """Legacy method - kept for compatibility"""
        face_id = face.bounding_boxes.track_id
        if face_id in face_employee_map:
            return face_employee_map[face_id], 1.0

        if not known_embeddings:
            return "Unknown", 0.0

        face_embedding = self.face_embedding_generator.predict(frame, face.key_points)
        if face_embedding is None or len(face_embedding) == 0:
            return "Unknown", 0.0

        best_similarity = 0.0
        matched_employee_id = "Unknown"
        for target_embedding in known_embeddings:
            decoded_bytes = base64.b64decode(target_embedding.vector_data)
            embedding_array = np.frombuffer(decoded_bytes, dtype=np.float32)
            similarity = self._compute_similarity(embedding_array, face_embedding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                matched_employee_id = str(target_embedding.employee_id)

        return matched_employee_id, best_similarity

    @staticmethod
    def _compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    @staticmethod
    def _crop_face(frame, x, y, w, h, margin_ratio: float = 0.15):
        """Return a copy of the region surrounding the detected face.

        We grab the pixels *before* any annotation happens by copying the
        expanded slice from the frame. The slight margin makes the snapshot
        feel more contextual instead of being tightly cropped to the face.
        """

        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        pad_x = int(max(w, 0) * margin_ratio)
        pad_y = int(max(h, 0) * margin_ratio)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, width)
        y2 = min(y + h + pad_y, height)

        if x1 >= x2 or y1 >= y2:
            return None

        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _encode_image(image) -> str | None:
        if image is None or image.size == 0:
            return None
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _assign_tracks_to_faces(face_detections, human_tracks):
        """Legacy method - no longer needed with new approach"""
        if not face_detections:
            return []
        if not human_tracks:
            return face_detections

        for face in face_detections:
            face_box = face.bounding_boxes
            best_track = None
            best_iou = 0.0

            for track in human_tracks:
                iou = UnknownAlertPipeline._calculate_iou(face_box, track)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track and best_iou > 0.1:
                face.bounding_boxes.track_id = best_track.track_id

        return [face for face in face_detections if face.bounding_boxes.track_id is not None]

    @staticmethod
    def _calculate_iou(face_box, track_box):
        xA = max(face_box.x, track_box.x)
        yA = max(face_box.y, track_box.y)
        xB = min(face_box.x + face_box.width, track_box.x + track_box.width)
        yB = min(face_box.y + face_box.height, track_box.y + track_box.height)

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0.0

        face_area = face_box.width * face_box.height
        track_area = track_box.width * track_box.height
        union_area = face_area + track_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _store_snapshot(self, image) -> str | None:
        if image is None or image.size == 0:
            return None
        base_dir = Path(settings.MEDIA_ROOT) / "alerts" / datetime.now().strftime("%Y_%m_%d")
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = f"unknown_{datetime.now().strftime('%H%M%S%f')}.jpg"
        full_path = base_dir / filename
        cv2.imwrite(str(full_path), image)
        try:
            relative_path = full_path.relative_to(settings.MEDIA_ROOT)
        except ValueError:
            relative_path = full_path
        return str(relative_path).replace('\\', '/')
