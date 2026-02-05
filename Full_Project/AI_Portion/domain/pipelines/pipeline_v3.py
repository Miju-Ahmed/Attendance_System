import base64
from datetime import datetime
from pathlib import Path
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
from domain.service import HumanFaceDetection, HumanFaceEmbedding
from utils import FaceTracker, get_logger, open_video_capture, resolve_local_media_path

from .pipeline import Pipeline

logger = get_logger(__name__)
face_tracker = FaceTracker()


class PipelineV3(BaseModel, Pipeline):
    face_detection: InstanceOf[HumanFaceDetection]
    face_embedding_generator: InstanceOf[HumanFaceEmbedding]
    face_vector_db: InstanceOf[FaceVectorRepository]
    employee_db: InstanceOf[EmployeeRepository]
    stream_db: InstanceOf[StreamRepository]
    recorded_stream_db: InstanceOf[RecordedStreamRepository]
    alert_event_db: InstanceOf[AlertEventRepository]
    is_debugging: bool = False
    count: int = 0
    unknown_presence_threshold: float = 10.0

    def process(self, stream_data: StreamActivityData, task_id: str | None = None):
        logger.info("Processing unknown alert pipeline for %s", stream_data.stream_id)
        global stream
        channel_layer = get_channel_layer()
        face_employee_map: dict[int, str] = {}
        employee_height_reference: dict[str, float] = {}
        presence_frames: dict[str, int] = {}
        unknown_first_seen: dict[int, float] = {}
        unknown_alerted_tracks: set[int] = set()

        try:
            all_employees = {
                str(emp.employee_id): f"{emp.first_name} {emp.last_name}".strip()
                for emp in self.employee_db.get_all_employees()
            }
        except Exception as exc:
            logger.error("Unable to fetch employees: %s", exc)
            all_employees = {}

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

        known_embeddings = self.face_vector_db.get_face_vectors()

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

                if task_id is not None:
                    task = AsyncResult(task_id)
                    if task.state == states.REVOKED:
                        logger.info("Task %s revoked. Stopping stream processing.", task_id)
                        stream_data.success = True
                        stream_data.status = StreamStatus.STOPPED
                        break

                frame = self._rescale_frame(frame)
                frame_for_crops = frame.copy()
                self.count += 1
                current_time = self.count / fps

                faces = self.face_detection.predict(frame)
                tracked_faces = face_tracker.update(faces)

                metadata_presence = []
                metadata_alerts = []
                known_faces: set[str] = set()
                unknown_face_count = 0

                recognized_candidates = []

                for face in tracked_faces:
                    face_id = face.bounding_boxes.track_id
                    x, y, w, h = (
                        int(face.bounding_boxes.x),
                        int(face.bounding_boxes.y),
                        int(face.bounding_boxes.width),
                        int(face.bounding_boxes.height),
                    )

                    employee_id, similarity = self._identify_employee(
                        frame,
                        face,
                        known_embeddings,
                        face_employee_map,
                    )

                    if employee_id != "Unknown":
                        recognized_candidates.append(
                            {
                                "employee_id": employee_id,
                                "face_id": face_id,
                                "bbox": (x, y, w, h),
                                "similarity": similarity,
                            }
                        )
                        continue

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "Unknown",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                    unknown_face_count += 1
                    if face_id not in unknown_first_seen:
                        unknown_first_seen[face_id] = current_time
                        continue

                    duration = current_time - unknown_first_seen[face_id]
                    if duration < self.unknown_presence_threshold or face_id in unknown_alerted_tracks:
                        continue

                    crop = self._crop_face(frame_for_crops, x, y, w, h)
                    snapshot_path = self._store_snapshot(crop)

                    alert_event = AlertEvent(
                        stream_id=stream_data.stream_id,
                        event_id=stream_data.event_id,
                        branch_id=branch_id,
                        floor_id=floor_id,
                        camera_no=camera_label,
                        track_id=face_id,
                        duration_seconds=int(duration),
                        snapshot_path=snapshot_path,
                        metadata={"similarity": similarity},
                    )

                    stored_alert = self.alert_event_db.add_alert_event(alert_event)
                    payload = stored_alert.to_json()
                    payload["snapshot_base64"] = self._encode_image(crop)
                    metadata_alerts.append(payload)
                    unknown_alerted_tracks.add(face_id)
                    logger.info(
                        "Created alert %s for unknown track %s after %.2f seconds",
                        stored_alert.alert_event_id,
                        face_id,
                        duration,
                    )

                if recognized_candidates:
                    best_matches: dict[str, dict] = {}
                    for candidate in recognized_candidates:
                        employee_id = candidate["employee_id"]
                        _, _, _, height = candidate["bbox"]
                        reference_height = employee_height_reference.get(employee_id)

                        if reference_height is None:
                            score = candidate["similarity"]
                        else:
                            score = -abs(height - reference_height)

                        existing = best_matches.get(employee_id)
                        if existing is None or score > existing.get("score", float("-inf")):
                            best_matches[employee_id] = {**candidate, "score": score}

                    for employee_id, candidate in best_matches.items():
                        face_id = candidate["face_id"]
                        x, y, w, h = candidate["bbox"]

                        new_reference = employee_height_reference.get(employee_id, float(h))
                        new_reference = (new_reference * 0.7) + (h * 0.3)
                        employee_height_reference[employee_id] = new_reference

                        face_employee_map[face_id] = employee_id
                        presence_frames[employee_id] = presence_frames.get(employee_id, 0) + 1
                        seconds_present = presence_frames[employee_id] / fps
                        employee_name = all_employees.get(employee_id, "Unknown")
                        known_faces.add(employee_name)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{employee_name} ({seconds_present:.1f}s)",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        metadata_presence.append(
                            {
                                "employee_id": employee_id,
                                "employee_name": employee_name,
                                "presence_seconds": round(seconds_present, 1),
                            }
                        )

                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

                metadata_payload = {}
                if metadata_presence:
                    metadata_payload["presence"] = metadata_presence
                if metadata_alerts:
                    metadata_payload["unknown_alerts"] = metadata_alerts

                broadcast_targets = {str(stream_data.stream_id)}
                if stream_data.event_id:
                    broadcast_targets.add(str(stream_data.event_id))

                for target in broadcast_targets:
                    message = {
                        'type': 'send_frame',
                        'frame': frame_data,
                        'unknown': unknown_face_count,
                    }
                    if metadata_payload:
                        message['metadata'] = metadata_payload
                    if known_faces:
                        message['known'] = sorted(known_faces)

                    async_to_sync(channel_layer.group_send)(
                        f"video_stream_{target}",
                        message,
                    )

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
            if similarity > best_similarity and similarity > 0.35:
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