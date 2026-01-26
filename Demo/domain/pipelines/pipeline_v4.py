from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import imageio
import numpy as np
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from pydantic import BaseModel, InstanceOf

from domain.model import StreamActivityData
from domain.repository import EmployeeRepository, FaceVectorRepository
from domain.service import (
    HumanDetection,
    HumanFaceDetection,
    HumanFaceEmbedding,
    IdCardDetection,
)
from utils import (
    EuclideanDistTracker,
    FaceTracker,
    get_logger,
    open_video_capture,
    resolve_local_media_path,
)

from .pipeline import Pipeline

logger = get_logger(__name__)


class PipelineV4(BaseModel, Pipeline):
    """Authorize subjects via face recognition or ID-card+logo verification."""

    face_detection: InstanceOf[HumanFaceDetection]
    face_embedding_generator: InstanceOf[HumanFaceEmbedding]
    human_detection: InstanceOf[HumanDetection]
    id_card_detection: InstanceOf[IdCardDetection]
    face_vector_db: InstanceOf[FaceVectorRepository]
    employee_db: InstanceOf[EmployeeRepository]
    is_debugging: bool = False
    similarity_threshold: float = 0.35
    authorization_hold_frames: int = 150

    def process(self, stream_data: StreamActivityData, task_id: str | None = None):  # noqa: D401
        logger.info("Processing pipeline v4 for stream %s", stream_data.stream_id)
        stream_source = resolve_local_media_path(stream_data.stream_url)
        if stream_source is None:
            logger.error("Unable to resolve stream path for %s", stream_data.stream_url)
            return

        if not stream_data.processed_uri:
            logger.error("Processed URI missing for stream %s", stream_data.stream_id)
            return

        processed_output_path = Path(stream_data.processed_uri)
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)

        employees = self._load_employee_map()
        known_embeddings = self._load_known_embeddings()

        human_tracker = EuclideanDistTracker()
        tracked_faces = FaceTracker()
        authorization_cache: Dict[int, Dict[str, object]] = {}
        track_employee_map: Dict[int, str] = {}
        face_employee_map: Dict[int, str] = {}

        channel_layer = get_channel_layer()

        try:
            stream = open_video_capture(stream_source)
            if not stream:
                raise FileNotFoundError(f"Stream not found: {stream_source}")

            fps = max(stream.get(cv2.CAP_PROP_FPS), 1)
            writer = imageio.get_writer(str(processed_output_path), fps=fps)

            while True:
                ret, frame = stream.read()
                if not ret:
                    logger.info("End of stream reached for %s", stream_source)
                    break

                frame = self._rescale_frame(frame)
                human_boxes = self.human_detection.predict(frame)
                tracked_humans = human_tracker.update(human_boxes)

                faces = self.face_detection.predict(frame)
                face_tracks = tracked_faces.update(faces)

                self._associate_faces(
                    frame,
                    tracked_humans,
                    face_tracks,
                    face_employee_map,
                    track_employee_map,
                    known_embeddings,
                )

                credential_boxes = self.id_card_detection.predict(frame)
                id_boxes, logo_boxes = self._split_credentials(credential_boxes)

                frame_metadata = []
                seen_tracks = set()

                for human in tracked_humans:
                    track_id = human.track_id
                    if track_id is None:
                        continue
                    seen_tracks.add(track_id)

                    status = self._resolve_status(
                        human,
                        track_employee_map,
                        authorization_cache,
                        employees,
                        id_boxes,
                        logo_boxes,
                    )

                    self._draw_human(frame, human, status)
                    frame_metadata.append(status)

                self._decay_authorizations(authorization_cache, seen_tracks)

                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                encoded_frame = self._encode_frame(frame)

                metadata_payload = {
                    "authorization": frame_metadata,
                    "authorized": sum(1 for item in frame_metadata if item["status"] == "authorized"),
                    "unauthorized": sum(1 for item in frame_metadata if item["status"] == "unauthorized"),
                }

                broadcast_targets = {str(stream_data.stream_id)}
                if stream_data.event_id:
                    broadcast_targets.add(str(stream_data.event_id))

                for target in broadcast_targets:
                    message = {
                        "type": "send_frame",
                        "frame": encoded_frame,
                        "metadata": metadata_payload,
                    }
                    async_to_sync(channel_layer.group_send)(f"video_stream_{target}", message)

        except FileNotFoundError as exc:
            logger.error("Stream error: %s", exc)
            stream_data.success = False
            stream_data.errors = [str(exc)]
            return stream_data
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected error in pipeline v4")
            stream_data.success = False
            stream_data.errors = ["Unexpected error while processing stream"]
            return stream_data
        finally:
            if "stream" in locals() and stream and stream.isOpened():
                stream.release()
                logger.info("Released video resource")
            if "writer" in locals():
                writer.close()

    def _load_employee_map(self) -> Dict[str, str]:
        try:
            return {
                str(employee.employee_id): f"{employee.first_name} {employee.last_name}".strip()
                for employee in self.employee_db.get_all_employees()
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unable to load employees: %s", exc)
            return {}

    def _load_known_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        embeddings = []
        try:
            for vector in self.face_vector_db.get_face_vectors():
                decoded = base64.b64decode(vector.vector_data)
                embeddings.append((str(vector.employee_id), np.frombuffer(decoded, dtype=np.float32)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Unable to load face vectors: %s", exc)
        return embeddings

    def _associate_faces(
        self,
        frame: np.ndarray,
        tracked_humans,
        face_tracks,
        face_employee_map: Dict[int, str],
        track_employee_map: Dict[int, str],
        known_embeddings: List[Tuple[str, np.ndarray]],
    ) -> None:
        for face in face_tracks:
            face_box = face.bounding_boxes
            if face_box.track_id is None:
                continue

            if face_box.track_id not in face_employee_map:
                match_id = self._identify_employee(frame, face, known_embeddings)
                face_employee_map[face_box.track_id] = match_id

            for human in tracked_humans:
                if human.track_id is None or human.track_id in track_employee_map:
                    continue
                if self._boxes_overlap(human, face_box):
                    track_employee_map[human.track_id] = face_employee_map[face_box.track_id]

    def _identify_employee(self, frame, face, known_embeddings) -> str:
        if not known_embeddings:
            return "Unknown"
        face_embedding = self.face_embedding_generator.predict(frame, face.key_points)
        if face_embedding is None or len(face_embedding) == 0:
            return "Unknown"

        best_similarity = 0.0
        matched_employee = "Unknown"
        for employee_id, embedding in known_embeddings:
            similarity = self._compute_similarity(embedding, face_embedding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                matched_employee = employee_id
        return matched_employee

    def _split_credentials(self, boxes: Iterable) -> Tuple[List, List]:
        id_boxes = []
        logo_boxes = []
        for box in boxes:
            if box.class_id == self.id_card_detection.id_class_index:
                id_boxes.append(box)
            elif box.class_id == self.id_card_detection.logo_class_index:
                logo_boxes.append(box)
        return id_boxes, logo_boxes

    def _resolve_status(
        self,
        human,
        track_employee_map: Dict[int, str],
        authorization_cache: Dict[int, Dict[str, object]],
        employees: Dict[str, str],
        id_boxes: List,
        logo_boxes: List,
    ) -> Dict[str, object]:
        track_id = human.track_id
        if track_id is None:
            return {"trackId": None, "status": "unauthorized"}

        existing = authorization_cache.get(track_id)
        if existing:
            existing["frames_lost"] = 0
            return existing

        employee_id = track_employee_map.get(track_id)
        if employee_id and employee_id != "Unknown":
            status = {
                "trackId": track_id,
                "status": "authorized",
                "source": "face",
                "employeeId": employee_id,
                "label": employees.get(employee_id, "Authorized"),
                "frames_lost": 0,
            }
            authorization_cache[track_id] = status
            return status

        if self._has_id_with_logo(human, id_boxes, logo_boxes):
            status = {
                "trackId": track_id,
                "status": "authorized",
                "source": "id-card",
                "employeeId": None,
                "label": "Authorized (ID)",
                "frames_lost": 0,
            }
            authorization_cache[track_id] = status
            return status

        return {
            "trackId": track_id,
            "status": "unauthorized",
            "source": "none",
            "employeeId": None,
            "label": "Unauthorized",
            "frames_lost": 0,
        }

    def _decay_authorizations(self, cache: Dict[int, Dict[str, object]], seen_tracks: set) -> None:
        for track_id in list(cache.keys()):
            if track_id in seen_tracks:
                continue
            cache[track_id]["frames_lost"] = cache[track_id].get("frames_lost", 0) + 1
            if cache[track_id]["frames_lost"] > self.authorization_hold_frames:
                cache.pop(track_id, None)

    def _draw_human(self, frame, human, status: Dict[str, object]) -> None:
        x, y, w, h = human.x, human.y, human.width, human.height
        color = (0, 255, 0) if status["status"] == "authorized" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = status.get("label", status["status"].upper())
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    def _has_id_with_logo(self, human, id_boxes, logo_boxes) -> bool:
        human_rect = (human.x, human.y, human.x + human.width, human.y + human.height)
        for card in id_boxes:
            if not self._center_inside(card, human_rect):
                continue
            for logo in logo_boxes:
                if self._center_inside(logo, (card.x, card.y, card.x + card.width, card.y + card.height)):
                    return True
        return False

    @staticmethod
    def _center_inside(box, bounds) -> bool:
        cx = box.x + box.width / 2
        cy = box.y + box.height / 2
        x1, y1, x2, y2 = bounds
        return x1 <= cx <= x2 and y1 <= cy <= y2

    @staticmethod
    def _boxes_overlap(human, face_box) -> bool:
        hx1, hy1 = human.x, human.y
        hx2, hy2 = human.x + human.width, human.y + human.height
        fx1, fy1 = face_box.x, face_box.y
        fx2, fy2 = face_box.x + face_box.width, face_box.y + face_box.height
        return hx1 < fx2 and hx2 > fx1 and hy1 < fy2 and hy2 > fy1

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
    def _rescale_frame(frame, percent=100):
        height, width = frame.shape[:2]
        if width / max(height, 1) == 16 / 9:
            return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        height = int(height * percent / 100)
        width = int(width * percent / 100)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode_frame(frame) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
