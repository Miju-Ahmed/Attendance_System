import cv2
import time
import cvzone
import base64
import imageio
import numpy as np
from cvzone import putTextRect
from pathlib import Path
from PIL.Image import Image, fromarray
from datetime import timedelta, datetime
from pydantic import BaseModel, InstanceOf
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache


from domain.model import StreamActivityData
from domain.model import BoundingBox, DetectionResult
from domain.service import HumanEmotionRecognition, HumanFaceDetection, HumanFaceEmbedding, HumanDetection
from domain.repository import FaceVectorRepository, RegionRepository, EmployeeActivityDataRepository, EmployeeRepository

from .pipeline import Pipeline

from utils import (
    get_logger,
    EuclideanDistTracker,
    FaceTracker,
    resolve_local_media_path,
    open_video_capture,
)

logger = get_logger(__name__)
tracker = EuclideanDistTracker()
face_tracker = FaceTracker()

class PipelineV1(BaseModel, Pipeline):
    face_detection: InstanceOf[HumanFaceDetection]
    face_embedding_generator: InstanceOf[HumanFaceEmbedding]
    human_detection: InstanceOf[HumanDetection]
    # emotion_detector: InstanceOf[HumanEmotionRecognition]
    employee_db: InstanceOf[EmployeeRepository]
    face_vector_db: InstanceOf[FaceVectorRepository]
    region_db: InstanceOf[RegionRepository]
    activity_db: InstanceOf[EmployeeActivityDataRepository]
    is_debugging: bool = False
    count: int = 0

    def process(self, stream_data: StreamActivityData):
        logger.info(f"Processing pipeline v1 for {stream_data}")
        global stream
        channel_layer = get_channel_layer()
        presence_times = {}
        absence_times = {}
        track_employee_map = {}
        face_employee_map = {}
        previous_track_ids = []
        
        try:
            all_employees = {e.employee_id: e.first_name +" "+e.last_name for e in self.employee_db.get_all_employees()}
            logger.info(f"All employees: {all_employees}")
        except Exception as e:
            logger.error(f"Error fetching employees: {e}")
        
        resolved_stream_url = resolve_local_media_path(stream_data.stream_url)
        if resolved_stream_url is None:
            logger.error("Recorded stream path %s could not be resolved", stream_data.stream_url)
            return

        if resolved_stream_url != stream_data.stream_url:
            logger.info(
                "Resolved stream source from %s to %s",
                stream_data.stream_url,
                resolved_stream_url,
            )

        if not stream_data.processed_uri:
            logger.error("Processed URI missing for stream %s", stream_data.stream_id)
            return

        processed_output_path = Path(stream_data.processed_uri)
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            stream = open_video_capture(resolved_stream_url)

            if not stream:
                logger.error("Failed to load recorded video from %s", resolved_stream_url)
                return
            fps = stream.get(cv2.CAP_PROP_FPS)
            # logger.info(f"Stream FPS: {fps}")

            writer = imageio.get_writer(str(processed_output_path), fps=fps)

            while True:
                ret, frame = stream.read()
                if not ret:
                    logger.error("Failed to read the video.")
                    break

                frame = self._rescale_frame(frame)
                self.count += 1

                logger.info(f"Processing frame {self.count}")

                humans = self.human_detection.predict(frame)
                human_tracked_bbox = tracker.update(humans)
                track_ids = [human.track_id for human in human_tracked_bbox]

                if track_ids not in previous_track_ids:
                    faces = self.face_detection.predict(frame)
                    face_tracked_bbox = face_tracker.update(faces)

                for human in human_tracked_bbox:
                    track_id = human.track_id

                    # draw human bounding boxes
                    cv2.rectangle(frame, (human.x, human.y), (human.x + human.width, human.y + human.height), (255, 0, 0), 2)
                    putTextRect(
                        frame,
                        f"Track id: {human.track_id}",
                        (int(human.x), int(human.y - 50)),
                        scale=1,
                        thickness=2,
                        offset=10,
                        colorR=(255, 0, 0),
                        colorT=(255, 255, 255),
                    )

                    if track_id not in track_employee_map:

                        for face in face_tracked_bbox:
                            face_id = face.bounding_boxes.track_id

                            if face_id not in face_employee_map:
                                target_embeddings = self.face_vector_db.get_face_vectors()
                                
                                face_embedding = self.face_embedding_generator.predict(frame, face.key_points)
                                # logger.info(f"Face embedding: {face_embedding}")
                                if not face_embedding:
                                    logger.error("Face embedding is None")
                                    continue
                                face_embedding_bytes = base64.b64decode(face_embedding)
                                face_embedding_array = np.frombuffer(face_embedding_bytes, dtype=np.float32)
                                # logger.info(f"Face embedding array: {face_embedding_array}")

                                max_similarity = 0
                                matched_employee_id = "Unknown"
                                
                                for target_embedding in target_embeddings:
                                    decoded_bytes = base64.b64decode(target_embedding.vector_data)
                                    embedding_array = np.frombuffer(decoded_bytes, dtype=np.float32)
                                    similarity = self._compute_similarity(embedding_array, face_embedding_array)
                                    logger.info(f"Similarity: {similarity}")
                                    if similarity > max_similarity and similarity > 0.5:
                                        max_similarity = similarity
                                        matched_employee_id = target_embedding.employee_id
                                face_employee_map[face_id] = matched_employee_id

                            if self. check_overlap(human, face):
                                track_employee_map[track_id] = face_employee_map[face_id]
                    employee_id = track_employee_map.get(track_id)
                    if not employee_id:
                        continue
                    employee_name = all_employees.get(employee_id, "Unknown")
                    cv2.rectangle(frame, (human.x, human.y), (human.x + human.width, human.y + human.height), (0, 255, 0), 2)
                    putTextRect(
                        frame,
                        f"Employee: {employee_name}",
                        (int(human.x), int(human.y - 20)),
                        scale=1,
                        thickness=2,
                        offset=10,
                        colorR=(0, 255, 0),
                        colorT=(255, 255, 255),
                    )
                    logger.info(f"Employee ID: {employee_id}, Track ID: {track_id}")
                    presence_times[employee_id] = presence_times.get(employee_id, 0) + 1
                    logger.info(f"Presence time for {employee_id}: {presence_times[employee_id]}")
                    # if self.region_db.check_desk_overlap(human):
                    # else:
                    #     absence_times[employee_id] = absence_times.get(employee_id, 0) + 1

                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                broadcast_targets = {str(stream_data.stream_id)}
                if stream_data.event_id:
                    broadcast_targets.add(str(stream_data.event_id))

                for target in broadcast_targets:
                    async_to_sync(channel_layer.group_send)(
                        f"video_stream_{target}",
                        {
                            'type': 'send_frame',
                            'frame': frame_data,
                        }
                    )

        except FileNotFoundError as fnf_error:
            logger.error(fnf_error)
            stream_data.success = False
            stream_data.errors = [str(fnf_error)]
            return stream_data

        except Exception as e:
            logger.exception("An unexpected error occurred while loading the Stream.")
            stream_data.success = False
            stream_data.errors = ["An unexpected error occurred while loading the Stream."]
            return stream_data

        finally:
            if 'stream' in locals() and stream.isOpened():
                stream.release()
                logger.info("Released the video resource.")


    @staticmethod
    def _rescale_frame(frame, percent=100):
        height = frame.shape[0]
        width = frame.shape[1]
        if (width / height) == (16 / 9):
            width = 1280
            height = 720
        else:
            height = height * percent // 100
            width = width * percent // 100
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def check_overlap(self, human: BoundingBox, face: DetectionResult):
        human_bbox = human.x, human.y, human.width, human.height
        if not human_bbox:
            logger.warning("Human bounding box is empty")
            return False
        face_bbox = face.bounding_boxes.x, face.bounding_boxes.y, face.bounding_boxes.width, face.bounding_boxes.height
        if not face:
            logger.warning("Face bounding box is empty")
            return False
        return self.do_bboxes_overlap(human_bbox, face_bbox)

    def do_bboxes_overlap(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        return inter_x1 < inter_x2 and inter_y1 < inter_y2
    
    @staticmethod
    def _compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return cosine_similarity
    
    @staticmethod
    def _is_stream_active(stream_id):
        return cache.get(f"active_stream_{stream_id}", 0) > 0