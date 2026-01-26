import cv2
import time
import cvzone
import base64
import imageio
import numpy as np
from uuid import UUID, uuid4
from cvzone import putTextRect
from pathlib import Path
from PIL.Image import Image, fromarray
from datetime import timedelta, datetime
from pydantic import BaseModel, InstanceOf
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache
from celery import states
from celery.result import AsyncResult

from domain.model import StreamActivityData , Stream , StreamStatus
from domain.model import BoundingBox, DetectionResult, EmployeeActivityData, TimeData , EmotionStats
from domain.service import HumanEmotionRecognition, HumanFaceDetection, HumanFaceEmbedding, HumanDetection
from domain.repository import (
    FaceVectorRepository,
    RegionRepository,
    EmployeeActivityDataRepository,
    EmployeeRepository,
    StreamRepository,
    RecordedStreamRepository,
)

from .pipeline import Pipeline

from utils import (
    get_logger,
    EuclideanDistTracker,
    FaceTracker,
    resolve_local_media_path,
    open_video_capture,
)

logger = get_logger(__name__)
face_tracker = FaceTracker()

class PipelineV2(BaseModel, Pipeline):
    face_detection: InstanceOf[HumanFaceDetection]
    face_embedding_generator: InstanceOf[HumanFaceEmbedding]
    human_emotion_recognition: InstanceOf[HumanEmotionRecognition]
    employee_db: InstanceOf[EmployeeRepository]
    stream_db: InstanceOf[StreamRepository]
    recorded_stream_db: InstanceOf[RecordedStreamRepository]
    face_vector_db: InstanceOf[FaceVectorRepository]
    region_db: InstanceOf[RegionRepository]
    activity_db: InstanceOf[EmployeeActivityDataRepository]
    is_debugging: bool = False
    count: int = 0

    def process(self, stream_data: StreamActivityData , task_id: str | None = None):
        logger.info(f"Processing pipeline v2 for {stream_data}")
        global stream
        channel_layer = get_channel_layer()
        employee_presence_times = {}
        presence_times = {}
        face_employee_map = {}
        employee_last_seen_time = {}
        employee_first_appearance_time = {}
        employee_activity_records = {}
        employee_emotion_map = {}
        employee_emotion_count = {}


        try:
            all_employees = {e.employee_id: e.first_name + " " + e.last_name for e in self.employee_db.get_all_employees()}
            logger.info(f"All employees: {all_employees}")
        except Exception as e:
            logger.error(f"Error fetching employees: {e}")
            
        stream_info = None
        recorded_stream_info = None
        resolved_branch_id = None
        resolved_floor_id = None

        normalized_stream_id = None
        if stream_data.stream_id:
            normalized_stream_id = (
                stream_data.stream_id
                if isinstance(stream_data.stream_id, UUID)
                else UUID(str(stream_data.stream_id))
            )

        try:
            if normalized_stream_id:
                stream_info = self.stream_db.get_stream(normalized_stream_id)
                if not stream_info:
                    logger.warning(
                        "Stream %s not found; continuing without metadata",
                        normalized_stream_id,
                    )
        except Exception as e:
            logger.error(f"Error fetching stream: {e}")
            stream_info = None

        if not stream_info and normalized_stream_id:
            try:
                recorded_stream_info = self.recorded_stream_db.get_recorded_stream(
                    normalized_stream_id
                )
                if recorded_stream_info and recorded_stream_info.stream_id:
                    linked_stream_id = (
                        recorded_stream_info.stream_id
                        if isinstance(recorded_stream_info.stream_id, UUID)
                        else UUID(str(recorded_stream_info.stream_id))
                    )
                    logger.info(
                        "Recorded stream %s linked to stream %s; loading metadata",
                        normalized_stream_id,
                        linked_stream_id,
                    )
                    stream_info = self.stream_db.get_stream(linked_stream_id)
            except Exception as e:
                logger.error("Error fetching recorded stream metadata: %s", e)

        if stream_info:
            resolved_branch_id = getattr(stream_info, "branch_id", None)
            resolved_floor_id = getattr(stream_info, "floor_id", None)

        if not stream_info and recorded_stream_info:
            logger.info(
                "Proceeding with recorded stream %s without branch/floor metadata",
                normalized_stream_id,
            )

        stream_source = resolve_local_media_path(stream_data.stream_url)
        if stream_source is None:
            logger.error(
                "Recorded stream path %s could not be resolved",
                stream_data.stream_url,
            )
            return

        if stream_source != stream_data.stream_url:
            logger.info(
                "Resolved stream source from %s to %s",
                stream_data.stream_url,
                stream_source,
            )

        if not stream_data.processed_uri:
            logger.error("Processed URI missing for recorded stream %s", stream_data.stream_id)
            return

        processed_output_path = Path(stream_data.processed_uri)
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            stream = open_video_capture(stream_source)

            if not stream:
                logger.error("Failed to load recorded video from %s", stream_source)
                return

            fps = stream.get(cv2.CAP_PROP_FPS)
            writer = imageio.get_writer(str(processed_output_path), fps=fps)
            
            # unique_id = set()
            if task_id is not None:
                task = AsyncResult(task_id)
                # logger.info(f"Task state : {task.state}")
                # logger.info(f"Task ID: {task_id}")

            while True:
                ret, frame = stream.read()
                if not ret:
                    logger.error("Failed to read the video. Retrying...")
                    
                    stream.release()
                    stream_source = resolve_local_media_path(stream_data.stream_url)
                    if stream_source is None:
                        raise FileNotFoundError(
                            "Stream path is unavailable; ensure the recorded video is stored"
                        )
                    stream = open_video_capture(stream_source)

                    if not stream:
                        raise FileNotFoundError(
                            f"Stream not found at path: {stream_source}"
                        )
                    fps = stream.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Reopened the video stream successfully. FPS:{fps}")
                    continue
                if task_id is not None:
                    task = AsyncResult(task_id)
                    # logger.info(f"Task state : {task.state}")
                    if task.state == states.REVOKED:
                        logger.info(
                            f"Task {task_id} has been revoked. Stopping stream processing."
                        )
                        stream_data.success = True
                        stream_data.status = StreamStatus.STOPPED
                        break



                frame = self._rescale_frame(frame)
                self.count += 1
                current_time = self.count / fps

                faces = self.face_detection.predict(frame)
                tracked_faces = face_tracker.update(faces)

                current_employee_ids = set()

                for face in tracked_faces:
                    face_id = face.bounding_boxes.track_id
                    x, y, w, h = face.bounding_boxes.x, face.bounding_boxes.y, face.bounding_boxes.width, face.bounding_boxes.height

                    if face_id not in face_employee_map:
                        target_embeddings = self.face_vector_db.get_face_vectors()
                        face_embedding = self.face_embedding_generator.predict(frame, face.key_points)

                        if face_embedding is None or len(face_embedding) == 0:
                            continue

                        max_similarity = 0
                        matched_employee_id = "Unknown"
                        for target_embedding in target_embeddings:
                            decoded_bytes = base64.b64decode(target_embedding.vector_data)
                            embedding_array = np.frombuffer(decoded_bytes, dtype=np.float32)
                            similarity = self._compute_similarity(embedding_array, face_embedding)

                            if similarity > max_similarity and similarity > 0.3:
                                max_similarity = similarity
                                matched_employee_id = target_embedding.employee_id

                        if matched_employee_id in face_employee_map.values():
                            for key, value in face_employee_map.items():
                                if value == matched_employee_id:
                                    face_id = key
                                    break
                        else:
                            face_employee_map[face_id] = matched_employee_id

                    employee_id = face_employee_map.get(face_id, "Unknown")
                    # if employee_id == "Unknown":
                    #     continue
                    
                    if employee_id == "Unknown":
                        # Draw bounding box for unknown face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for unknown

                        # Display label above the bounding box
                        label = "Unknown"
                        font_scale = 0.8
                        thickness = 2
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                        rect_x = x
                        rect_y = y - text_h - 10
                        rect_y = max(0, rect_y)

                        overlay = frame.copy()
                        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + text_w + 10, rect_y + text_h + 10), (0, 0, 0), -1)
                        alpha = 0.6
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                        cv2.putText(frame, label, (rect_x + 5, rect_y + text_h + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        continue


                    current_employee_ids.add(employee_id)
                    employee_name = all_employees.get(employee_id, "Unknown")

                    presence_times[employee_id] = presence_times.get(employee_id, 0) + 1
                    employee_presence_times[employee_id] = presence_times[employee_id] / fps

                    employee_last_seen_time[employee_id] = current_time

                    if employee_id not in employee_first_appearance_time:
                        employee_first_appearance_time[employee_id] = current_time

                    x, y, w, h = face.bounding_boxes.x, face.bounding_boxes.y, face.bounding_boxes.width, face.bounding_boxes.height
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    first_appearance_time = employee_first_appearance_time[employee_id]
                    total_time = current_time - first_appearance_time
                    absence_time = total_time - employee_presence_times[employee_id]
                    calculated_time = timedelta(seconds=employee_presence_times[employee_id])
                    presence_time_str = f"Presence: {int(calculated_time.total_seconds() // 3600)}h {int((calculated_time.total_seconds() % 3600) // 60)}m {int(calculated_time.total_seconds() % 60)}s"
                    absence_time_str = f"Absence: {int(absence_time // 3600)}h {int((absence_time % 3600) // 60)}m {int(absence_time % 60)}s" if absence_time > 0 else f"Absence: 0h 0m 0s"
                    
                    if self.count % 30 == 0:
                        crop_img = frame[y:y + h, x:x + w]
                        emotion = self.human_emotion_recognition.predict(crop_img)
                        if employee_id not in employee_emotion_count:
                            employee_emotion_count[employee_id] = {"Happy": 0, "Sad": 0, "Neutral": 0}
                        employee_emotion_count[employee_id][emotion] += 1                 
                        employee_emotion_map[employee_id] = emotion
                    
                    branch_id = resolved_branch_id
                    floor_id = resolved_floor_id

                    if employee_id not in employee_activity_records:
                        activity_data = EmployeeActivityData(
                            employee_activity_id=uuid4(),
                            employee_id=employee_id,
                            stream_id=stream_data.stream_id,
                            branch_id=branch_id,
                            floor_id=floor_id,
                            activity_type="present",
                            emotion_stats =EmotionStats(
                                                happy=round((employee_emotion_count[employee_id]["Happy"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2),
                                                sad=round((employee_emotion_count[employee_id]["Sad"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2),
                                                neutral=round((employee_emotion_count[employee_id]["Neutral"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2)
                                            ) if employee_id in employee_emotion_count else EmotionStats(),
                                        
                            track_id=face_id,
                            timedata=TimeData(
                                start_time=datetime.now(),
                                presence_time=int(employee_presence_times[employee_id]),
                                absence_time=0,
                                created_at=datetime.now(),
                                modified_at=datetime.now(),
                                date_of_activity=datetime.now().date()
                            )
                        )
                        try:
                            self.activity_db.add_employee_activity_data(activity_data)
                            employee_activity_records[employee_id] = activity_data.employee_activity_id
                        except Exception as e:
                            logger.error(f"Error adding employee activity data: {e}")
                    else:
                        updated_data = EmployeeActivityData(
                            employee_activity_id=employee_activity_records[employee_id],
                            employee_id=employee_id,
                            stream_id=stream_data.stream_id,
                            branch_id=branch_id,
                            floor_id=floor_id,
                            activity_type="present",
                            emotion_stats =EmotionStats(
                                                happy=round((employee_emotion_count[employee_id]["Happy"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2),
                                                sad=round((employee_emotion_count[employee_id]["Sad"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2),
                                                neutral=round((employee_emotion_count[employee_id]["Neutral"] / (sum(employee_emotion_count[employee_id].values()) or 1)) * 100, 2)
                                            ) if employee_id in employee_emotion_count else EmotionStats(), 
                            track_id=face_id,
                            timedata=TimeData(
                                end_time=datetime.now(),
                                presence_time=int(employee_presence_times[employee_id]),
                                absence_time=int(absence_time),
                                modified_at=datetime.now(),
                            )
                        )
                        if self.count % 30 == 0:
                            try:
                                self.activity_db.update_employee_activity_data(updated_data)
                            except Exception as e:
                                logger.error(f"Error updating employee activity data: {e}")

                    # if self.count % 30 == 0:
                    #     crop_img = frame[y:y + h, x:x + w]
                    #     emotion = self.human_emotion_recognition.predict(crop_img)
                    #     if employee_id not in employee_emotion_count:
                    #         employee_emotion_count[employee_id] = {"Happy": 0, "Sad": 0, "Neutral": 0}
                    #     employee_emotion_count[employee_id][emotion] += 1                 
                    #     employee_emotion_map[employee_id] = emotion

                    emotion = employee_emotion_map.get(employee_id, "Neutral")

                    text_lines = [
                        f"Track ID: {face_id}",
                        f"Employee: {employee_name}",
                        f"Emotion: {emotion}",
                        f"Present: {presence_time_str}",
                        f"Absent: {absence_time_str}"
                    ]

                    # Render text with colored boundary around Emotion line
                    font_scale = 0.6
                    thickness = 2
                    line_height = 25
                    text_padding = 10
                    rect_x = x
                    rect_y = max(0, y - (line_height * len(text_lines)) - text_padding * 2)
                    max_text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for line in text_lines)
                    rect_w = max_text_width + text_padding * 2
                    rect_h = line_height * len(text_lines) + text_padding * 2

                    overlay = frame.copy()
                    cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)
                    alpha = 0.6
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    for i, line in enumerate(text_lines):
                        text_y = rect_y + text_padding + (i + 1) * line_height - 5
                        if i == 2:  # Emotion line
                            (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                            cv2.rectangle(
                                frame,
                                (rect_x + text_padding - 5, text_y - h - 5),
                                (rect_x + text_padding + w + 5, text_y + baseline + 5),
                                self.get_emotion_color(emotion),
                                2
                            )
                        cv2.putText(
                            frame,
                            line,
                            (rect_x + text_padding, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )

                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
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

    @staticmethod
    def _compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2) / (norm1 * norm2)

    @staticmethod
    def _is_stream_active(stream_id):
        return cache.get(f"active_stream_{stream_id}", 0) > 0

    @staticmethod
    def get_emotion_color(emotion):
        if emotion == "Happy":
            return (0, 255, 255)  # Yellow
        elif emotion == "Sad":
            return (0, 0, 255)    # Red
        else:  # Neutral
            return (0, 255, 0)    # Green