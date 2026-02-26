from uuid import UUID, uuid4
import json
from attrs import define, field, validators
from django.http import JsonResponse

import base64
import numpy as np
from datetime import datetime
from domain.model import Employee, TimeData, UserData, User, FaceVector, DetectionResult
from domain.repository import EmployeeRepository, FaceVectorRepository, AlertEventRepository
from domain.service import HumanFaceDetection, HumanFaceEmbedding

from utils import get_logger
logger = get_logger(__name__)
from django.utils.timezone import now

@define
class RegisterEmployee:
    human_face_detection: HumanFaceDetection = field(
        validator=validators.instance_of(HumanFaceDetection)
    )
    human_face_embedding: HumanFaceEmbedding = field(
        validator=validators.instance_of(HumanFaceEmbedding)
    )
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )
    face_vector_repository: FaceVectorRepository = field(
        validator=validators.instance_of(FaceVectorRepository)
    )
    alert_event_repository: AlertEventRepository | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(AlertEventRepository))
    )

    def invoke(self, request, attribute_name) -> dict:
        try:
            employee_info = json.loads(request.body)
        except Exception as e:
            logger.error(f"Error in parsing request body: {e}")
            return {"error": "Invalid request body"}

        first_name = employee_info.get("first_name")
        last_name = employee_info.get("last_name")
        email = employee_info.get("email")
        phone_number = employee_info.get("phone_number")
        branch = employee_info.get("branch")
        department = employee_info.get("department")
        role = employee_info.get("role")
        user_created = employee_info.get("user_created")
        branch_id = employee_info.get("branch_id")

        if not first_name:
            return {"error": "Missing required fields"}

        employee = Employee(
            employee_id=uuid4(),
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
            branch=branch,
            branch_id=branch_id,
            department=department,
            role=role,
            timedata=TimeData(
                created_at=datetime.now(),
                modified_at=None
            ),
            userdata=UserData(
                user_created=User(user_id=user_created),
                user_modified=None
            )
        )
        result = self.employee_repository.add_employee(employee=employee)

        logger.info(f"Employee added: {employee}")

        if result.get("success") is False:
            return result

        employee_images = employee_info.get(attribute_name, [])
        logger.info(f"Employee images: {employee_images}")

        if not employee_images:
            return {
                "success": False,
                "message": "No images provided"
            }

        for img_base64 in employee_images:
            image = base64_to_cv2(img_base64)
            if image is None:
                logger.error("Skipping invalid image")
                continue
            face_detections = self.human_face_detection.predict(image, max_faces=1)
            logger.info(f"Face detections: {face_detections}")
            if not face_detections:
                logger.error("No face detected in image")
                continue
            for detection in face_detections:
                keypoints = detection.key_points
                embedding = self.human_face_embedding.predict(image, keypoints)
                embedding_bytes = embedding.tobytes()
                embedding_str = base64.b64encode(embedding_bytes).decode("utf-8")
                if not embedding_str:
                    logger.error("Error embedding face")
                    continue
                face_vector = FaceVector(
                    face_vector_id=uuid4(),
                    employee_id=employee.employee_id,
                    vector_data=embedding_str,
                    timedata=TimeData(
                        created_at=datetime.now(),
                        modified_at=None
                    )
                )
                result = self.face_vector_repository.add_face_vector(face_vector)
                logger.info(f"Face vector added: {result}")

        alert_event_id = employee_info.get("alert_event_id")
        if alert_event_id and self.alert_event_repository:
            try:
                self.alert_event_repository.resolve_alert_event(
                    UUID(alert_event_id),
                    employee.employee_id,
                )
                logger.info("Resolved alert event %s after registration", alert_event_id)
            except Exception as exc:
                logger.error("Failed to resolve alert event %s: %s", alert_event_id, exc)

        return {"success": True, "message": "Employee registered successfully"}

def base64_to_cv2(base64_string: str) -> np.ndarray:
    import cv2
    try:
        # Extract the base64 content from the data URL
        header, encoded = base64_string.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None
