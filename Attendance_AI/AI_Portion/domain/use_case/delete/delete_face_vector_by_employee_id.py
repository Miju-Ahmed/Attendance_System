#delete_face_vector_by_employee_id_use_case

from attrs import define, field, validators
from domain.repository import FaceVectorRepository
from uuid import UUID  
from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteFaceVectorByEmployeeId:
    face_vector_repository: FaceVectorRepository = field(
        validator=validators.instance_of(FaceVectorRepository)
    )

    def invoke(self, employeeId: str) -> dict:
        logger.info(f"Attempting to delete face vector for employee ID: {employeeId}")
        result = self.face_vector_repository.delete_face_vector_by_employee_id(employeeId)
        logger.info(f"Delete operation result: {result}")
        return result