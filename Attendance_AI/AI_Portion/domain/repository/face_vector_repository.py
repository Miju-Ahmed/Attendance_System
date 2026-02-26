from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional
from ..model import FaceVector

class FaceVectorRepository(ABC):
    @abstractmethod
    def add_face_vector(self, face_vector: FaceVector) -> FaceVector:
        raise NotImplementedError("Implement add_face_vector method")
    
    @abstractmethod
    def update_face_vector(self, face_vector: FaceVector) -> FaceVector:
        raise NotImplementedError("Implement update_face_vector method")

    @abstractmethod
    def get_face_vector_by_id(self, face_vector_id: str) -> Optional[FaceVector]:
        raise NotImplementedError("Implement get_face_vector_by_id method")

    @abstractmethod
    def delete_face_vector(self, face_vector_id: str) -> bool:
        raise NotImplementedError("Implement delete_face_vector method")
    
    @abstractmethod
    def get_face_vectors(self) -> list[FaceVector]:
        raise NotImplementedError("Implement get_face_vectors method")
    
    @abstractmethod
    def delete_face_vector_by_employee_id(self, employee_id: str) -> dict:
        raise NotImplementedError("Implement delete_face_vector_by_employee_id method")
