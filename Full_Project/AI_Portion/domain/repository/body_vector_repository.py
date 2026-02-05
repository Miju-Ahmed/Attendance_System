from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional
from ..model import BodyVector

class BodyVectorRepository(ABC):
    @abstractmethod
    def add_body_vector(self, body_vector: BodyVector) -> BodyVector:
        raise NotImplementedError("Implement add_body_vector method")

    @abstractmethod
    def update_body_vector(self, body_vector: BodyVector) -> BodyVector:
        raise NotImplementedError("Implement update_body_vector method")

    @abstractmethod
    def get_body_vector_by_id(self, body_vector_id: str) -> Optional[BodyVector]:
        raise NotImplementedError("Implement get_body_vector_by_id method")

    @abstractmethod
    def delete_body_vector(self, body_vector_id: str) -> bool:
        raise NotImplementedError("Implement delete_body_vector method")
