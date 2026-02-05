from abc import ABC, abstractmethod
from uuid import UUID
from ..model import Region

class RegionRepository(ABC):
    @abstractmethod
    def add_region(self, region: Region) -> dict:
        raise NotImplementedError("Implement add_region method")
    
    @abstractmethod
    def get_region(self, identifier: UUID) -> Region:
        raise NotImplementedError("Implement get_region method")
    
    @abstractmethod
    def delete_region_by_id(self, identifier: UUID) -> dict:
        raise NotImplementedError("Implement delete_region_by_id method")

    @abstractmethod
    def update_region(self, region: Region) -> dict:
        raise NotImplementedError("Implement update_region method")