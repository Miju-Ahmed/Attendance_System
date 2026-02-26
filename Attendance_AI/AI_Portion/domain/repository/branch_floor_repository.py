from abc import ABC, abstractmethod

from ..model import BranchFloorData


class BranchFloorRepository(ABC):
    @abstractmethod
    def add_branch_floor(self, branch_floor: BranchFloorData) -> BranchFloorData:
        raise NotImplementedError("Implement add_branch_floor method")  
    @abstractmethod
    def update_branch_floor(self, branch_floor: BranchFloorData) -> dict:
        raise NotImplementedError("Implement update_branch_floor method")
    @abstractmethod
    def get_branch_floor_by_id(self, branch_id: str, floor_id: str) -> BranchFloorData:
        raise NotImplementedError("Implement get_branch_floor_by_id method")
    @abstractmethod
    def delete_branch_floor(self, branch_id: str, floor_id: str) -> bool:
        raise NotImplementedError("Implement delete_branch_floor method")