from abc import ABC, abstractmethod

from ..model import BranchData


class BranchRepository(ABC):
    @abstractmethod
    def add_branch(self, branch: BranchData) -> dict:
        raise NotImplementedError("Implement add_branch method")

    @abstractmethod
    def update_branch(self, branch: BranchData) -> BranchData:
        raise NotImplementedError("Implement update_branch method")

    @abstractmethod
    def get_branch_by_id(self, branch_id: str) -> BranchData:
        raise NotImplementedError("Implement get_branch_by_id method")

    @abstractmethod
    def delete_branch_by_Id(self, branch_id: str) -> dict:
        raise NotImplementedError("Implement delete_branch method")
    
    @abstractmethod
    def get_all_branches(self) -> list[BranchData]:
        raise NotImplementedError("Implement get_all_branches method")
   