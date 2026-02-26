from attrs import define, field, validators
from domain.model import BranchData
from domain.repository import BranchRepository
from typing import List

@define
class GetAllBranches:
    branch_repository: BranchRepository = field(
        validator=validators.instance_of(BranchRepository)
    )
    
    def invoke(self) -> List[BranchData]:
        return self.branch_repository.get_all_branches()
        