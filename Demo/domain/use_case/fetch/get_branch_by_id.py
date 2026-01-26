from attrs import define, field, validators
from domain.model import BranchData
from domain.repository import BranchRepository
from uuid import UUID

@define
class GetBranchById:
    branch_repository: BranchRepository= field(
        validator=validators.instance_of(BranchRepository)
    )
    def invoke(
        self,
        branch_id: UUID,
    ) -> BranchData:
        return self.branch_repository.get_branch_by_id(branch_id)