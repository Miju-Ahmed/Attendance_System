from attrs import define, field, validators
from domain.repository import BranchRepository

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteBranchById:
    branch_repository: BranchRepository = field(
        validator=validators.instance_of(BranchRepository)
    )
    def invoke(self, branch_id: str) -> dict:
        return self.branch_repository.delete_branch_by_Id(branch_id)

    