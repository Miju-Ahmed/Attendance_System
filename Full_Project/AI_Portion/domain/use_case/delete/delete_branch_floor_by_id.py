from attrs import define, field, validators
from domain.repository import BranchFloorRepository

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteBranchFloorById:
    branch_repository: BranchFloorRepository = field(
        validator=validators.instance_of(BranchFloorRepository)
    )
    def invoke(self, branch_id: str, floor_id : str) -> dict:
        return self.branch_repository.delete_branch_floor(branch_id , floor_id)

    