from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime, date
from domain.model import  TimeData, UserData, BranchFloorData, User
from domain.repository import BranchFloorRepository
from datetime import time
from utils import get_logger
logger = get_logger(__name__)

@define
class AddBranchFloor:
    branch_repository: BranchFloorRepository= field(
        validator=validators.instance_of(BranchFloorRepository)
    )
    def invoke(
            self, 
            branch_id: str,
            floor_name: str,
            description: str | None = None,
            user_created: UUID | None = None
        ) -> dict:
        branch_floor = BranchFloorData(
            floor_id=uuid4(),
            branch_id=branch_id,
            floor_name=floor_name,
            description=description,
            timedata= TimeData(
                created_at=datetime.now(),
                modified_at=None,
            ),
            userdata= UserData(
                user_created=User(
                    user_id=user_created,
                ),
                user_modified=None
            )
        )
        logger.info(f"Branch Data: {branch_floor}")
        return self.branch_repository.add_branch_floor(branch_floor)
        