from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime, time
from domain.model import  TimeData, UserData, BranchData , User
from domain.repository import BranchRepository
from utils import get_logger
logger = get_logger(__name__)

@define
class UpdateBranch:
    branch_repository: BranchRepository= field(
        validator=validators.instance_of(BranchRepository)
    )
    def invoke(
            self, 
            branch_id: UUID,
            branch_name: str,
            working_days: list[str] ,
            address: str ,
            mobile: str ,
            email: str ,
            overtime: time | None = None,
            office_start_time: datetime | None = None,
            office_end_time: datetime | None = None,
            off_days: list[str] | None = None,
            casual_day: list[str] | None = None,
            user_created: UUID | None = None,
            user_modified: UUID | None = None,
            created_at: datetime | None = None,
        ) -> dict:
        branch = BranchData(
            branch_id= branch_id,
            branch_name=branch_name,
            office_start_time=office_start_time,
            office_end_time=office_end_time,
            overtime=overtime,
            working_days=working_days,
            off_days=off_days,
            casual_day=casual_day,
            address=address,
            mobile=mobile,
            email=email,
            timedata= TimeData(
                created_at=created_at,
                modified_at=datetime.now(),
                office_start_time=office_start_time,
                office_end_time=office_end_time,
            ),
            userdata= UserData(
                user_created=User(user_id=user_created),
                user_modified=User(user_id=user_modified)
            )
        )
        logger.info(f"Branch Data: {branch}")
        return self.branch_repository.update_branch(branch)
        