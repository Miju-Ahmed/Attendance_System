from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime, date
from domain.model import  TimeData, UserData, BranchData, User
from domain.repository import BranchRepository
from datetime import time
from utils import get_logger
logger = get_logger(__name__)

@define
class AddBranch:
    branch_repository: BranchRepository= field(
        validator=validators.instance_of(BranchRepository)
    )
    def invoke(
            self, 
            branch_name: str,
            working_days: list[str] ,
            address: str ,
            mobile: str ,
            email: str ,
            office_start_time: time | None = None,
            office_end_time: time | None = None,
            off_days: list[str] | None = None,
            overtime: datetime | None = None,
            casual_day: list[str] | None = None,
            user_created: UUID | None = None
        ) -> dict:
        branch = BranchData(
            branch_id=uuid4(),
            branch_name=branch_name,
            working_days=working_days,
            off_days=off_days,
            overtime=overtime,
            casual_day=casual_day,
            address=address,
            mobile=mobile,
            email=email,
            timedata= TimeData(
                created_at=datetime.now(),
                modified_at=None,
                office_start_time=office_start_time,
                office_end_time=office_end_time,
            ),
            userdata= UserData(
                user_created=User(
                    user_id=user_created,
                ),
                user_modified=None
            )
        )
        logger.info(f"Branch Data: {branch}")
        return self.branch_repository.add_branch(branch)
        