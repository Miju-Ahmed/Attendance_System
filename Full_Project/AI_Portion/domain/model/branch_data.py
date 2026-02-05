from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional
from datetime import  time
from .time_data import TimeData
from .user_data import UserData


class BranchData(BaseModel):
    branch_id: UUID = Field(default_factory=uuid4)
    branch_name: str 
    working_days: List[str] = None
    off_days: List[str] = None     
    casual_day: List[str] = None
    address: str = None
    mobile: str = None
    email: str = None
    overtime: time = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)
    

    def to_json(self):
        return {
            "branch_id": str(self.branch_id),
            "branch_name": self.branch_name,
            "working_days": self.working_days,
            "off_days": self.off_days,
            "casual_day": self.casual_day,
            "office_start_time": str(self.timedata.office_start_time),
            "office_end_time": str(self.timedata.office_end_time),
            "address": self.address,
            "mobile": self.mobile,
            "email": self.email,
            "overtime": str(self.overtime),
            "created_at": self.timedata.created_at.isoformat() if self.timedata.created_at else None,
            "modified_at": self.timedata.modified_at.isoformat() if self.timedata.modified_at else None,
        }
  