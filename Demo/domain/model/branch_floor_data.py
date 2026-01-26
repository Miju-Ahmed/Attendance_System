from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from .time_data import TimeData
from .user_data import UserData


class BranchFloorData(BaseModel):
    floor_id: UUID = Field(default_factory=uuid4)
    branch_id: str
    floor_name: str 
    description: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)


    def to_json(self):
        return {
            "floor_id": str(self.floor_id),
            "branch_id": self.branch_id,
            "floor_name": self.floor_name,
            "description": self.description,
            "created_at": self.timedata.created_at.isoformat() if self.timedata.created_at else None,
            "modified_at": self.timedata.modified_at.isoformat() if self.timedata.modified_at else None,
            "user_created": str(self.userdata.user_created.user_id) if self.userdata.user_created else None,
            "user_modified": str(self.userdata.user_modified.user_id) if self.userdata.user_modified else None,
        }