import uuid
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from .user_data import UserData
from .time_data import TimeData

class Role(BaseModel):
    identifier: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    userdata: UserData = Field(default_factory=UserData)
    timedata: TimeData = Field(default_factory=TimeData)

    def to_json(self) -> dict:
        return {
            "identifier": str(self.identifier),
            "name": self.name,
            "description": self.description,
            "created_at": self.timedata.created_at,
            "modified_at": self.timedata.modified_at,
            "created_by": str(self.userdata.user_created.user_id) if self.userdata.user_created else None,
            "modified_by": str(self.userdata.user_modified.user_id) if self.userdata.user_modified else None,
        }