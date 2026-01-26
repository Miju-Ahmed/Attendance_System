from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from .time_data import TimeData
from .user_data import UserData


class AccessPermission(BaseModel):
    identifier: UUID = Field(default_factory=uuid4)
    role_id: UUID
    permissions: list[str] = Field(default_factory=list)
    userdata: UserData = Field(default_factory=UserData)
    timedata: TimeData = Field(default_factory=TimeData)

    def to_json(self):
        return {
            "identifier": str(self.identifier),
            "role_id": str(self.role_id),
            "permissions": self.permissions,
            "created_at": self.timedata.created_at,
            "modified_at": self.timedata.modified_at,
            "created_by": str(self.userdata.user_created.user_id) if self.userdata.user_created else None,
            "modified_by": str(self.userdata.user_modified.user_id) if self.userdata.user_modified else None,
        }
