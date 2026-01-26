import uuid
from uuid import UUID , uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from .time_data import TimeData
from .user_data import UserData

class Employee(BaseModel):
    employee_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    branch: str | None = None
    branch_id: str | None = None
    department: str | None = None
    role: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)

    def to_json(self):
        return {
            "employee_id": str(self.employee_id),
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone_number": self.phone_number,
            "branch": self.branch,
            "branch_id": self.branch_id,
            "department": self.department,
            "role": self.role,
            "created_at": self.timedata.created_at.isoformat() if self.timedata.created_at else None,
            "modified_at": self.timedata.modified_at.isoformat() if self.timedata.modified_at else None,
        }    