from uuid import UUID , uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from .time_data import TimeData
from .user_data import UserData

class Region(BaseModel):
    region_id: UUID = Field(default_factory=uuid4)
    region_name: str = None
    description: str = None
    stream_id: UUID | None = None
    region_bounding_box: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)
    