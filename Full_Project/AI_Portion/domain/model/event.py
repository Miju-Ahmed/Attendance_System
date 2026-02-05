from uuid import UUID
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path
from .time_data import TimeData

class Event(BaseModel):
    event_id: UUID  | None = None
    task_id: UUID | None = None
    event_type: str | None = None
    status: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)

