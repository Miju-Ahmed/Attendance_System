from uuid import UUID , uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path
from .time_data import TimeData

class ScreenActivityData(BaseModel):
    screen_activity_id: UUID = Field(default_factory=uuid4)
    computer_id: UUID | None = None
    screenshot_url: str | None = None
    detected_activity: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
