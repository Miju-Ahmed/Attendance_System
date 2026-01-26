from uuid import UUID , uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path
from .time_data import TimeData

class FaceVector(BaseModel):
    face_vector_id: UUID = Field(default_factory=uuid4)
    employee_id: UUID | None = None
    vector_data: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
