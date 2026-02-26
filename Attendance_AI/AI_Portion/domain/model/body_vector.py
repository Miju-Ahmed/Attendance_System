from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime
from .time_data import TimeData


class BodyVector:
    body_vector_id: UUID = Field(default_factory=uuid4)
    employee_id: UUID | None = None
    vector_data: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)