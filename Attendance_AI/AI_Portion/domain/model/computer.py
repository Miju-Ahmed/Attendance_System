from uuid import UUID , uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path
from .computer_status import ComputerStatus
from .time_data import TimeData


class Computer:
    computer_id: UUID = Field(default_factory=uuid4)
    employee_id: UUID  | None = None
    computer_name: str | None = None
    ip_address: str | None = None
    status: ComputerStatus = Field(default=ComputerStatus.ACTIVE)
    timedata: TimeData = Field(default_factory=TimeData)