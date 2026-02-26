from uuid import UUID, uuid4
from django.http import JsonResponse
from pydantic import BaseModel, Field
from datetime import datetime
from .stream_status import StreamStatus
from .time_data import TimeData
from .user_data import UserData



class StreamActivityData(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    success: bool = Field(default=False)
    event_id: UUID | None = None
    stream_id: UUID | None = None
    stream_url: str | None = None
    pipeline: str | None = None
    processed_uri: str | None = None
    status: StreamStatus = Field(default=StreamStatus.PROCESSING)
    errors: str | None = None
    timedata : TimeData = Field(default_factory=TimeData)
    userdata : UserData = Field(default_factory=UserData)
    stream_datetime: datetime = Field(default_factory=datetime.now)
    
