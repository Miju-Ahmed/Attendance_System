import uuid
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path
from django.http import JsonResponse

from .time_data import TimeData
from .stream_status import StreamStatus
from .user_data import UserData

class Stream(BaseModel):
    stream_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    stream_url: str | None = None
    stream_name: str | None = None
    rtsp_username: str | None = None
    rtsp_password: str | None = None
    rtsp_host: str | None = None
    rtsp_port: str | None = None
    rtsp_endpoint: str | None = None
    rtsp_transport_mode: str | None = None
    rtsp_stream_profile: str | None = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)
    stream_status: str = StreamStatus.STOPPED.value
    branch_id: uuid.UUID | None = None
    branch_name: str | None = None
    floor_id: uuid.UUID | None = None

    def to_json(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "stream_url": self.stream_url,
            "stream_name": self.stream_name,
            "username": self.rtsp_username,
            "host": self.rtsp_host,
            "port": self.rtsp_port,
            "endpoint": self.rtsp_endpoint,
            "transport_mode": self.rtsp_transport_mode,
            "stream_profile": self.rtsp_stream_profile,
            "created_at": self.timedata.created_at,
            "user_created": self.userdata.user_created.user_name if self.userdata.user_created else None,
            "modified_on": self.timedata.modified_at,
            "user_modified": self.userdata.user_modified.user_name if self.userdata.user_modified else None,
            "stream_status": self.stream_status,
            "branch_id": str(self.branch_id) if self.branch_id else None,
            "branch_name": self.branch_name if self.branch_name else "Unknown",
            "floor_id": str(self.floor_id) if self.floor_id else None,
        }
