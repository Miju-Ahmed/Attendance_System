from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, date, timedelta
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .video_status import VideoStatus
from .time_data import TimeData
from .user import User
from .user_data import UserData
from utils import get_logger
logger = get_logger(__name__)


FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z]+)(?P<number>\d+)_(?P<timestamp>\d{14})$"
)


def parse_recorded_stream_filename(file_name: str) -> tuple[str, int, datetime, str]:
    """Parse recorded stream file name and return metadata.

    Args:
        file_name: Name of the uploaded file (e.g., ``C1_20251122150437.mp4``).

    Returns:
        A tuple containing the camera label (e.g., ``C1``), the camera number,
        the parsed start time as a ``datetime`` instance, and a human friendly
        camera name (e.g., ``Camera 1``).

    Raises:
        ValueError: If the file name does not match the required pattern.
    """
    logger.info(f"Parsing recorded stream filename: {file_name}")
    stem = Path(file_name).stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(
            "File name must follow the pattern <Camera><Number>_YYYYMMDDHHMMSS.ext"
        )

    camera_prefix = match.group("prefix")
    camera_number = int(match.group("number"))
    timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S")
    camera_label = f"{camera_prefix}{camera_number}"
    camera_name = f"Camera {camera_number}"
    return camera_label, camera_number, timestamp, camera_name


class RecordedStream(BaseModel):
    identifier: UUID = Field(default_factory=uuid4)
    original_file_name: str
    stored_file_name: str | None = None
    filepath: str | None = None
    file_url: str | None = None
    processed_filepath: str | None = None
    event_id: UUID | None = None
    camera_code: str | None = None
    camera_name: str | None = None
    camera_number: int | None = None
    video_start_time: datetime | None = None
    video_end_time: datetime | None = None
    video_date: date | None = None
    total_chunks: int | None = None
    uploaded_chunks: int = 0
    file_size: int = 0
    video_status: VideoStatus = Field(default=VideoStatus.UNPROCESSED)
    processing_status: bool = False
    processing_type: str | None = None
    process_datetime: datetime | None = None
    processing_duration: timedelta | None = None
    duration: timedelta | None = None
    pipeline: str | None = None
    errors: str | None = None
    user_uploaded: UUID | None = None
    user_processed: UUID | None = None
    stream_id: UUID | None = None
    timedata: TimeData = Field(default_factory=TimeData)
    userdata: UserData = Field(default_factory=UserData)

    def mark_modified(self, user_id: UUID | None = None) -> None:
        if user_id:
            self.userdata.user_modified = User(user_id=user_id)
        self.timedata.modified_at = datetime.now()

    @property
    def file_id(self) -> UUID:
        """Backward compatible identifier access."""
        return self.identifier

    @property
    def is_upload_complete(self) -> bool:
        if self.total_chunks is None:
            return False
        return self.uploaded_chunks >= self.total_chunks

