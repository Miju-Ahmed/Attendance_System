from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from random import randint
from uuid import UUID

from attrs import define, field, validators
from environs import Env
from django.utils import timezone

from domain.model import (
    RecordedStream,
    StreamActivityData,
    StreamStatus,
    TimeData,
    User,
    UserData,
    VideoStatus,
)
from domain.repository import RecordedStreamRepository, StreamActivityRepository

from utils import get_logger


logger = get_logger(__name__)
env = Env()
env.read_env()


@define
class PrepareRecordedStreamActivityData:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )
    stream_activity_repository: StreamActivityRepository = field(
        validator=validators.instance_of(StreamActivityRepository)
    )

    def invoke(
        self,
        file_id: str,
        event_id: str | None = None,
        user_streamed: str | None = None,
        processing_type: str | None = None,
    ) -> StreamActivityData:
        recorded_stream = self.recorded_stream_repository.get_recorded_stream(UUID(file_id))
        if recorded_stream is None:
            logger.error("Recorded stream %s not found", file_id)
            return StreamActivityData(success=False, errors="Recorded stream not found")

        recorded_stream.video_status = VideoStatus.PROCESSING
        recorded_stream.event_id = UUID(event_id) if event_id else None
        recorded_stream.processing_status = True
        now = timezone.now()
        recorded_stream.process_datetime = now
        recorded_stream.processing_type = processing_type
        recorded_stream.timedata.modified_at = now
        if user_streamed:
            recorded_stream.userdata.user_modified = User(user_id=UUID(user_streamed))
            recorded_stream.user_processed = UUID(user_streamed)

        processed_path = self._make_path()
        recorded_stream.processed_filepath = processed_path

        pipeline_version = env.str("PIPELINE_VERSION")
        recorded_stream.pipeline = pipeline_version

        self.recorded_stream_repository.update_recorded_stream(recorded_stream)

        try:
            stream_activity = self.stream_activity_repository.add_stream_activity(
                StreamActivityData(
                    success=True,
                    event_id=UUID(event_id) if event_id else None,
                    stream_id=recorded_stream.identifier,
                    stream_url=recorded_stream.filepath,
                    pipeline=pipeline_version,
                    processed_uri=processed_path,
                    status=StreamStatus.PROCESSING,
                    stream_datetime=recorded_stream.video_start_time or now,
                    timedata=TimeData(
                        start_time=recorded_stream.video_start_time or now,
                        created_at=now,
                        modified_at=recorded_stream.timedata.modified_at,
                        date_of_activity=recorded_stream.video_date or date.today(),
                    ),
                    userdata=UserData(
                        user_processed=User(user_id=UUID(user_streamed)) if user_streamed else None,
                        user_modified=User(user_id=UUID(user_streamed)) if user_streamed else None,
                    ),
                )
            )
            return stream_activity
        except Exception as error:  # pragma: no cover - defensive
            logger.error("Failed to prepare recorded stream activity data: %s", error)
            recorded_stream.video_status = VideoStatus.FAILED
            recorded_stream.processing_status = False
            recorded_stream.errors = str(error)
            self.recorded_stream_repository.update_recorded_stream(recorded_stream)
            return StreamActivityData(success=False, errors=str(error))

    @staticmethod
    def _make_path() -> str:
        base_path = Path("mediafiles/recorded_streams") / date.today().strftime("%Y_%m_%d")
        os.makedirs(base_path, exist_ok=True)
        return str(base_path / f"processed_{randint(1000, 9999)}.mp4")
