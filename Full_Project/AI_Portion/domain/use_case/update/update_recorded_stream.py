from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from attrs import define, field, validators

from domain.model import RecordedStream, VideoStatus, User
from domain.repository import RecordedStreamRepository


@define
class UpdateRecordedStream:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )

    def invoke(
        self,
        file_id: str,
        video_status: VideoStatus | None = None,
        processed_filepath: str | None = None,
        event_id: str | None = None,
        processing_status: bool | None = None,
        process_datetime: datetime | None = None,
        processing_duration: timedelta | None = None,
        user_modified: str | None = None,
        user_processed: str | None = None,
        errors: str | None = None,
        pipeline: str | None = None,
        processing_type: str | None = None,
        file_url: str | None = None,
    ) -> RecordedStream | None:
        recorded_stream = self.recorded_stream_repository.get_recorded_stream(UUID(file_id))
        if not recorded_stream:
            return None

        if video_status:
            recorded_stream.video_status = video_status
        if processed_filepath:
            recorded_stream.processed_filepath = processed_filepath
        if event_id:
            recorded_stream.event_id = UUID(event_id)
        if processing_status is not None:
            recorded_stream.processing_status = processing_status
        if process_datetime:
            recorded_stream.process_datetime = process_datetime
        if processing_duration:
            recorded_stream.processing_duration = processing_duration
        if user_modified:
            recorded_stream.userdata.user_modified = User(user_id=UUID(user_modified))
            recorded_stream.user_processed = recorded_stream.user_processed or UUID(user_modified)
        if user_processed:
            recorded_stream.user_processed = UUID(user_processed)
        if errors is not None:
            recorded_stream.errors = errors
        if pipeline:
            recorded_stream.pipeline = pipeline
        if processing_type:
            recorded_stream.processing_type = processing_type
        if file_url:
            recorded_stream.file_url = file_url
        recorded_stream.timedata.modified_at = datetime.now()

        return self.recorded_stream_repository.update_recorded_stream(recorded_stream)
