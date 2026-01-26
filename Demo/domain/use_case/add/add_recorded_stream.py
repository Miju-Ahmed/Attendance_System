from __future__ import annotations

from datetime import datetime
from uuid import UUID

from attrs import define, field, validators

from domain.model import RecordedStream, VideoStatus, TimeData, User, UserData, parse_recorded_stream_filename
from domain.repository import RecordedStreamRepository, StreamRepository
from domain.service import RecordedStreamStorage
from utils import MakeUrl


@define
class AddRecordedStream:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )
    recorded_stream_storage: RecordedStreamStorage
    stream_repository: StreamRepository | None = field(default=None)

    def invoke(
        self,
        original_file_name: str,
        total_chunks: int,
        user_created: str | None = None,
        stream_id: str | None = None,
    ) -> RecordedStream:
        camera_code, camera_number, start_time, camera_name = parse_recorded_stream_filename(
            original_file_name
        )

        resolved_stream_id = self._resolve_stream_id(stream_id, camera_code)

        recorded_stream = RecordedStream(
            original_file_name=original_file_name,
            camera_code=camera_code,
            camera_number=camera_number,
            camera_name=camera_name,
            video_start_time=start_time,
            video_date=start_time.date(),
            total_chunks=total_chunks,
            video_status=VideoStatus.UNPROCESSED,
            stream_id=resolved_stream_id,
            user_uploaded=UUID(user_created) if user_created else None,
            timedata=TimeData(created_at=datetime.now(), modified_at=datetime.now()),
            userdata=UserData(
                user_created=User(user_id=UUID(user_created)) if user_created else None
            ),
        )

        file_path, stored_file_name = self.recorded_stream_storage.allocate_path(
            recorded_stream.identifier, original_file_name
        )
        recorded_stream.filepath = str(file_path)
        try:
            recorded_stream.file_url = MakeUrl().invoke(path=str(file_path))
        except ValueError:
            recorded_stream.file_url = str(file_path)
        recorded_stream.stored_file_name = stored_file_name

        return self.recorded_stream_repository.create_recorded_stream(recorded_stream)

    def _resolve_stream_id(
        self, provided_stream_id: str | None, camera_code: str | None
    ) -> UUID | None:
        if provided_stream_id:
            return UUID(provided_stream_id)

        if self.stream_repository and camera_code:
            stream = self.stream_repository.get_stream_by_name(camera_code)
            if stream:
                return stream.stream_id

        return None
