from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

from attrs import define, field, validators

from domain.model import RecordedStream, VideoStatus, User
from domain.repository import RecordedStreamRepository
from domain.service import RecordedStreamStorage


@define
class UploadRecordedStreamChunk:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )
    recorded_stream_storage: RecordedStreamStorage

    def invoke(
        self,
        file_id: str,
        chunk: bytes,
        chunk_index: int,
        total_chunks: int,
        user_id: str | None = None,
    ) -> RecordedStream:
        recorded_stream = self.recorded_stream_repository.get_recorded_stream(UUID(file_id))
        if recorded_stream is None:
            raise ValueError("Recorded stream not found")

        if recorded_stream.total_chunks is None or recorded_stream.total_chunks == 0:
            recorded_stream.total_chunks = total_chunks

        if chunk_index < recorded_stream.uploaded_chunks:
            # Chunk already processed; ignore duplicate transmissions.
            return recorded_stream

        if chunk_index != recorded_stream.uploaded_chunks:
            raise ValueError("Chunks must be uploaded sequentially")

        bytes_written = self.recorded_stream_storage.write_chunk(
            Path(recorded_stream.filepath), chunk, chunk_index
        )
        recorded_stream.uploaded_chunks += 1
        recorded_stream.file_size += bytes_written
        recorded_stream.video_status = VideoStatus.UNPROCESSED
        recorded_stream.timedata.modified_at = datetime.now()
        if user_id:
            recorded_stream.userdata.user_modified = User(user_id=UUID(user_id))
            recorded_stream.user_uploaded = recorded_stream.user_uploaded or UUID(user_id)

        return self.recorded_stream_repository.update_recorded_stream(recorded_stream)
