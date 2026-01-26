from __future__ import annotations

from uuid import UUID

from attrs import define, field, validators

from domain.model import RecordedStream
from domain.repository import RecordedStreamRepository


@define
class GetRecordedStreamById:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )

    def invoke(self, file_id: str) -> RecordedStream | None:
        return self.recorded_stream_repository.get_recorded_stream(UUID(file_id))
