from __future__ import annotations

from attrs import define, field, validators

from domain.model import RecordedStream
from domain.repository import RecordedStreamRepository


@define
class GetRecordedStreamByFileName:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )

    def invoke(self, file_name: str) -> RecordedStream | None:
        return self.recorded_stream_repository.get_recorded_stream_by_file_name(
            file_name
        )
