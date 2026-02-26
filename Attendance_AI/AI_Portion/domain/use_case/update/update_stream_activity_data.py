from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import StreamActivityData, TimeData, User, UserData
from domain.repository import StreamActivityRepository


@define
class UpdateStreamActivityData:
    stream_activity_data_repository: StreamActivityRepository = field(
        validator=validators.instance_of(StreamActivityRepository)
    )
    def invoke(self, stream_data: StreamActivityData) -> dict:
        return self.stream_activity_data_repository.update_stream_activity(stream_activity=stream_data)