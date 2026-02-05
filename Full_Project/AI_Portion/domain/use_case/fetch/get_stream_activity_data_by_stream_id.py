from attrs import define, field, validators
from domain.model import StreamActivityData
from domain.repository import StreamActivityRepository

@define
class GetStreamActivityByStreamId:
    stream_activity_data_repository: StreamActivityRepository = field(
        validator=validators.instance_of(StreamActivityRepository)
    )
    
    def invoke(
        self,
        stream_id: str,
    ) -> list[StreamActivityData]:
        stream_activity_data = self.stream_activity_data_repository.get_stream_activity_by_stream_id(stream_id=stream_id)
        return stream_activity_data