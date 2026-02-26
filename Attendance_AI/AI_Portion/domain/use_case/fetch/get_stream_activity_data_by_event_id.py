from attrs import define, field, validators
from domain.model import StreamActivityData
from domain.repository import StreamActivityRepository

@define
class GetStreamActivityByEventId:
    stream_activity_data_repository: StreamActivityRepository = field(
        validator=validators.instance_of(StreamActivityRepository)
    )
    
    def invoke(
        self,
        event_id : str,
    ) -> StreamActivityData:
        stream_activity_data = self.stream_activity_data_repository.get_stream_activity_by_event_id(event_id=event_id)
        return stream_activity_data