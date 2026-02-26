from attrs import define, field, validators
from domain.model import Stream
from domain.repository import StreamRepository

@define
class GetStreamById:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )
    
    def invoke(
        self,
        stream_id: str,
    ) -> Stream:
        stream = self.stream_repository.get_stream(stream_id=stream_id)
        return stream