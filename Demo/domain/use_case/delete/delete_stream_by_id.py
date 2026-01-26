from attrs import define, field, validators
from domain.repository import StreamRepository

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteStreamById:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )

    def invoke(self, stream_id: str) -> dict:
        return self.stream_repository.delete_stream_by_id(stream_id)