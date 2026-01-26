from attrs import define, field, validators

from domain.model import RecordedStream
from domain.repository import RecordedStreamRepository
from utils import get_logger

logger = get_logger(__name__)


@define
class GetRecordedStreams:
    recorded_stream_repository: RecordedStreamRepository = field(
        validator=validators.instance_of(RecordedStreamRepository)
    )

    def invoke(self) -> list[RecordedStream]:
        try:
            recorded_streams = self.recorded_stream_repository.get_recorded_streams()
            logger.info("Retrieved %s recorded streams", len(recorded_streams))
            return recorded_streams
        except Exception as exc:
            logger.error("Failed to retrieve recorded streams: %s", exc)
            return []
