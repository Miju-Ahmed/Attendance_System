from datetime import date
from attrs import define, field, validators
from domain.model import Stream
from domain.repository import StreamRepository

from utils import get_logger

logger = get_logger(__name__)


@define
class GetAllStreams:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )

    def invoke(
        self,
        sort_by: str = "created_at",
        order: str = "desc",
        stream_name: str = None,
        user_created: str = None,
        user_modified: str = None,
        stream_status: str = None,
        start_date: date = None,
        end_date: date = None,
    ) -> list[Stream]:
        try:
            streams = self.stream_repository.get_all_streams(
                sort_by=sort_by,
                order=order,
                stream_name=stream_name,
                user_created=user_created,
                user_modified=user_modified,
                stream_status=stream_status,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"Retrieved {len(streams)} streams")
            return streams
        except Exception as e:
            logger.error(f"Error retrieving streams: {e}")
            return []
