from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Stream, TimeData, User, UserData
from domain.repository import StreamRepository


@define
class UpdateStream:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )
    def invoke(self, stream: Stream, user_modified: str) -> dict:
        stream.userdata = UserData(
            user_created=stream.userdata.user_created if stream.userdata.user_created is not None else User(user_id=user_modified),
            user_modified=User(user_id=user_modified) if user_modified is not None else stream.userdata.user_modified
        )
        stream.timedata.modified_at = timezone.now()
        return self.stream_repository.update_stream(stream=stream)