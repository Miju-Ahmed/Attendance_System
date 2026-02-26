from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime, date
from domain.model import Stream, TimeData, UserData, StreamStatus, User
from domain.repository import StreamRepository
from utils import get_logger
logger = get_logger(__name__)

@define
class AddStream:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )
    def invoke(
            self, 
            username: str,
            password: str,
            stream_name: str,
            host: str,
            port: str,
            endpoint: str,
            trasport_mode: str | None = None,
            profile: str | None = None,
            rtsp_url: str | None = None,
            user_created: UUID | None = None,
            branch_id: UUID | None = None,
            floor_id: UUID | None = None
        ) -> dict:
        processed_rtsp_url = self._make_url(
            username=username,
            password=password,
            host=host,
            port=port,
            endpoint=endpoint,
            transport_mode=trasport_mode,
            profile=profile
        ) if rtsp_url is None else rtsp_url
        logger.info(f"Processed RTSP URL: {processed_rtsp_url}")
        stream = Stream(
            stream_id=uuid4(),
            stream_name=stream_name,
            stream_url=processed_rtsp_url,
            rtsp_username=username,
            rtsp_password=password,
            rtsp_host=host,
            rtsp_port=port,
            rtsp_endpoint=endpoint,
            rtsp_transport_mode=trasport_mode,
            rtsp_stream_profile=profile,
            timedata= TimeData(
                created_at=datetime.now(),
                modified_on=None
            ),
            userdata= UserData(
                user_created=User(user_id=user_created),
                user_modified=None
            ),
            stream_status=StreamStatus.STOPPED,
            branch_id=branch_id,
            floor_id=floor_id
            
        )
        return self.stream_repository.add_stream(stream=stream)
    

    def _make_url(
            self,
            username: str,
            password: str,
            host: str,
            port: str,
            endpoint: str,
            transport_mode: str | None = None,
            profile: str | None = None
        ) -> str:
        base_url = f"rtsp://{username}:{password}@{host}:{port}/{endpoint}"
        if transport_mode and profile:
            url = f"{base_url}?transport={transport_mode}&profile={profile}"
        else:
            url = base_url
        logger.info(f"RTSP URL constructed with transport mode: {transport_mode} and profile: {profile}")
        return url
