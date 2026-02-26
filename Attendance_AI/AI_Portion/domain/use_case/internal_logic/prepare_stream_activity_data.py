import os
from pathlib import Path
from uuid import UUID
from environs import Env
from random import randint
from attrs import define, field, validators
from datetime import datetime, date, time
from domain.model import Stream, StreamActivityData, StreamStatus, TimeData, UserData, User
from domain.repository import StreamRepository, StreamActivityRepository

from utils import get_logger
logger = get_logger(__name__)

env = Env()
env.read_env()

@define
class PrepareStreamActivityData:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )
    stream_data_repository: StreamActivityRepository = field(
        validator=validators.instance_of(StreamActivityRepository)
    )

    def invoke(
        self,
        stream_id: str,
        event_id: str | None = None,
        user_streamed: str | None = None,
    ) -> StreamActivityData:
        stream = self.stream_repository.get_stream(stream_id=stream_id)
        stream.stream_status = StreamStatus.PROCESSING
        stream.userdata.user_modified = User(user_id=UUID(user_streamed))
        stream.timedata.modified_at = datetime.now()
        result = self.stream_repository.update_stream(stream=stream)
        pipeline_version = env.str("PIPELINE_VERSION")
        if result["success"]:
            try:
                stream_data = self.stream_data_repository.add_stream_activity(
                    stream_activity=StreamActivityData(
                        success=True,
                        event_id=UUID(event_id) if event_id else None,
                        stream_id=UUID(stream_id),
                        stream_url=stream.stream_url,
                        pipeline=pipeline_version,
                        processed_uri=self._make_path(),
                        status=stream.stream_status,
                        stream_datetime=datetime.now(),
                        timedata=TimeData(
                            start_time=datetime.now(),
                            video_date=date.today(),
                            created_at=datetime.now(),
                            modified_on=stream.timedata.modified_at,
                        ),
                        userdata=UserData(
                            user_processed=User(user_id=UUID(user_streamed)),
                            user_modified=User(user_id=UUID(user_streamed)),
                        ),
                    )              
                )
                return stream_data
            
            except Exception as e:
                logger.error(f"Error adding stream data: {e}")
                stream.stream_status = StreamStatus.STOPPED
                self.stream_repository.update_stream(stream=stream)
                return StreamActivityData(success=False, errors=str(e))
        else:
            return StreamActivityData(success=False, errors="Error updating stream")
            

    
    def _make_path(self) -> str:
        date_path = date.today().strftime("%Y_%m_%d")
        base_path = Path("mediafiles/stream") / date_path
        base_path.mkdir(parents=True, exist_ok=True)
        return str(base_path / f"{randint(1000,5000)}.mp4")
