from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional
from ..model import StreamActivityData

class StreamActivityRepository(ABC):
    @abstractmethod
    def add_stream_activity(self, stream_activity: StreamActivityData) -> StreamActivityData:
        raise NotImplementedError("Implement add_stream_activity method")

    @abstractmethod
    def update_stream_activity(self, stream_activity: StreamActivityData) -> dict:
        raise NotImplementedError("Implement update_stream_activity method")

    @abstractmethod
    def get_stream_activity_by_id(self, stream_activity_id: str) -> StreamActivityData:
        raise NotImplementedError("Implement get_stream_activity_by_id method")
    
    @abstractmethod
    def get_stream_activity_by_stream_id(self, stream_id: str) -> list[StreamActivityData]:
        raise NotImplementedError("Implement get_stream_activity_by_stream_id method")
    
    def get_stream_activity_by_event_id(self, event_id: str) -> StreamActivityData:
        raise NotImplementedError("Implement get_stream_activity_by_event_id method")

    @abstractmethod
    def delete_stream_activity(self, stream_activity_id: str) -> bool:
        raise NotImplementedError("Implement delete_stream_activity method")
