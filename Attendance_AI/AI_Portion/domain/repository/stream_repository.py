from abc import ABC, abstractmethod
from uuid import UUID

from ..model import Stream

class StreamRepository(ABC):
    @abstractmethod
    def add_stream(self, stream: Stream) -> dict:
        raise NotImplementedError("Implement add_stream method")
    
    @abstractmethod
    def get_stream(self, stream_id: UUID) -> Stream:
        raise NotImplementedError("Implement get_stream method")

    @abstractmethod
    def get_stream_by_name(self, stream_name: str) -> Stream | None:
        raise NotImplementedError("Implement get_stream_by_name method")

    @abstractmethod
    def get_all_streams(
        self,
        sort_by: str = "created_at",
        order: str = "desc",
        stream_name: str = None,
        user_created: str = None,
        user_modified: str = None,
        stream_status: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> list[Stream]:
        raise NotImplementedError("Implement get_all_streams method")
    
    @abstractmethod
    def delete_stream_by_id(self, stream_id: UUID) -> dict:
        raise NotImplementedError("Implement delete_stream_by_id method")

    @abstractmethod
    def update_stream(self, stream: Stream) -> dict:
        raise NotImplementedError("Implement update_stream method")
    
    @abstractmethod
    def get_streams_by_branch_id(self, branch_id: UUID) -> list[Stream]:
        raise NotImplementedError("Implement get_streams_by_branch_id method")
