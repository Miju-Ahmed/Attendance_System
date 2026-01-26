from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from domain.model import RecordedStream


class RecordedStreamRepository(ABC):
    @abstractmethod
    def create_recorded_stream(self, recorded_stream: RecordedStream) -> RecordedStream:
        """Persist a newly created recorded stream."""

    @abstractmethod
    def update_recorded_stream(self, recorded_stream: RecordedStream) -> RecordedStream:
        """Persist updates to an existing recorded stream."""

    @abstractmethod
    def get_recorded_stream(self, file_id: UUID) -> Optional[RecordedStream]:
        """Retrieve a recorded stream by its identifier."""

    @abstractmethod
    def get_recorded_stream_by_file_name(self, file_name: str) -> Optional[RecordedStream]:
        """Retrieve a recorded stream by its original file name."""

    @abstractmethod
    def get_recorded_streams(self) -> list[RecordedStream]:
        """Return every recorded stream that has been uploaded."""

