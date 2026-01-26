from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID


class RecordedStreamStorage(Protocol):
    def allocate_path(self, file_id: UUID, original_file_name: str) -> tuple[Path, str]:
        """Allocate a filesystem path for the recorded stream."""

    def write_chunk(self, file_path: Path, chunk: bytes, chunk_index: int) -> int:
        """Append a chunk to the recorded stream file and return bytes written."""

