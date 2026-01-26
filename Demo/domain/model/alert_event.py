from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AlertEvent(BaseModel):
    alert_event_id: UUID = Field(default_factory=uuid4)
    stream_id: Optional[UUID] = None
    event_id: Optional[UUID] = None
    branch_id: Optional[UUID] = None
    floor_id: Optional[UUID] = None
    camera_no: str | None = None
    track_id: int | None = None
    duration_seconds: int = 0
    snapshot_path: str | None = None
    snapshot_url: str | None = None
    status: str = "open"
    employee_id: Optional[UUID] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    resolved_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "alert_event_id": str(self.alert_event_id),
            "stream_id": str(self.stream_id) if self.stream_id else None,
            "event_id": str(self.event_id) if self.event_id else None,
            "branch_id": str(self.branch_id) if self.branch_id else None,
            "floor_id": str(self.floor_id) if self.floor_id else None,
            "camera_no": self.camera_no,
            "track_id": self.track_id,
            "duration_seconds": self.duration_seconds,
            "snapshot_path": self.snapshot_path,
            "snapshot_url": self.snapshot_url,
            "status": self.status,
            "employee_id": str(self.employee_id) if self.employee_id else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }
