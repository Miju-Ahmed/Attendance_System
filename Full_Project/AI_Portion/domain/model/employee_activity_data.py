from uuid import UUID, uuid4
from django.http import JsonResponse
from pydantic import BaseModel, Field
from datetime import datetime
from . time_data import TimeData
from . emotion_stats import EmotionStats


class EmployeeActivityData(BaseModel):
    employee_activity_id: UUID = Field(default_factory=uuid4)
    employee_id: UUID | None = None
    branch_id: UUID | None = None
    branch_name: str | None = None
    floor_id: UUID | None = None
    employee_name: str | None = None
    emotion_stats: EmotionStats = Field(default_factory=EmotionStats)
    stream_id: UUID | None = None
    stream_name: str | None = None
    activity_type: str | None = None
    track_id: int | None = None
    timedata: TimeData = Field(default_factory=TimeData)


    def to_json(self):
        return {
            "employee_activity_id": str(self.employee_activity_id),
            "employee_id": str(self.employee_id) if self.employee_id else None,
            "branch_id": str(self.branch_id) if self.branch_id else None,
            "branch_name": self.branch_name if self.branch_name else "Unknown",
            "stream_name": self.stream_name if self.stream_name else "Unknown",
            "floor_id": str(self.floor_id) if self.floor_id else None,
            "employee_name": self.employee_name if self.employee_name else "Unknown",
            "average_emotion": "Happy" if self.emotion_stats.happy > self.emotion_stats.neutral or self.emotion_stats.happy > self.emotion_stats.sad else "Neutral",
            "happy_stats": f"{self.emotion_stats.happy}%" if self.emotion_stats and self.emotion_stats.happy is not None else None,
            "sad_stats": f"{self.emotion_stats.sad}%" if self.emotion_stats and self.emotion_stats.sad is not None else None,
            "neutral_stats": f"{self.emotion_stats.neutral}%" if self.emotion_stats and self.emotion_stats.neutral is not None else None,
            "stream_id": str(self.stream_id) if self.stream_id else None,
            "activity_type": self.activity_type,
            "track_id": self.track_id,
            "start_time": self.timedata.start_time.isoformat() if self.timedata.start_time else None,
            "end_time": self.timedata.end_time.isoformat() if self.timedata.end_time else None,
            "present_time": self.timedata.presence_time if self.timedata.presence_time else 0,
            "absence_time": self.timedata.absence_time if self.timedata.absence_time else 0,
            "date": self.timedata.date_of_activity.isoformat() if self.timedata.date_of_activity else None,
            "created_at": self.timedata.created_at.isoformat() if self.timedata.created_at else None,
            "modified_at": self.timedata.modified_at.isoformat() if self.timedata.modified_at else None,
        }
