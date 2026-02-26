from uuid import UUID, uuid4
from django.http import JsonResponse
from pydantic import BaseModel, Field
from datetime import datetime
from .activity_type import ActivityType


class ActivityData(BaseModel):
    activity_id: UUID = Field(default_factory=uuid4)
    activity_type: ActivityType = Field(default=ActivityType.MODIFY)
    activity_datetime: datetime = Field(default_factory=datetime.now)
    activity_details: str | None = None
    stream_id: UUID | None = None
    user_id: UUID | None = None
    user_name: str | None = None

    @property
    def formatted_creation_datetime(self) -> str:
        return self.activity_datetime.strftime("%m/%d/%Y %H:%M:%S")
    
    def to_json(self) -> dict:
        return {
            # fmt: off
            "activity_id": self.activity_id,
            "activity_type": self.activity_type,
            "activity_datetime": self.formatted_creation_datetime,
            "activity_details": self.activity_details,
            "stream_id": self.stream_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            # fmt: on
        }
    
    def to_response(self) -> JsonResponse:
        return JsonResponse(self.to_json())