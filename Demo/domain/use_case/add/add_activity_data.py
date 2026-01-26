from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID, uuid4
from domain.model import ActivityData, ActivityType
from domain.repository import ActivityDataRepository

@define
class AddActivity:
    activity_repository: ActivityDataRepository = field(
        validator=validators.instance_of(ActivityDataRepository)
    )


    def invoke(self, user: UUID, activity_type: ActivityType) -> dict:
        activity = ActivityData(
            activity_id=uuid4(),
            activity_type=activity_type,
            user_id=user,
            activity_datetime=timezone.now(),
            activity_details=f"User {user} {activity_type.value} stream created at {timezone.now()}"
        )
        return self.activity_repository.add_activity(activity_data=activity)