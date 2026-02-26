from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Event
from domain.repository import EventDataRepository

@define
class AddEvent:
    event_data_repository: EventDataRepository = field(
        validator=validators.instance_of(EventDataRepository)
    )
    def invoke(self, event_id: str, task_id: str, event_type : str) -> Event:
        event = Event(
            event_id=UUID(event_id),
            task_id=UUID(task_id),
            event_type=event_type,
            status="processing",
            created_at=timezone.now(),
            updated_at=None
        )
        return self.event_data_repository.add_event_data(event=event)