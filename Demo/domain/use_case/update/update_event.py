from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Event
from domain.repository import EventDataRepository


@define
class UpdateEvent:
    event_data_repository: EventDataRepository = field(
        validator=validators.instance_of(EventDataRepository)
    )

    def invoke(self, event: Event, status: str) -> Event:
        event.status = status
        event.timedata.modified_at = timezone.now()
        return self.event_data_repository.update_event_data(event=event)
