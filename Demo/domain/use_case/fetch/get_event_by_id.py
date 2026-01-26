from attrs import define, field, validators
from domain.model import Event
from domain.repository import EventDataRepository


@define
class GetEventById:
    event_data_repository: EventDataRepository = field(
        validator=validators.instance_of(EventDataRepository)
    )

    def invoke(
        self,
        event_id: str,
    ) -> Event:
        return self.event_data_repository.get_event_by_id(event_id=event_id)
