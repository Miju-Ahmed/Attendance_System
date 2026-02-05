from attrs import define, field, validators

from domain.repository import AlertEventRepository


@define
class GetAlertEvents:
    alert_event_repository: AlertEventRepository = field(
        validator=validators.instance_of(AlertEventRepository)
    )

    def invoke(self, status: str | None = None, branch_id: str | None = None, stream_id: str | None = None):
        events = self.alert_event_repository.get_alert_events(
            status=status,
            branch_id=branch_id,
            stream_id=stream_id,
        )
        return [event.to_json() for event in events]
