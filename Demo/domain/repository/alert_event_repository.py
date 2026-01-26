from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from domain.model import AlertEvent


class AlertEventRepository(ABC):
    @abstractmethod
    def add_alert_event(self, alert_event: AlertEvent) -> AlertEvent:
        raise NotImplementedError

    @abstractmethod
    def get_alert_events(
        self,
        status: Optional[str] = None,
        branch_id: Optional[str] = None,
        stream_id: Optional[str] = None,
    ) -> List[AlertEvent]:
        raise NotImplementedError

    @abstractmethod
    def resolve_alert_event(self, alert_event_id: UUID, employee_id: Optional[UUID] = None) -> AlertEvent:
        raise NotImplementedError
