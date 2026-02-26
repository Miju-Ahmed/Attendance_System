from abc import ABC, abstractmethod
from uuid import UUID
from datetime import datetime

from ..model import Event

class EventDataRepository(ABC):
    @abstractmethod
    def add_event_data(self, event: Event) -> Event:
        raise NotImplementedError("Implement add_event_data method")
    
    @abstractmethod
    def update_event_data(self, event : Event) -> Event:
        raise NotImplementedError("Implement update_event_date method")
    
    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Event:
        raise NotImplementedError("Implement get_event_by_id method")