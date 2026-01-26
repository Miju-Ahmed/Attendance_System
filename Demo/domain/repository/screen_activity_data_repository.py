from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional
from ..model import ScreenActivityData

class ScreenActivityDataRepository(ABC):
    @abstractmethod
    def add_screen_activity(self, screen_activity: ScreenActivityData) -> ScreenActivityData:
        raise NotImplementedError("Implement add_screen_activity method")

    @abstractmethod
    def get_screen_activity_by_id(self, screen_activity_id: str) -> ScreenActivityData:
        raise NotImplementedError("Implement get_screen_activity_by_id method")

    @abstractmethod
    def delete_screen_activity(self, screen_activity_id: str) -> bool:
        raise NotImplementedError("Implement delete_screen_activity method")
