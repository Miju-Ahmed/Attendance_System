from abc import ABC, abstractmethod

from ..model import ActivityData

class ActivityDataRepository(ABC):
    @abstractmethod
    def add_activity(self, activity_data: ActivityData) -> ActivityData:
        raise NotImplementedError("Implement add_egg_data method")
    @abstractmethod
    def get_activity_by_id(self, activity_id: str) -> ActivityData:
        raise NotImplementedError("Implement get_egg_data_by_id method")
    @abstractmethod
    def delete_activity(self, activity_id: str) -> bool:
        raise NotImplementedError("Implement delete_egg_data method")

    