from abc import ABC, abstractmethod
from datetime import date
from uuid import UUID
from typing import Optional
from ..model import EmployeeActivityData, Alert

class EmployeeActivityDataRepository(ABC):
    @abstractmethod
    def add_employee_activity_data(self, employee_activity_data: EmployeeActivityData) -> EmployeeActivityData:
        raise NotImplementedError("Implement add_employee_activity method")

    @abstractmethod
    def update_employee_activity_data(self, employee_activity_data: EmployeeActivityData) -> EmployeeActivityData:
        raise NotImplementedError("Implement update_employee_activity method")
    
    @abstractmethod
    def get_all_employee_activity_data(
        self,
        sort_by: str = "created_at",
        order: str = "desc",
        branch_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_all_employee_activity_data method")

    @abstractmethod
    def get_employee_activity_data_by_id(self, employee_activity_data_id: str) -> EmployeeActivityData:
        raise NotImplementedError("Implement get_employee_activity_by_id method")

    @abstractmethod
    def delete_employee_activity_data(self, employee_activity_data_id: str) -> bool:
        raise NotImplementedError("Implement delete_employee_activity method")

    @abstractmethod
    def get_weekly_presence_status(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> dict:
        raise NotImplementedError("Implement get_weekly_presence_status method")
    
    @abstractmethod
    def get_active_vs_idle_time(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_active_vs_idle_time method")
    
    @abstractmethod
    def get_mood_trends(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_mood_trends method")
    
    @abstractmethod
    def get_weekly_emotion_status(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_weekly_emotion_status method")
    
    @abstractmethod
    def get_emotion_summary(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_emotion_summary method")
    
    @abstractmethod
    def get_daily_peak_productivity(self, branch_id: Optional[str] = None, employee_id: Optional[str] = None, stream_id: Optional[str] = None, productivity_date: Optional[date] = None) -> list[EmployeeActivityData]:
        raise NotImplementedError("Implement get_daily_peak_productivity method")
    
    @abstractmethod
    def get_alerts(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        employee_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        activity_type: Optional[str] = None
    ) -> list[Alert]:
        raise NotImplementedError("Implement get_alerts method")