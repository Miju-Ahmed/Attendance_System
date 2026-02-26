from uuid import UUID
from datetime import date
from attrs import define
from typing import Optional

@define
class Alert:
    date: date 
    employee_id: str | None = None
    employee_name: str | None = None
    camera_no: str | None = None
    branch_id: str | None = None
    activity_type: str = "unknown"
    description: str = "No description provided"


    def to_json(self) -> dict:        
        return {
            "date": self.date.isoformat(),
            "employee_id": str(self.employee_id) if self.employee_id else None,
            "employee_name": self.employee_name,
            "camera_no": self.camera_no,
            "branch_id": str(self.branch_id) if self.branch_id else None,
            "activity_type": self.activity_type,
            "description": self.description
        }