from attrs import define, field, validators
from typing import Optional, List
from datetime import date
from uuid import UUID
from domain.model import Alert
from domain.repository import EmployeeActivityDataRepository

from utils import get_logger

logger = get_logger(__name__)

@define
class GetAlerts:
    employee_activity_data_repository: EmployeeActivityDataRepository = field(
        validator=validators.instance_of(EmployeeActivityDataRepository)
    )

    def invoke(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        employee_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        activity_type: Optional[str] = None
    ) -> List[Alert]:
        try:
            alerts = self.employee_activity_data_repository.get_alerts(
                start_date=start_date,
                end_date=end_date,
                employee_id=employee_id,
                stream_id=stream_id,
                branch_id=branch_id,
                activity_type=activity_type
            )
            
            logger.info(f"Retrieved {len(alerts)} alerts")
            return alerts
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return None
