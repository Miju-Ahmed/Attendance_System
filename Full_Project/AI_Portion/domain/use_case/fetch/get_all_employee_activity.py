from datetime import date
from attrs import define, field, validators
from domain.model import EmployeeActivityData
from domain.repository import EmployeeActivityDataRepository
from typing import Optional


from utils import get_logger

logger = get_logger(__name__)


@define
class GetAllEmployeeActivity:
    employee_activity_data_repository: EmployeeActivityDataRepository = field(
        validator=validators.instance_of(EmployeeActivityDataRepository)
    )

    def invoke(
        self,
        sort_by: str = "created_at",
        order: str = "desc",
        branch_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[EmployeeActivityData]:
        try:
            employee_activity_data = self.employee_activity_data_repository.get_all_employee_activity_data(
                sort_by=sort_by,
                order=order,
                branch_id=branch_id,
                employee_id=employee_id,
                stream_id=stream_id,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"Retrieved {len(employee_activity_data)} employee activity data")
            return employee_activity_data
        except Exception as e:
            logger.error(f"Error retrieving employee activity data: {e}")
            return []