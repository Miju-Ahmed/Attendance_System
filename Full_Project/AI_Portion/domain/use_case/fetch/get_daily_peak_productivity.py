from datetime import date
from attrs import define, field, validators
from domain.model import EmployeeActivityData
from domain.repository import EmployeeActivityDataRepository

from utils import get_logger

logger = get_logger(__name__)


@define
class GetDailyPeakProductivity:
    employee_activity_data_repository: EmployeeActivityDataRepository = field(
        validator=validators.instance_of(EmployeeActivityDataRepository)
    )

    def invoke(
        self,
        branch_id: str | None = None,
        employee_id: str | None = None,
        stream_id: str | None = None,
        productivity_date: date | None = None,
    ):
        try:
            employee_peak_productivity = self.employee_activity_data_repository.get_daily_peak_productivity(
                branch_id=branch_id,
                employee_id=employee_id,
                stream_id=stream_id,
                productivity_date=productivity_date,
            )
            logger.info(f"Retrieved {len(employee_peak_productivity)} employee activity data")

            return employee_peak_productivity
        except Exception as e:
            logger.error(f"Error retrieving employee activity data: {e}")
            return []