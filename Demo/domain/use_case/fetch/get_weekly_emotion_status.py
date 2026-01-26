from datetime import date
from attrs import define, field, validators
from domain.repository import EmployeeActivityDataRepository

from utils import get_logger

logger = get_logger(__name__)


@define
class GetWeeklyEmotionStatus:
    employee_activity_data_repository: EmployeeActivityDataRepository = field(
        validator=validators.instance_of(EmployeeActivityDataRepository)
    )

    def invoke(
        self,
        branch_id: str | None = None,
        employee_id: str | None = None,
        stream_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        try:
            employee_emotion_status = self.employee_activity_data_repository.get_weekly_emotion_status(
                branch_id=branch_id,
                employee_id=employee_id,
                stream_id=stream_id,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"Retrieved {len(employee_emotion_status)} employee activity data")

            return employee_emotion_status
        except Exception as e:
            logger.error(f"Error retrieving employee activity data: {e}")
            return []