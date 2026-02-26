from datetime import date
from attrs import define, field, validators
from domain.model import EmployeeActivityData
from domain.repository import EmployeeActivityDataRepository

from utils import get_logger

logger = get_logger(__name__)


@define
class GetEmotionSummary:
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
            employee_emotion_summary = self.employee_activity_data_repository.get_emotion_summary(
                branch_id=branch_id,
                employee_id=employee_id,
                stream_id=stream_id,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"Retrieved {len(employee_emotion_summary)} employee activity data")

            return employee_emotion_summary
        except Exception as e:
            logger.error(f"Error retrieving employee activity data: {e}")
            return []