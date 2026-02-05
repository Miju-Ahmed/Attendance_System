from attrs import define, field, validators
from domain.repository import EmployeeRepository
from uuid import UUID

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteEmployeeById:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )

    def invoke(self, identifier: UUID) -> dict:
        return self.employee_repository.delete_employee_by_id(identifier)