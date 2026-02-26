from attrs import define, field, validators
from domain.model import Employee
from domain.repository import EmployeeRepository
from uuid import UUID

@define
class GetEmployeeById:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )
    
    def invoke(
        self,
        identifier: UUID,
    ) -> Employee:
        employee = self.employee_repository.get_employee_by_id(identifier)
        return employee