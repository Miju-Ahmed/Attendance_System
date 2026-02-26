from attrs import define, field, validators
from domain.model import Employee
from domain.repository import EmployeeRepository
from uuid import UUID

@define
class GetEmployeesByBranchId:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )
    
    def invoke(
        self,
        branch_id: UUID,
    ) -> list[Employee]:
        employees = self.employee_repository.get_employees_by_branch_id(branch_id)
        return employees
