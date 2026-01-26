from attrs import define, field, validators
from domain.model import Employee
from domain.repository import EmployeeRepository
from typing import List

@define
class GetAllEmployees:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )
    
    def invoke(self) -> List[Employee]:
        employees = self.employee_repository.get_all_employees()
        return employees