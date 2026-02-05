from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Employee, TimeData, User, UserData
from domain.repository import EmployeeRepository


@define
class UpdateEmployee:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )
    def invoke(self, employee: Employee, user_modified: str) -> dict:
        
        employee.userdata = UserData(
            user_modified=User(user_id=user_modified) if user_modified is not None else employee.userdata.user_modified
        )
        
        return self.employee_repository.update_employee(employee=employee)