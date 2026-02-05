from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime
from domain.model import Employee, TimeData, UserData, User
from domain.repository import EmployeeRepository

@define
class AddEmployee:
    employee_repository: EmployeeRepository = field(
        validator=validators.instance_of(EmployeeRepository)
    )

    def invoke(
            self, 
            first_name: str,
            last_name: str,
            email: str,
            phone_number: str | None = None,
            branch: str | None = None,
            department: str | None = None,
            role: str | None = None,
            user_created: UUID | None = None
        ) -> dict:
        employee = Employee(
            identifier=uuid4(),
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
            branch=branch,
            department=department,
            role=role,
            time_data=TimeData(
                created_at=datetime.now(),
                modified_at=None
            ),
            user_data=UserData(
                user_created=User(user_id=user_created),
                user_modified=None
            )
        )
        return self.employee_repository.add_employee(employee=employee)