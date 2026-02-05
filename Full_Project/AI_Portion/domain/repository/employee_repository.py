from abc import ABC, abstractmethod
from uuid import UUID

from ..model import Employee

class EmployeeRepository(ABC):
    @abstractmethod
    def add_employee(self, employee: Employee) -> dict:
        raise NotImplementedError("Implement add_employee method")
    
    @abstractmethod
    def get_employee_by_id(self, identifier: UUID) -> Employee:
        raise NotImplementedError("Implement get_employee_by_id method")
    
    @abstractmethod
    def get_all_employees(self) -> list[Employee]:
        raise NotImplementedError("Implement get_all_employees method")
    
    @abstractmethod
    def delete_employee_by_id(self, identifier: UUID) -> dict:
        raise NotImplementedError("Implement delete_employee_by_id method")

    @abstractmethod
    def update_employee(self, employee: Employee) -> dict:
        raise NotImplementedError("Implement update_employee method")
    
    @abstractmethod
    def get_employees_by_branch_id(self, branch_id: UUID) -> list[Employee]:
        raise NotImplementedError("Implement get_employees_by_branch_id method")
