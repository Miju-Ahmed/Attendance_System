from abc import ABC, abstractmethod

from ..model import Role, AccessPermission

class AccessControlRepository(ABC):
    @abstractmethod
    def get_all_roles(self, sort_by: str, order: str) -> list[Role]:
        raise NotImplementedError("Implement get_all_roles method")
    
    @abstractmethod
    def get_role_by_id(self, role_id: str) -> Role:
        raise NotImplementedError("Implement get_role_by_id method")
    
    @abstractmethod
    def add_role(self, role: Role) -> dict:
        raise NotImplementedError("Implement add_role method")
    
    @abstractmethod
    def update_role(self, role: Role) -> dict:
        raise NotImplementedError("Implement update_role method")
    
    @abstractmethod
    def delete_role(self, role_id: str) -> dict:
        raise NotImplementedError("Implement delete_role method")
    
    @abstractmethod
    def get_role_permissions(self, role_id: str) -> AccessPermission:
        raise NotImplementedError("Implement get_role_permissions method")
    
    @abstractmethod
    def set_role_permissions(self, permissions: AccessPermission) -> dict:
        raise NotImplementedError("Implement set_role_permissions method")
