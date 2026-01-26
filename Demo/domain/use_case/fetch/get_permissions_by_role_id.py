from attrs import define, field, validators
from domain.model import AccessPermission
from domain.repository import AccessControlRepository

@define
class GetPermissionsByRoleId:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )
    
    def invoke(
        self,
        role_id: str,
    ) -> AccessPermission:
        permissions = self.access_control_repository.get_role_permissions(role_id=role_id)
        return permissions