from attrs import define, field, validators
from domain.model import Role
from domain.repository import AccessControlRepository
from uuid import UUID

@define
class GetRoleById:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )
    def invoke(
        self,
        role_id: str,
    ) -> Role:
        role = self.access_control_repository.get_role_by_id(role_id=role_id)
        return role