from django.utils import timezone
from attrs import define, field, validators
from domain.model import Role
from domain.repository import AccessControlRepository

@define
class AddRole:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )
    def invoke(self, role: Role) -> dict:
        return self.access_control_repository.add_role(role=role)
