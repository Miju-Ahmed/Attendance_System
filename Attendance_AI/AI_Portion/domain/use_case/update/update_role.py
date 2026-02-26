from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Role, TimeData, User, UserData
from domain.repository import AccessControlRepository

@define
class UpdateRole:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )

    def invoke(self, role: Role) -> dict:
        return self.access_control_repository.update_role(role=role)