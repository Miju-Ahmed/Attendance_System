from attrs import define, field, validators
from domain.repository import AccessControlRepository
from uuid import UUID

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteRoleById:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )

    def invoke(self, role_id: str) -> dict:
        return self.access_control_repository.delete_role(role_id=role_id)