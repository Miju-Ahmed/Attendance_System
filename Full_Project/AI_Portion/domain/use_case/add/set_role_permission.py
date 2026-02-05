from attrs import define, field, validators
from domain.model import AccessPermission
from domain.repository import AccessControlRepository

from utils import get_logger
logger = get_logger(__name__)

@define
class SetRolePermission:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )
    def invoke(self, permissions: AccessPermission) -> dict:
        try:
            return self.access_control_repository.set_role_permissions(permissions=permissions)
        except Exception as e:
            logger.error(f"Error in setting role permissions: {e}")
            return {"error": "Failed to set role permissions"}
