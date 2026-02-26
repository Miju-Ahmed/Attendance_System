from datetime import date
from attrs import define, field, validators
from domain.model import Role
from domain.repository import AccessControlRepository

from utils import get_logger

logger = get_logger(__name__)


@define
class GetAllRoles:
    access_control_repository: AccessControlRepository = field(
        validator=validators.instance_of(AccessControlRepository)
    )

    def invoke(
        self,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> list[Role]:
        try:
            roles = self.access_control_repository.get_all_roles(
                sort_by=sort_by,
                order=order,
            )
            logger.info(f"Retrieved {len(roles)} roles")
            return roles
        except Exception as e:
            logger.error(f"Error retrieving roles: {e}")
            return []