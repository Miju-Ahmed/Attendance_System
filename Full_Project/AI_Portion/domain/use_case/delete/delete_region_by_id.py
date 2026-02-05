from attrs import define, field, validators
from domain.repository import RegionRepository
from uuid import UUID

from utils import get_logger
logger = get_logger(__name__)

@define
class DeleteRegionById:
    region_repository: RegionRepository = field(
        validator=validators.instance_of(RegionRepository)
    )

    def invoke(self, identifier: UUID) -> dict:
        return self.region_repository.delete_region_by_id(identifier)