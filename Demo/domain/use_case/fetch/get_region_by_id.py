from attrs import define, field, validators
from domain.model import Region
from domain.repository import RegionRepository
from uuid import UUID

@define
class GetRegionById:
    region_repository: RegionRepository = field(
        validator=validators.instance_of(RegionRepository)
    )
    
    def invoke(
        self,
        identifier: UUID,
    ) -> Region:
        region = self.region_repository.get_region(region_id=identifier)
        return region