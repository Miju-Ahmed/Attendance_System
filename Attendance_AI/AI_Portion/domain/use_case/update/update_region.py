from django.utils import timezone
from attrs import define, field, validators
from uuid import UUID
from domain.model import Region, TimeData, User, UserData
from domain.repository import RegionRepository

@define
class UpdateRegion:
    region_repository: RegionRepository = field(
        validator=validators.instance_of(RegionRepository)
    )

    def invoke(self, region: Region) -> dict:
        return self.region_repository.update_region(region=region)