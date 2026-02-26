from attrs import define, field, validators
from uuid import UUID, uuid4
from datetime import datetime
from domain.model import Region,  UserData , TimeData , User
from domain.repository import RegionRepository

@define
class AddRegion:
    region_repository: RegionRepository = field(
        validator=validators.instance_of(RegionRepository)
    )

    def invoke(
            self, 
            name: str,
            description: str,
            stream_id: UUID,
            bounding_box: str,
            user_created: UUID | None
            
        ) -> dict:
        region = Region(
            identifier=uuid4(),
            name=name,
            description=description,
            stream_id=stream_id,
            bounding_box=bounding_box,
            timedata=TimeData(
                created_at=datetime.now(),
                modified_at=None
            ),
            user_data=UserData(
                user_created=User(user_id=user_created),
                user_modified=None
            )
        )
        return self.region_repository.add_region(region=region)