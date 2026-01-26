from attrs import define, field, validators
from domain.model import Employee
from domain.repository import StreamRepository
from uuid import UUID

@define
class GetStreamsByBranchId:
    stream_repository: StreamRepository = field(
        validator=validators.instance_of(StreamRepository)
    )
    
    def invoke(
        self,
        BranchId: str,
    ) -> Employee:
        streams = self.stream_repository.get_streams_by_branch_id(BranchId)
        return streams