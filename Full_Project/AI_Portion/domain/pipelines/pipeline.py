from abc import ABC, abstractmethod
from ..model import StreamActivityData


class Pipeline(ABC):
    @abstractmethod
    def process(self, stream_data: StreamActivityData ,task_id: str | None = None ) -> StreamActivityData:
        raise NotImplementedError("Implement process method")