from environs import Env

from domain.model import StreamActivityData, StreamStatus
from domain.factory.pipeline_factory import PipelineFactory
from utils import get_logger


logger = get_logger(__name__)


class ProcessStream:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        env = Env()
        env.read_env()
        self.__is_debugging = env.bool("DEBUG", default=False)
        self.__pipeline_factory = PipelineFactory(is_debugging=self.__is_debugging)

    def invoke(self, stream_data: StreamActivityData, task_id: str | None = None) -> StreamActivityData:
        logger.info("Using pipeline version: %s", stream_data.pipeline)
        pipeline = self.__pipeline_factory.get_pipeline(version=stream_data.pipeline)

        if not stream_data.processed_uri:
            logger.error("Processed URI missing for stream %s", stream_data.stream_id)
            stream_data.success = False
            stream_data.errors = "Processed URI missing"
            stream_data.status = StreamStatus.FAILED
            return stream_data

        processed_stream_data = pipeline.process(stream_data=stream_data, task_id=task_id)
        if processed_stream_data is None:
            logger.warning(
                "Pipeline %s returned no data; falling back to provided stream activity",
                stream_data.pipeline,
            )
            processed_stream_data = stream_data

        if (
            processed_stream_data.success
            and processed_stream_data.status == StreamStatus.PROCESSING
        ):
            processed_stream_data.status = StreamStatus.PROCESSED
        elif (
            not processed_stream_data.success
            and processed_stream_data.status == StreamStatus.PROCESSING
        ):
            processed_stream_data.status = StreamStatus.FAILED

        return processed_stream_data
