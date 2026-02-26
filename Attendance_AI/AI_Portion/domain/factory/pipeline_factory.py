from di.service import Service
from di.repository import Repository
from ..pipelines import Pipeline, PipelineV1 , PipelineV2, PipelineV3, PipelineV4, UnknownAlertPipeline


class PipelineFactory:
    def __init__(
        self,
        is_debugging: bool = False,
    ):
        service = Service()
        repository = Repository()

        self._pipelines = {
            "v1": PipelineV1(
                face_detection=service.human_face_detection,
                face_embedding_generator=service.human_face_embedding,
                human_detection=service.human_detection,
                # emotion_detector=service.human_emotion_recognition,
                face_vector_db=repository.face_vector_repository,
                employee_db=repository.employee_repository,
                region_db=repository.region_repository,
                activity_db=repository.employee_activity_data_repository,
                is_debugging=is_debugging,
                
            ),
            "v2": PipelineV2(
                face_detection=service.human_face_detection,
                face_embedding_generator=service.human_face_embedding,
                # human_detection=service.human_detection,
                human_emotion_recognition=service.human_emotion_recognition,
                face_vector_db=repository.face_vector_repository,
                stream_db=repository.stream_repository,
                recorded_stream_db=repository.recorded_stream_repository,
                employee_db=repository.employee_repository,
                region_db=repository.region_repository,
                activity_db=repository.employee_activity_data_repository,
                is_debugging=is_debugging,

            ),
            "v3": PipelineV3(
                face_detection=service.human_face_detection,
                face_embedding_generator=service.human_face_embedding,
                face_vector_db=repository.face_vector_repository,
                employee_db=repository.employee_repository,
                stream_db=repository.stream_repository,
                recorded_stream_db=repository.recorded_stream_repository,
                activity_db=repository.employee_activity_data_repository,
                alert_event_db=repository.alert_event_repository,
                is_debugging=is_debugging,
            ),
            "v4": PipelineV4(
                face_detection=service.human_face_detection,
                face_embedding_generator=service.human_face_embedding,
                human_detection=service.human_detection,
                id_card_detection=service.id_card_detection,
                face_vector_db=repository.face_vector_repository,
                employee_db=repository.employee_repository,
                is_debugging=is_debugging,
            ),
            
            "unknown": UnknownAlertPipeline(
                face_detection=service.human_face_detection,
                face_embedding_generator=service.human_face_embedding,
                human_tracking=service.human_tracking,
                face_vector_db=repository.face_vector_repository,
                employee_db=repository.employee_repository,
                stream_db=repository.stream_repository,
                recorded_stream_db=repository.recorded_stream_repository,
                alert_event_db=repository.alert_event_repository,
                is_debugging=is_debugging,

            ),
        }

    def get_pipeline(self, version: str = "v1") -> Pipeline:
        if version not in self._pipelines.keys():
            return self._pipelines["v1"]
        return self._pipelines[version]
