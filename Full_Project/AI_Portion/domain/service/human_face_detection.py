from abc import ABC, abstractmethod

from PIL.Image import Image
from domain.model import DetectionResult


class HumanFaceDetection(ABC):
    @abstractmethod
    def predict(self, image: Image)-> list[DetectionResult]:
        raise NotImplementedError("Implement predict method")
