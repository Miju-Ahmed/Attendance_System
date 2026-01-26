from abc import ABC, abstractmethod

from PIL.Image import Image
from domain.model import BoundingBox


class HumanDetection(ABC):
    @abstractmethod
    def predict(self, image: Image)-> list[BoundingBox]:
        raise NotImplementedError("Implement predict method")