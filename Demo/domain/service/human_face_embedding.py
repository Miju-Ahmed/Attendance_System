from abc import ABC, abstractmethod

from PIL.Image import Image


class HumanFaceEmbedding(ABC):
    @abstractmethod
    def predict(self, image: Image):
        raise NotImplementedError("Implement predict method")
