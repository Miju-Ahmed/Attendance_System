from abc import ABC, abstractmethod

from PIL.Image import Image


class HumanEmotionRecognition(ABC):
    @abstractmethod
    def predict(self, image: Image):
        raise NotImplementedError("Implement predict method")
