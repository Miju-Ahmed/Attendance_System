from abc import ABC, abstractmethod

import numpy as np


class HumanTracking(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray) -> list:
        """Return tracked human bounding boxes for a given frame."""

