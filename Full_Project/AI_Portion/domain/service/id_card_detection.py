from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from domain.model import BoundingBox


class IdCardDetection(ABC):
    """Detects ID cards and visual security marks (e.g., logos) inside a frame."""

    @property
    @abstractmethod
    def id_class_index(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def logo_class_index(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[BoundingBox]:
        raise NotImplementedError
