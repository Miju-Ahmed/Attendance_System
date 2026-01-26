from pydantic import BaseModel
from .bounding_box import BoundingBox
from typing import List


class DetectionResult(BaseModel):
    bounding_boxes: BoundingBox
    key_points: List[List[float]]
