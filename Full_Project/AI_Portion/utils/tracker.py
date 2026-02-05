import math
from typing import List, Dict, Tuple
from utils import get_logger
from domain.model import BoundingBox, DetectionResult

logger = get_logger(__name__)

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect: list[BoundingBox]) -> list[BoundingBox]:
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append(
                        BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            track_id=id,
                        )
                    )
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append(
                    BoundingBox(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        track_id=self.id_count,
                    )
                )
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = (
                obj_bb_id.x,
                obj_bb_id.y,
                obj_bb_id.width,
                obj_bb_id.height,
                obj_bb_id.track_id,
            )
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
class FaceTracker:
    def __init__(self, max_distance: float = 25.0, max_age: int = 30):
        self.center_points: Dict[int, Tuple[int, int]] = {}
        self.id_count = 0
        self.age_count: Dict[int, int] = {}
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        tracked_results = []
        unmatched_detections = detection_results.copy()
        unmatched_trackers = list(self.center_points.keys())

        # Match existing trackers with new detections
        for id, pt in list(self.center_points.items()):
            best_match = None
            min_dist = float('inf')
            for result in unmatched_detections:
                bbox = result.bounding_boxes
                keypoints = result.key_points
                x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
                cx = x + w // 2
                cy = y + h // 2
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_match = result

            if best_match:
                bbox = best_match.bounding_boxes
                keypoints = best_match.key_points
                x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
                cx = x + w // 2
                cy = y + h // 2
                self.center_points[id] = (cx, cy)
                self.age_count[id] = 0
                tracked_results.append(
                    DetectionResult(
                        bounding_boxes=BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            track_id=id,
                        ),
                        key_points=keypoints,
                    )
                )
                unmatched_detections.remove(best_match)
                unmatched_trackers.remove(id)

        # Create new trackers for unmatched detections
        for result in unmatched_detections:
            bbox = result.bounding_boxes
            keypoints = result.key_points
            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
            cx = x + w // 2
            cy = y + h // 2
            self.center_points[self.id_count] = (cx, cy)
            self.age_count[self.id_count] = 0
            tracked_results.append(
                DetectionResult(
                    bounding_boxes=BoundingBox(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        track_id=self.id_count,
                    ),
                    key_points=keypoints,
                )
            )
            self.id_count += 1

        # Increment age for unmatched trackers and remove old ones
        for id in unmatched_trackers:
            self.age_count[id] += 1
            if self.age_count[id] > self.max_age:
                del self.center_points[id]
                del self.age_count[id]

        return tracked_results
