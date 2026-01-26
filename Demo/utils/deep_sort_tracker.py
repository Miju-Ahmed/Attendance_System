"""
DeepSORT Tracker for Face Tracking
===================================
Lightweight DeepSORT implementation for tracking faces using:
- Kalman Filter for motion prediction
- Cosine distance on face embeddings for appearance matching
- Hungarian algorithm for optimal assignment

Based on: "Simple Online and Realtime Tracking with a Deep Association Metric"
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from typing import List, Tuple, Optional


class KalmanFilter:
    """
    Kalman Filter for tracking 2D bounding boxes.
    State: [x, y, w, h, vx, vy, vw, vh] (center_x, center_y, width, height, velocities)
    """
    
    def __init__(self):
        # State transition matrix (constant velocity model)
        self.dt = 1.0
        self.F = np.eye(8)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        self.F[3, 7] = self.dt
        
        # Measurement matrix (we only observe position and size, not velocity)
        self.H = np.eye(4, 8)
        
        # Process noise covariance
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocities
        
        # Measurement noise covariance
        self.R = np.eye(4) * 10.0
        
        # State covariance
        self.P = np.eye(8) * 1000.0
        
        # State vector
        self.x = np.zeros(8)
    
    def init_from_bbox(self, bbox: np.ndarray):
        """Initialize state from bounding box [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.P = np.eye(8) * 1000.0
    
    def predict(self) -> np.ndarray:
        """Predict next state"""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox: np.ndarray):
        """Update state with measurement"""
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        z = np.array([cx, cy, w, h])
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def get_bbox(self) -> np.ndarray:
        """Get current bounding box [x1, y1, x2, y2]"""
        cx, cy, w, h = self.x[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2])
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point (cx, cy)"""
        return self.x[0], self.x[1]


class Track:
    """Represents a single tracked object"""
    
    def __init__(self, track_id: int, bbox: np.ndarray, embedding: np.ndarray, 
                 max_age: int = 30, embedding_buffer_size: int = 10):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.init_from_bbox(bbox)
        
        self.bbox = bbox
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_age = max_age
        
        # Embedding buffer for robust matching
        self.embedding_buffer = deque(maxlen=embedding_buffer_size)
        if embedding is not None:
            self.embedding_buffer.append(embedding.copy())
        
        # State
        self.state = "Tentative"  # Tentative, Confirmed, Deleted
        self.min_hits = 3  # Minimum hits to confirm track
    
    def predict(self):
        """Predict next position"""
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox
    
    def update(self, bbox: np.ndarray, embedding: Optional[np.ndarray] = None):
        """Update track with new detection"""
        self.kf.update(bbox)
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        
        # Update embedding buffer
        if embedding is not None:
            self.embedding_buffer.append(embedding.copy())
        
        # Confirm track if enough hits
        if self.state == "Tentative" and self.hits >= self.min_hits:
            self.state = "Confirmed"
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from buffer"""
        if len(self.embedding_buffer) == 0:
            return None
        
        avg_emb = np.mean(list(self.embedding_buffer), axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        return avg_emb
    
    def mark_missed(self):
        """Mark track as missed (no detection matched)"""
        self.time_since_update += 1
        if self.time_since_update > self.max_age:
            self.state = "Deleted"
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.state == "Confirmed"
    
    def is_deleted(self) -> bool:
        """Check if track should be deleted"""
        return self.state == "Deleted"


class DeepSORTTracker:
    """
    DeepSORT tracker for face tracking using embeddings.
    
    Features:
    - Kalman filtering for motion prediction
    - Embedding-based appearance matching
    - Hungarian algorithm for optimal assignment
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 embedding_distance_threshold: float = 0.6,
                 embedding_weight: float = 0.7):
        """
        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching
            embedding_distance_threshold: Cosine distance threshold for embeddings
            embedding_weight: Weight for embedding distance (vs IoU) in matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_distance_threshold = embedding_distance_threshold
        self.embedding_weight = embedding_weight
        
        self.tracks: List[Track] = []
        self.next_id = 1
    
    def update(self, detections: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, embedding) tuples
                       bbox: [x1, y1, x2, y2, confidence]
                       embedding: face embedding vector (optional)
        
        Returns:
            List of (track_id, bbox) for confirmed tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._match(detections)
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            bbox, embedding = detections[det_idx]
            self.tracks[trk_idx].update(bbox, embedding)
        
        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, embedding = detections[det_idx]
            new_track = Track(self.next_id, bbox, embedding, self.max_age)
            new_track.min_hits = self.min_hits
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.is_confirmed():
                results.append((track.track_id, track.bbox))
        
        return results
    
    def _match(self, detections: List[Tuple[np.ndarray, Optional[np.ndarray]]]) -> Tuple[List, List, List]:
        """
        Match detections to tracks using IoU and embedding distance.
        
        Returns:
            matched: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for d, (det_bbox, det_embedding) in enumerate(detections):
            for t, track in enumerate(self.tracks):
                # IoU distance
                iou = self._iou(det_bbox, track.bbox)
                iou_dist = 1.0 - iou
                
                # Embedding distance (cosine distance)
                emb_dist = 1.0  # Default: maximum distance
                if det_embedding is not None:
                    track_emb = track.get_average_embedding()
                    if track_emb is not None:
                        # Cosine distance = 1 - cosine similarity
                        det_emb_norm = det_embedding / (np.linalg.norm(det_embedding) + 1e-8)
                        track_emb_norm = track_emb / (np.linalg.norm(track_emb) + 1e-8)
                        cosine_sim = np.dot(det_emb_norm, track_emb_norm)
                        emb_dist = 1.0 - cosine_sim
                
                # Combined cost (weighted average)
                cost_matrix[d, t] = (self.embedding_weight * emb_dist + 
                                    (1 - self.embedding_weight) * iou_dist)
        
        # Hungarian algorithm for optimal assignment
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        
        for d, t in zip(det_indices, trk_indices):
            cost = cost_matrix[d, t]
            
            # Accept match if cost is below threshold
            if cost < 0.7:  # Threshold for combined cost
                matched.append((d, t))
                if d in unmatched_dets:
                    unmatched_dets.remove(d)
                if t in unmatched_trks:
                    unmatched_trks.remove(t)
        
        return matched, unmatched_dets, unmatched_trks
    
    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def get_all_tracks(self) -> List[Tuple[int, np.ndarray, str]]:
        """
        Get all tracks (including tentative ones) for debugging.
        
        Returns:
            List of (track_id, bbox, state)
        """
        return [(t.track_id, t.bbox, t.state) for t in self.tracks]
