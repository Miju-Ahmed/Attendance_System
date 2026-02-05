"""
EfficientDet-D0 ONNX Model Wrapper
===================================
Wrapper for EfficientDet-D0 object detection model in ONNX format.
Supports person detection only for attendance system.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class EfficientDet:
    """EfficientDet-D0 ONNX model wrapper for person detection."""

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.3,
        nms_thres: float = 0.5,
        providers: tuple = ("CPUExecutionProvider",),
        input_size: int = 512,
        person_class_id: int = 0,
    ):
        """
        Initialize EfficientDet-D0 model.

        Args:
            model_path: Path to ONNX model file
            conf_thres: Confidence threshold for detections
            nms_thres: NMS IoU threshold
            providers: ONNX Runtime execution providers
            input_size: Model input size (default 512 for D0)
            person_class_id: Class id for "person" in the model label space
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.input_size = input_size

        # COCO person class ID (configurable in case model uses 1-based IDs)
        self.person_class_id = int(person_class_id)

        # Initialize ONNX Runtime session
        try:
            sess_options = ort.SessionOptions()
            # Suppress noisy output-size warnings from dynamic detection counts.
            sess_options.log_severity_level = 3
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=list(providers),
            )
            logging.info(f"EfficientDet-D0 loaded from {model_path}")
            logging.info(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load EfficientDet model: {e}")

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logging.info(f"Model input: {self.input_name}")
        logging.info(f"Model outputs: {self.output_names}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for EfficientDet-D0 inference.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (preprocessed_image, scale_factor, padding)
        """
        h, w = image.shape[:2]

        # Resize with aspect ratio preservation
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Convert to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        return input_tensor, scale, (left, top)

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """
        Post-process model outputs to get final detections.

        Args:
            outputs: Raw model outputs
            scale: Scale factor used in preprocessing
            pad: Padding (left, top) used in preprocessing
            orig_shape: Original image shape (height, width)

        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        detections = []

        try:
            # Debug: Print output shapes
            logging.debug(f"Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                logging.debug(f"Output {i} shape: {out.shape}, dtype: {out.dtype}")
            
            if len(outputs) >= 3:
                # Multiple outputs: [boxes, classes, scores] or [boxes, scores, classes]
                # Try to identify which is which based on shape
                
                # Find boxes (should have shape [..., 4])
                boxes_idx = None
                scores_idx = None
                classes_idx = None
                
                for i, out in enumerate(outputs):
                    if len(out.shape) >= 2 and out.shape[-1] == 4:
                        boxes_idx = i
                
                # Use output names when available
                name_scores = None
                name_classes = None
                name_boxes = None
                for i, name in enumerate(self.output_names):
                    lname = name.lower()
                    if name_boxes is None and any(k in lname for k in ("box", "boxes", "bbox")):
                        name_boxes = i
                    if name_scores is None and any(k in lname for k in ("score", "scores", "confidence", "prob")):
                        name_scores = i
                    if name_classes is None and any(k in lname for k in ("class", "classes", "label", "labels", "id")):
                        name_classes = i

                if name_boxes is not None:
                    boxes_idx = name_boxes
                if name_scores is not None:
                    scores_idx = name_scores
                if name_classes is not None:
                    classes_idx = name_classes

                # Heuristics based on value ranges and dtypes
                remaining = [i for i in range(len(outputs)) if i != boxes_idx]
                for i in remaining:
                    if i == scores_idx or i == classes_idx:
                        continue
                    arr = outputs[i].flatten()
                    if arr.size == 0:
                        continue
                    arr_min = float(arr.min())
                    arr_max = float(arr.max())
                    is_int_like = np.allclose(arr, np.round(arr), atol=1e-3)
                    if classes_idx is None and (arr.dtype in (np.int32, np.int64) or (is_int_like and arr_max > 1)):
                        classes_idx = i
                        continue
                    if scores_idx is None and arr_min >= 0.0 and arr_max <= 1.0 + 1e-3:
                        scores_idx = i

                # Fallback: assume remaining order
                if boxes_idx is None:
                    boxes_idx = 0
                if scores_idx is None or classes_idx is None:
                    candidates = [i for i in range(len(outputs)) if i != boxes_idx]
                    if len(candidates) >= 2:
                        if scores_idx is None:
                            scores_idx = candidates[0]
                        if classes_idx is None:
                            classes_idx = candidates[1]

                logging.debug(f"Detected indices - boxes: {boxes_idx}, scores: {scores_idx}, classes: {classes_idx}")
                
                # Extract arrays
                boxes = outputs[boxes_idx]
                scores = outputs[scores_idx] if scores_idx is not None else None
                classes = outputs[classes_idx] if classes_idx is not None else None
                
                # Flatten to 1D if needed
                if boxes.ndim > 2:
                    boxes = boxes.reshape(-1, 4)
                elif boxes.ndim == 2 and boxes.shape[0] == 1:
                    boxes = boxes[0].reshape(-1, 4)
                
                if scores is not None:
                    scores = scores.flatten()
                if classes is not None:
                    classes = classes.flatten()

                # If classes look like scores and scores look like classes, swap
                if scores is not None and classes is not None:
                    cls_max = float(classes.max()) if classes.size else 0.0
                    cls_min = float(classes.min()) if classes.size else 0.0
                    scr_max = float(scores.max()) if scores.size else 0.0
                    scr_min = float(scores.min()) if scores.size else 0.0
                    classes_int_like = np.allclose(classes, np.round(classes), atol=1e-3) and cls_max > 1
                    scores_prob_like = (scr_min >= 0.0 and scr_max <= 1.0 + 1e-3)
                    classes_prob_like = (cls_min >= 0.0 and cls_max <= 1.0 + 1e-3)
                    scores_int_like = np.allclose(scores, np.round(scores), atol=1e-3) and scr_max > 1
                    if classes_prob_like and scores_int_like and not classes_int_like and not scores_prob_like:
                        classes, scores = scores, classes

                # Ensure we have the same number of detections
                num_dets = boxes.shape[0]
                if scores is None:
                    scores = np.ones(num_dets, dtype=np.float32)
                if classes is None:
                    classes = np.zeros(num_dets, dtype=np.int32)
                if scores.size != num_dets:
                    scores = scores[:num_dets]
                if classes.size != num_dets:
                    classes = classes[:num_dets]
                
                logging.debug(f"Processing {num_dets} detections")
                
                # Process each detection
                for i in range(num_dets):
                    try:
                        box = boxes[i]
                        score = float(scores[i])
                        class_id = int(classes[i])
                        
                        # Filter by confidence and class
                        if score < self.conf_thres:
                            continue
                        if class_id != self.person_class_id:
                            continue
                        
                        x1, y1, x2, y2 = box
                        
                        # Rescale coordinates
                        x1 = (float(x1) - pad[0]) / scale
                        y1 = (float(y1) - pad[1]) / scale
                        x2 = (float(x2) - pad[0]) / scale
                        y2 = (float(y2) - pad[1]) / scale
                        
                        # Clip to image bounds
                        x1 = max(0, min(x1, orig_shape[1]))
                        y1 = max(0, min(y1, orig_shape[0]))
                        x2 = max(0, min(x2, orig_shape[1]))
                        y2 = max(0, min(y2, orig_shape[0]))
                        
                        if x2 > x1 and y2 > y1:
                            detections.append(np.array([x1, y1, x2, y2, score], dtype=np.float32))
                            logging.debug(f"Detection {i}: class={class_id}, score={score:.3f}, box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                    
                    except Exception as e:
                        logging.debug(f"Error processing detection {i}: {e}")
                        continue
            
            elif len(outputs) == 1:
                # Single output tensor: [batch, num_detections, 7]
                # Format: [x1, y1, x2, y2, score, class_id, ...]
                dets = outputs[0]
                
                # Remove batch dimension if present
                if dets.ndim == 3:
                    dets = dets[0]
                
                for det in dets:
                    if len(det) >= 6:
                        x1, y1, x2, y2, score, class_id = det[:6]
                    else:
                        continue
                    
                    # Filter by confidence and class
                    if float(score) < self.conf_thres:
                        continue
                    if int(class_id) != self.person_class_id:
                        continue
                    
                    # Rescale coordinates
                    x1 = (float(x1) - pad[0]) / scale
                    y1 = (float(y1) - pad[1]) / scale
                    x2 = (float(x2) - pad[0]) / scale
                    y2 = (float(y2) - pad[1]) / scale
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, orig_shape[1]))
                    y1 = max(0, min(y1, orig_shape[0]))
                    x2 = max(0, min(x2, orig_shape[1]))
                    y2 = max(0, min(y2, orig_shape[0]))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append(np.array([x1, y1, x2, y2, float(score)], dtype=np.float32))

        except Exception as e:
            logging.warning(f"Error parsing EfficientDet outputs: {e}")
            import traceback
            logging.debug(traceback.format_exc())
            return []

        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections, self.nms_thres)
            logging.debug(f"After NMS: {len(detections)} detections")

        return detections

    def _nms(self, detections: List[np.ndarray], iou_threshold: float) -> List[np.ndarray]:
        """
        Apply Non-Maximum Suppression.

        Args:
            detections: List of [x1, y1, x2, y2, score] arrays
            iou_threshold: IoU threshold for NMS

        Returns:
            Filtered detections after NMS
        """
        if len(detections) == 0:
            return []

        # Convert to numpy array
        boxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])

        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Compute areas
        areas = (x2 - x1) * (y2 - y1)

        # Sort by score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect persons in image.

        Args:
            image: Input image (BGR format)

        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess
        input_tensor, scale, pad = self._preprocess(image)

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            logging.error(f"EfficientDet inference failed: {e}")
            return []

        # Post-process
        detections = self._postprocess(outputs, scale, pad, (orig_h, orig_w))

        return detections
