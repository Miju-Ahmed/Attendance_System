# EfficientDet-D0 Attendance System - README

## Overview

This is a **production-ready real-time attendance system** using **EfficientDet-D0** for human detection, combined with SCRFD for face detection and ArcFace for face recognition. The system tracks people across frames, authorizes them via face recognition, and logs entry/exit events using a line-crossing state machine.

## ✅ Hard Constraints Enforced

This implementation strictly enforces the following constraints:

- ✅ **EfficientDet-D0 ONLY** - No YOLO for body detection
- ✅ **NO motion prediction** - Tracks update ONLY when EfficientDet-D0 detects a body
- ✅ **NO face-based body heuristics** - Face boxes NEVER used to estimate/extend body boxes
- ✅ **Detection-driven tracking** - Tracks freeze when no detection exists
- ✅ **Persistent identity** - Once authorized, identity persists until track timeout
- ✅ **NO identity reassignment** - Identity never changes after authorization

## Architecture

### Pipeline Flow

```
1. EfficientDet-D0 Detection (Body-Only)
   ↓
2. IoU-Based Tracking (No Prediction)
   ↓
3. SCRFD Face Detection (Inside Body Boxes)
   ↓
4. ArcFace Recognition (Authorization Only)
   ↓
5. Persistent Tracking (Body-Only, Even After Face Loss)
   ↓
6. Line-Crossing State Machine (Entry/Exit Detection)
   ↓
7. SQLite Database Logging
```

### Key Components

#### 1. **EfficientDet-D0 Detection** (`models/EfficientDet.py`)
- ONNX model wrapper for person detection
- Input: 512x512 (D0 standard)
- Output: Person bounding boxes only (COCO class 0)
- Preprocessing: ImageNet normalization
- Post-processing: NMS, coordinate scaling

#### 2. **Detection-Driven Tracking** (`IoUTracker`)
- **NO Kalman filtering**
- **NO motion prediction**
- Tracks update ONLY when matched to EfficientDet-D0 detection
- Tracks freeze at last known position when no detection matches
- Track removal after `TRACK_MAX_AGE` frames without detection

#### 3. **Face Authorization** (`FaceAuthorizer`)
- Face detection: SCRFD (inside body boxes only)
- Face recognition: ArcFace (cosine similarity)
- **Authorization happens ONCE** - face used only to identify person
- After authorization, face visibility NOT required

#### 4. **Persistent Tracking**
- Once authorized, identity persists via body tracking
- Identity attached to `stable_id` (from database)
- Track continues even if face disappears
- Identity removed only after track timeout

#### 5. **Line-Crossing State Machine** (`LineCrossDetector`)
- Two virtual lines: Entry (green) and Exit (red)
- Crossing detection: bottom-center point of body box
- Per-person state machine: `UNKNOWN` → `INSIDE` / `OUTSIDE`
- Cooldown period prevents duplicate events
- **Only authorized persons trigger events**

#### 6. **Database Logging** (`AttendanceDatabase`)
- SQLite database: `attendance.db`
- Schema:
  ```sql
  CREATE TABLE attendance (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      person_name TEXT NOT NULL,
      stable_id INTEGER NOT NULL,
      event_type TEXT NOT NULL,  -- 'ENTRY' or 'EXIT'
      timestamp DATETIME NOT NULL,
      confidence REAL
  );
  ```
- Multiple entry/exit records per person per day allowed

## Installation

### Prerequisites

```bash
# Required Python packages
pip install opencv-python numpy onnxruntime-gpu  # or onnxruntime for CPU-only
```

### Directory Structure

```
Demo/
├── attendance_efficientnetdet.py  # Main script
├── models/
│   ├── EfficientDet.py           # EfficientDet-D0 wrapper
│   ├── SCRFD.py                  # Face detection
│   └── ArcFace.py                # Face recognition
├── weights/
│   ├── efficientdet_d0.onnx      # EfficientDet-D0 model (REQUIRED)
│   ├── face_detection/
│   │   └── det_10g.onnx          # SCRFD model
│   └── face_recognition/
│       └── w600k_r50.onnx        # ArcFace model
├── known_faces/                   # .npy face embeddings
│   ├── person1.npy
│   └── person2.npy
└── attendance.db                  # SQLite database (auto-created)
```

### EfficientDet-D0 Model Setup

**IMPORTANT:** You need an EfficientDet-D0 ONNX model. Options:

1. **Download pre-converted model:**
   ```bash
   # Download from a model zoo or convert from TensorFlow
   # Place at: ./weights/efficientdet_d0.onnx
   ```

2. **Convert from TensorFlow:**
   ```bash
   # Use tf2onnx or similar tools to convert
   # Ensure input size is 512x512 and output format matches expectations
   ```

3. **Use custom path:**
   ```bash
   python attendance_efficientnetdet.py --efficientdet-model-path /path/to/model.onnx
   ```

## Usage

### Basic Usage (Webcam)

```bash
python attendance_efficientnetdet.py --source 0
```

### RTSP Camera

```bash
python attendance_efficientnetdet.py --source "rtsp://admin:password@192.168.0.3:554/stream"
```

### Video File

```bash
python attendance_efficientnetdet.py --source /path/to/video.mp4
```

### Full Command-Line Options

```bash
python attendance_efficientnetdet.py \
    --source 0 \
    --efficientdet-model-path ./weights/efficientdet_d0.onnx \
    --face-det-model ./weights/face_detection/det_10g.onnx \
    --face-rec-model ./weights/face_recognition/w600k_r50.onnx \
    --device cuda \
    --detection-conf 0.3 \
    --nms-threshold 0.5 \
    --face-conf 0.5 \
    --similarity-threshold 0.40 \
    --iou-threshold 0.3 \
    --track-max-age 10 \
    --resize-width 960 \
    --output output.mp4 \
    --known-faces-dir ./known_faces \
    --database-path attendance.db \
    --rebuild-known-faces
```

### Interactive Line Drawing

When you start the script:

1. **Draw Entry Line (Green):**
   - Click two points to define the entry line
   - Press `s` to save
   - Press `q` to cancel

2. **Draw Exit Line (Red):**
   - Click two points to define the exit line
   - Press `s` to save
   - Press `q` to cancel

3. **Processing starts automatically**

### Viewing Attendance Records

```bash
python view_db.py
```

Or query directly:

```bash
sqlite3 attendance.db "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10;"
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--detection-conf` | 0.3 | EfficientDet confidence threshold |
| `--nms-threshold` | 0.5 | NMS IoU threshold |
| `--face-conf` | 0.5 | Face detection confidence |
| `--similarity-threshold` | 0.40 | Face recognition similarity threshold |
| `--iou-threshold` | 0.3 | IoU threshold for tracking |
| `--track-max-age` | 10 | Frames before track removal |
| `--resize-width` | 960 | Processing width (0 = no resize) |

### Performance Tuning

**For faster processing:**
- Reduce `--resize-width` (e.g., 640)
- Increase `--detection-conf` (e.g., 0.4)
- Use GPU: `--device cuda`

**For better accuracy:**
- Increase `--resize-width` (e.g., 1280)
- Decrease `--detection-conf` (e.g., 0.25)
- Decrease `--similarity-threshold` (e.g., 0.35)

## Verification Checklist

### ✅ Code Verification

- [x] NO YOLO imports in code
- [x] NO motion prediction in tracker
- [x] NO face-based body heuristics
- [x] NO identity reassignment after authorization
- [x] Tracks update ONLY when EfficientDet-D0 detects
- [x] Identity persists after face loss

### ✅ Runtime Verification

Run these tests to verify correct behavior:

1. **Detection Test:**
   ```bash
   python -c "from models.EfficientDet import EfficientDet; import cv2; model = EfficientDet('./weights/efficientdet_d0.onnx', 0.3, 0.5, ('CPUExecutionProvider',)); img = cv2.imread('test.jpg'); dets = model.detect(img); print(f'Detected {len(dets)} persons')"
   ```

2. **No YOLO Test:**
   ```bash
   grep -r "from ultralytics import YOLO" attendance_efficientnetdet.py
   # Should return nothing
   ```

3. **Tracking Test:**
   - Run with video showing person temporarily occluded
   - Verify track ID persists during occlusion
   - Verify track freezes (no position update) during occlusion

4. **Face Loss Test:**
   - Authorize a person (face recognized)
   - Person turns away (face not visible)
   - Verify identity still displayed on body box
   - Verify entry/exit events still trigger

5. **Line Crossing Test:**
   - Draw entry/exit lines
   - Walk across lines
   - Check database for correct events:
     ```bash
     sqlite3 attendance.db "SELECT * FROM attendance WHERE person_name='YourName' ORDER BY timestamp DESC;"
     ```

## Troubleshooting

### Model Loading Issues

**Error: "Failed to load EfficientDet model"**
- Verify model path: `--efficientdet-model-path`
- Check ONNX Runtime installation: `pip install onnxruntime-gpu`
- Try CPU mode: `--device cpu`

### Detection Issues

**No persons detected:**
- Lower confidence: `--detection-conf 0.2`
- Check input size compatibility (model expects 512x512)
- Verify model outputs person class (COCO class 0)

**Too many false positives:**
- Increase confidence: `--detection-conf 0.4`
- Adjust NMS threshold: `--nms-threshold 0.4`

### Tracking Issues

**Track IDs change frequently:**
- Increase IoU threshold: `--iou-threshold 0.4`
- Increase max age: `--track-max-age 15`

**Tracks persist too long:**
- Decrease max age: `--track-max-age 5`

### Face Recognition Issues

**Faces not recognized:**
- Lower similarity threshold: `--similarity-threshold 0.35`
- Check known faces directory has .npy files
- Rebuild database: `--rebuild-known-faces`

**Wrong person identified:**
- Increase similarity threshold: `--similarity-threshold 0.45`
- Improve face embeddings quality

## Performance Benchmarks

Expected performance on different hardware:

| Hardware | Resolution | FPS | Notes |
|----------|-----------|-----|-------|
| RTX 3090 | 1920x1080 | 25-30 | Full resolution, CUDA |
| RTX 3060 | 1280x720 | 20-25 | Resized, CUDA |
| CPU (i7) | 960x540 | 5-8 | Resized, CPU only |

## Database Schema

### `attendance` Table

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    stable_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,  -- 'ENTRY' or 'EXIT'
    timestamp DATETIME NOT NULL,
    confidence REAL
);
```

### `face_embeddings` Table

```sql
CREATE TABLE face_embeddings (
    person_name TEXT NOT NULL,
    stable_id INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
```

## Key Differences from YOLO Version

| Aspect | YOLO Version | EfficientDet-D0 Version |
|--------|--------------|-------------------------|
| Body Detection | YOLO (ultralytics) | EfficientDet-D0 (ONNX) |
| Model Format | .pt (PyTorch) | .onnx (ONNX Runtime) |
| Input Size | 640x640 | 512x512 |
| Preprocessing | YOLO built-in | Manual (ImageNet norm) |
| Post-processing | YOLO built-in | Manual (NMS) |
| Dependencies | ultralytics | onnxruntime only |

## License & Credits

- **EfficientDet:** Google Research
- **SCRFD:** InsightFace
- **ArcFace:** InsightFace
- **Implementation:** Custom attendance system

## Support

For issues or questions:
1. Check troubleshooting section
2. Verify all hard constraints are met
3. Test with sample video first
4. Check model compatibility

---

**Built with EfficientDet-D0 - No YOLO, No Prediction, Detection-Driven Only** ✅
