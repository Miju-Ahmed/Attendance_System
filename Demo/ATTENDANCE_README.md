# Entry/Exit Attendance System

## Overview

This system tracks people crossing a red line for entry/exit detection with face recognition:

- **Person Detection**: Uses YOLO for detecting people
- **Face Recognition**: Uses SCRFD for face detection and ArcFace for recognition
- **Multi-Embedding Fusion (MEF)**: Combines multiple face embeddings for better accuracy
- **Line Crossing Detection**: Tracks when people cross a red line
- **Database Storage**: Stores all entry/exit events with timestamps in SQLite

## Features

✅ **Real-time face recognition** from known faces in `known_faces/` folder  
✅ **Automatic entry/exit detection** based on red line crossing  
✅ **Multi-Embedding Fusion** for robust face matching  
✅ **SQLite database** with timestamps for all events  
✅ **Track-based identification** to prevent duplicate counts  
✅ **Interactive line drawing** at startup  
✅ **Visual feedback** with bounding boxes and labels

## How It Works

1. **Red Line Crossing**:
   - Crossing from left/top → right/bottom = **ENTRY**
   - Crossing from right/bottom → left/top = **EXIT**

2. **Face Recognition**:
   - Detects faces in person bounding boxes
   - Compares with known faces from `known_faces/` folder
   - Uses MEF to combine multiple embeddings for better accuracy
   - Only records attendance for recognized faces

3. **Database Storage**:
   - Stores person name, event type (ENTRY/EXIT), timestamp, confidence, and track ID
   - Creates `attendance.db` SQLite database automatically

## Known Faces Structure

The system expects face embeddings in `.npy` format in the `known_faces/` directory:

```
known_faces/
├── PERSON1.npy    # Pre-computed face embeddings
├── PERSON2.npy
├── PERSON3.npy
└── ...
```

Each `.npy` file contains face embeddings (512-dimensional vectors) for that person.

## Usage

### Basic Usage (Webcam)

```bash
python entry_exit_attendance.py --source 0
```

### Video File

```bash
python entry_exit_attendance.py --source /path/to/video.mp4
```

### Custom Model Paths

```bash
python entry_exit_attendance.py \
    --source 0 \
    --yolo-model yolov8n.pt \
    --face-det-model ./weights/face_detection/det_10g.onnx \
    --face-rec-model ./weights/face_recognition/w600k_r50.onnx \
    --known-faces-dir ./known_faces
```

### Save Output Video

```bash
python entry_exit_attendance.py \
    --source video.mp4 \
    --output output_attendance.mp4
```

### Run on CPU

```bash
python entry_exit_attendance.py --source 0 --device cpu
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video source (0 for webcam or video file path) |
| `--yolo-model` | `yolov8n.pt` | Path to YOLO model |
| `--face-det-model` | `./weights/face_detection/det_10g.onnx` | Face detection model (SCRFD) |
| `--face-rec-model` | `./weights/face_recognition/w600k_r50.onnx` | Face recognition model (ArcFace) |
| `--known-faces-dir` | `./known_faces` | Directory with known face embeddings |
| `--database-path` | `attendance.db` | SQLite database path |
| `--yolo-conf` | `0.4` | YOLO confidence threshold |
| `--face-conf` | `0.5` | Face detection confidence threshold |
| `--similarity-threshold` | `0.45` | Face similarity threshold |
| `--device` | `cuda` | Device (cuda or cpu) |
| `--output` | `None` | Output video path (optional) |
| `--skip-display` | `False` | Skip display (headless mode) |

## Interactive Setup

When you run the script:

1. **Draw Red Line**: Click 2 points to define the entry/exit line
   - First window will show up
   - Click point 1 and point 2
   - Press 's' to continue

2. **Processing**: The system will:
   - Detect people using YOLO
   - Recognize faces using SCRFD + ArcFace
   - Track line crossings
   - Record attendance in database

3. **Exit**: Press 'q' to quit

## Database Schema

The `attendance.db` database has the following structure:

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,
    event_type TEXT NOT NULL,           -- 'ENTRY' or 'EXIT'
    timestamp DATETIME NOT NULL,         -- Format: 'YYYY-MM-DD HH:MM:SS'
    confidence REAL,                     -- Face recognition confidence
    track_id INTEGER                     -- YOLO track ID
);
```

## View Database Records

You can view attendance records using SQLite:

```bash
sqlite3 attendance.db "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10;"
```

Or using Python:

```python
import sqlite3

conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Get all records
cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
for row in cursor.fetchall():
    print(row)

conn.close()
```

## Multi-Embedding Fusion (MEF)

The system uses MEF to improve face recognition accuracy:

- Maintains a buffer of the last 5 face embeddings per person
- Uses weighted average with higher weights for recent embeddings
- Weights: [0.4, 0.3, 0.2, 0.08, 0.02] (most recent to oldest)
- Provides more stable and accurate face recognition

## Configuration

You can adjust these parameters in the script:

```python
# Detection parameters
YOLO_CONFIDENCE = 0.4
FACE_CONFIDENCE = 0.5
FACE_SIMILARITY_THRESHOLD = 0.45

# Line crossing parameters
MIN_MOVEMENT = 5.0
COOLDOWN_FRAMES = 30  # Prevent duplicate counts

# Multi-Embedding Fusion
MEF_BUFFER_SIZE = 5
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLO)
- ONNX Runtime
- SQLite3 (built-in with Python)

## Troubleshooting

### No faces detected
- Make sure face models are in `weights/face_detection/` and `weights/face_recognition/`
- Check if person crops are large enough
- Adjust `--face-conf` threshold

### No known faces loaded
- Ensure `.npy` files are in `known_faces/` directory
- Check file permissions
- Verify embeddings are in correct format (512-dimensional vectors)

### Database errors
- Check write permissions in current directory
- Ensure `attendance.db` is not locked by another process

### Performance issues
- Use `--device cpu` if GPU issues occur
- Reduce video resolution
- Adjust detection intervals

## Example Workflow

1. **Prepare known faces**:
   ```bash
   # Your known_faces folder should have .npy files
   ls known_faces/
   # Output: ALAMIN.npy  AMIT.npy  MIJU.npy  ...
   ```

2. **Run the system**:
   ```bash
   python entry_exit_attendance.py --source 0
   ```

3. **Draw the red line** when prompted

4. **System will track**:
   - Detect people
   - Recognize faces
   - Log entry/exit events

5. **Check attendance**:
   ```bash
   sqlite3 attendance.db "SELECT * FROM attendance;"
   ```

## Notes

- The system uses **bottom-center** of bounding box as the tracking point (foot position)
- **Cooldown** prevents duplicate counts (default: 30 frames)
- Only **recognized faces** will have attendance recorded
- **Unknown** people are tracked but not logged in database
- The red line direction determines entry vs exit

## Credits

- YOLO: Ultralytics
- Face Detection: SCRFD (InsightFace)
- Face Recognition: ArcFace (InsightFace)
- Multi-Embedding Fusion: Based on MEF techniques
