# 🎭 Face-Based Attendance System - NEW & IMPROVED!

## ✨ What Changed?

**OLD System:** YOLO person detection → Face detection → Recognition  
**NEW System:** Direct face detection → Recognition ✅

### Benefits:
- 🚀 **Faster** - No YOLO overhead
- 💡 **Simpler** - Direct face-to-attendance pipeline
- 🎯 **More accurate** - Focused on faces only
- 💾 **Lighter** - No need for large YOLO models

---

## 🚀 Quick Start

### Easiest Way:
```bash
python test_with_video.py
```

### Manual:
```bash
python entry_exit_attendance.py --source /path/to/video.mp4
```

### With Good Test Video:
```bash
python entry_exit_attendance.py \
    --source "/run/media/miju_chowdhury/Miju/WorkSpace/Apex_All_Project_Demo/People-counting-system_good/input/merged_trimmed_D01_60609_3.mp4" \
    --output face_attendance.mp4
```

---

## 🎯 How It Works

### 1. **Face Detection** (SCRFD)
   - Detects all faces in each frame
   - No need for person bounding boxes!

### 2. **Face Recognition** (ArcFace)
   - Generates 512-D embeddings
   - Compares with known faces

### 3. **Multi-Embedding Fusion (MEF)**
   - Buffers last 5 embeddings per face
   - Weighted fusion for robustness
   - More accurate over time

### 4. **Face Tracking**
   - Tracks each face across frames
   - Maintains consistent ID
   - Associates detections intelligently

### 5. **Line Crossing**
   - Tracks face center position
   - Detects crossing of red line
   - Records ENTRY/EXIT events

---

## 📊 What You'll See

```
┌──────────────────────────────────────┐
│ Entry: 3    Exit: 2    Faces: 2     │  ← Stats
│                                      │
│    🟢 ID:1 MIJU (0.87)              │  ← Recognized face
│    [Green box around face]           │
│                                      │
│    🟠 ID:2 Unknown                  │  ← Unknown face
│    [Orange box around face]          │
│                                      │
│         Red Line ──────────         │  ← Entry/Exit
└──────────────────────────────────────┘
```

**Display:**
- 🟢 **Green boxes** = Recognized faces
- 🟠 **Orange boxes** = Unknown faces  
- 📍 **Center dots** = Tracking points
- 🔴 **Red line** = Entry/Exit boundary
- 📊 **Top stats** = Entries, Exits, Current faces

---

## 🎬 Step-by-Step Usage

### Step 1: Start
```bash
python entry_exit_attendance.py --source video.mp4
```

### Step 2: Draw Line
- Window opens with first frame
- Click **Point 1** anywhere on screen
- Click **Point 2** to complete line
- Press **'s'** to start

**Line Direction:**
- Left to Right / Top to Bottom = **ENTRY**
- Right to Left / Bottom to Top = **EXIT**

### Step 3: Watch
System automatically:
- ✅ Detects faces
- ✅ Recognizes from known_faces/
- ✅ Tracks across frames
- ✅ Records line crossings
- ✅ Saves to database

### Step 4: Quit
Press **'q'** when done

### Step 5: Results
```bash
python view_attendance.py --summary
```

---

## 📋 Command Options

```bash
python entry_exit_attendance.py \
    --source VIDEO_PATH \              # Video file or 0 for webcam
    --output OUTPUT_PATH \             # Save output video (optional)
    --face-conf 0.5 \                  # Face detection threshold
    --similarity-threshold 0.45 \      # Recognition threshold
    --database-path attendance.db      # Database path
```

---

## 💡 Key Features

### 1. **Smart Face Tracking**
   - Tracks faces across frames even with movement
   - Maintains consistent IDs
   - Associates nearby detections intelligently

### 2. **Multi-Embedding Fusion (MEF)**
   ```python
   Weights: [0.4, 0.3, 0.2, 0.08, 0.02]
          Most recent → Oldest
   ```
   - Gets more accurate over time
   - Handles pose/lighting variations
   - Robust to temporary occlusions

### 3. **Intelligent Attendance Recording**
   - Only records **recognized** faces
   - Requires **3+ detections** before recording (MIN_TRACK_CONFIDENCE)
   - **30-frame cooldown** prevents duplicates
   - Stores with timestamp and confidence

### 4. **Database Storage**
   ```sql
   attendance (
       id, person_name, event_type,
       timestamp, confidence, track_id
   )
   ```

---

## 🎓 Configuration

Edit in `entry_exit_attendance.py`:

```python
# Detection
FACE_CONFIDENCE = 0.5              # Face detection threshold
FACE_SIMILARITY_THRESHOLD = 0.45   # Recognition threshold

# Tracking  
MAX_TRACK_DISTANCE = 100           # Max pixels to associate faces
MIN_TRACK_CONFIDENCE = 3           # Min detections before recording

# Line Crossing
COOLDOWN_FRAMES = 30               # Frames between events
MIN_MOVEMENT = 3.0                 # Min movement to consider

# MEF
MEF_BUFFER_SIZE = 5                # Embedding buffer size
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]
```

---

## 📊 View Attendance

### Summary
```bash
python view_attendance.py --summary
```

### All Records
```bash
python view_attendance.py --all --limit 50
```

### Today's Records
```bash
python view_attendance.py --today
```

### Specific Person
```bash
python view_attendance.py --person MIJU
```

### Export CSV
```bash
python view_attendance.py --export attendance.csv
```

### SQLite Query
```bash
sqlite3 attendance.db "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10;"
```

---

## 🔧 How It Tracks Faces

### Detection Association Algorithm:

1. **Detect faces** in current frame
2. **Compute distances** to existing tracks
3. **Match** closest faces (within MAX_TRACK_DISTANCE)
4. **Update** matched tracks
5. **Create new tracks** for unmatched faces
6. **Clean up** old tracks (>60 frames inactive)

### Crossing Detection:

1. Track face **center position**
2. Determine **which side** of line
3. Compare with **previous side**
4. If **side changed**: Record event!
5. Apply **cooldown** to prevent duplicates

---

## 🎯 Advantages Over YOLO Version

| Feature | YOLO Version | Face-Only Version |
|---------|--------------|-------------------|
| **Speed** | Slower | ✅ Faster |
| **Accuracy** | Good | ✅ Better for faces |
| **Complexity** | Higher | ✅ Simpler |
| **Dependencies** | YOLO + Face | ✅ Face only |
| **Focus** | Person → Face | ✅ Direct face |
| **Memory** | ~120MB model | ✅ ~20MB models |

---

## ✅ System Status

**Models:**
- ✅ SCRFD (Face Detection) - 10MB
- ✅ ArcFace (Face Recognition) - 8MB  
- ✅ Multi-Embedding Fusion enabled

**Known Faces (11):**
ALAMIN, AMIT, ASIF, MAHIB, MEHEDI, MERAJ, MIJU, MITHU, NAFI, RUDRO, SADI

**Features:**
- ✅ Direct face detection
- ✅ Face tracking across frames
- ✅ MEF for robust recognition
- ✅ Line crossing detection
- ✅ Database with timestamps
- ✅ Entry/Exit events

---

## 🐛 Troubleshooting

### No faces detected?
- **Check lighting** - Faces need to be visible
- **Lower threshold**: `--face-conf 0.3`
- **Check video quality**

### Wrong recognition?
- **Increase threshold**: `--similarity-threshold 0.6`
- MEF will improve accuracy after a few frames
- Check if person is in known_faces/

### Faces not tracking well?
- Increase `MAX_TRACK_DISTANCE` in config
- Faces might be moving too fast
- Video might have low frame rate

### Duplicate counts?
- Default cooldown is 30 frames (~1 second at 30fps)
- Increase `COOLDOWN_FRAMES` if needed

---

## 💾 Database Schema

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,        -- "MIJU", "AMIT", etc.
    event_type TEXT NOT NULL,         -- "ENTRY" or "EXIT"
    timestamp DATETIME NOT NULL,      -- "2026-01-21 19:10:30"
    confidence REAL,                  -- 0.45 - 1.0
    track_id INTEGER                  -- Face track ID
);
```

---

## 📝 Example Workflow

```bash
# 1. Run the face-based system
python entry_exit_attendance.py \
    --source video.mp4 \
    --output result.mp4

# 2. Draw line (2 clicks), press 's'

# 3. System processes:
#    - Detects faces
#    - Recognizes people
#    - Tracks crossings
#    - Records events

# 4. Press 'q' when done

# 5. View results
python view_attendance.py --summary

# Output:
# ==============================
# ATTENDANCE SUMMARY
# ==============================
# Total Entries: 5
# Total Exits: 3
# ==============================
# 
# Per Person Summary:
# Name     Entries  Exits
# ────────────────────────
# MIJU     2        1
# AMIT     1        1
# MEHEDI   2        1
```

---

## 🎉 Ready to Use!

The system is **simpler, faster, and more focused** than before!

### Quick Test:
```bash
python test_with_video.py
```

### Manual:
```bash
python entry_exit_attendance.py --source video.mp4
```

**No YOLO needed - Pure face-based attendance! 🎭✨**
