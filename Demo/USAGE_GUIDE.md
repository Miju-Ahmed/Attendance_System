# Entry/Exit Attendance System - Complete Guide

## ✅ System Status

Your system is fully configured and ready to use! 

**Available Features:**
- ✅ Face Detection (SCRFD model)
- ✅ Face Recognition (ArcFace model)
- ✅ 11 Known Persons Loaded
- ✅ YOLO Person Detection
- ✅ Multi-Embedding Fusion
- ✅ SQLite Database
- ✅ Entry/Exit Tracking with Red Line

**Known Persons:**
1. ALAMIN
2. AMIT
3. ASIF
4. MAHIB
5. MEHEDI
6. MERAJ
7. MIJU
8. MITHU
9. NAFI
10. RUDRO
11. SADI

---

## 🚀 Quick Start

### Option 1: Interactive Launcher (Recommended)
```bash
./run_attendance.sh
```

This will:
1. Check all requirements
2. Ask you to select video source (webcam or file)
3. Ask if you want to save output video
4. Launch the system

### Option 2: Direct Python Command

**For Webcam:**
```bash
python entry_exit_attendance.py --source 0
```

**For Video File:**
```bash
python entry_exit_attendance.py --source /path/to/video.mp4
```

**With Output Video:**
```bash
python entry_exit_attendance.py --source 0 --output attendance_output.mp4
```

---

## 📋 How to Use

### Step 1: Draw the Red Line

When you start the system, you'll see a window asking you to draw the red line:

1. **Click Point 1**: Click on the first point of the line
2. **Click Point 2**: Click on the second point of the line
3. **Press 's'**: Continue to start tracking

**Line Direction:**
- Crossing from **Side A → Side B** = ENTRY
- Crossing from **Side B → Side A** = EXIT

💡 **Tip**: Draw the line perpendicular to the expected path of movement.

### Step 2: System Runs Automatically

The system will:
1. ✅ Detect people using YOLO
2. ✅ Recognize faces from your known_faces folder
3. ✅ Track when people cross the red line
4. ✅ Record ENTRY when crossing one way
5. ✅ Record EXIT when crossing the other way
6. ✅ Store all events in `attendance.db` with timestamps

### Step 3: View Results

**During tracking:**
- Green boxes = Recognized person
- Orange boxes = Unknown person
- Red line = Entry/Exit boundary
- Top-left corner shows Entry/Exit counts

**To quit:** Press 'q'

---

## 📊 View Attendance Records

### Quick Summary
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

### Export to CSV
```bash
python view_attendance.py --export attendance_export.csv
```

### Direct SQLite Query
```bash
sqlite3 attendance.db "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10;"
```

---

## 🎛️ Advanced Usage

### Adjust Face Recognition Threshold
```bash
python entry_exit_attendance.py \
    --source 0 \
    --similarity-threshold 0.5  # Higher = stricter (default: 0.45)
```

### Run on CPU
```bash
python entry_exit_attendance.py --source 0 --device cpu
```

### Headless Mode (No Display)
```bash
python entry_exit_attendance.py \
    --source video.mp4 \
    --output result.mp4 \
    --skip-display
```

### Custom Database
```bash
python entry_exit_attendance.py \
    --source 0 \
    --database-path custom_attendance.db
```

---

## 🔧 Configuration Options

Edit these values in `entry_exit_attendance.py`:

```python
# Detection thresholds
YOLO_CONFIDENCE = 0.4          # YOLO detection confidence
FACE_CONFIDENCE = 0.5          # Face detection confidence
FACE_SIMILARITY_THRESHOLD = 0.45  # Face matching threshold

# Tracking parameters
MIN_MOVEMENT = 5.0             # Minimum pixels for movement
COOLDOWN_FRAMES = 30           # Frames between events (same person)

# Multi-Embedding Fusion
MEF_BUFFER_SIZE = 5            # Number of embeddings to buffer
MEF_WEIGHTS = [0.4, 0.3, 0.2, 0.08, 0.02]  # Recent = higher weight
```

---

## 📈 Understanding the Output

### On-Screen Display

```
┌─────────────────────────────────────┐
│ Entry: 5                            │  ← Total entries
│ Exit: 3                             │  ← Total exits
│                                     │
│  ┌─────────────┐                   │
│  │ ID:1 MIJU   │  ← Green = Known  │
│  │ (0.87)      │  ← Confidence     │
│  └─────────────┘                   │
│                                     │
│  ┌─────────────────┐               │
│  │ ID:2 Unknown    │  ← Orange     │
│  └─────────────────┘               │
│                                     │
│        Red Line ────────────       │  ← Entry/Exit line
└─────────────────────────────────────┘
```

### Database Records

Each record contains:
- **person_name**: Recognized name (e.g., "MIJU")
- **event_type**: "ENTRY" or "EXIT"
- **timestamp**: "2026-01-21 18:30:45"
- **confidence**: Face recognition confidence (0.0 - 1.0)
- **track_id**: YOLO tracking ID

---

## 🎯 Use Cases

### 1. Office Entry/Exit Tracking
```bash
# Track employees entering/leaving office
python entry_exit_attendance.py --source 0
```

### 2. Event Attendance
```bash
# Track attendees at an event
python entry_exit_attendance.py \
    --source event_video.mp4 \
    --output event_attendance.mp4
```

### 3. Security Monitoring
```bash
# Monitor area with high accuracy
python entry_exit_attendance.py \
    --source 0 \
    --similarity-threshold 0.6 \
    --yolo-conf 0.5
```

---

## 🐛 Troubleshooting

### Problem: No Faces Detected

**Solutions:**
1. Check if person is close enough to camera
2. Adjust face confidence: `--face-conf 0.3`
3. Ensure good lighting
4. Verify models are loaded (run `python check_setup.py`)

### Problem: Wrong Person Recognition

**Solutions:**
1. Increase similarity threshold: `--similarity-threshold 0.6`
2. Add more face embeddings for the person
3. Check if lighting matches training images
4. MEF will help after a few frames

### Problem: Duplicate Counts

**Solution:**
- Increase `COOLDOWN_FRAMES` in the script (default: 30)
- The system has built-in cooldown to prevent this

### Problem: Database Locked

**Solutions:**
1. Close other programs accessing `attendance.db`
2. Use a different database: `--database-path attendance2.db`

### Problem: Slow Performance

**Solutions:**
1. Use CPU if GPU issues: `--device cpu`
2. Lower YOLO confidence: `--yolo-conf 0.3`
3. Reduce video resolution before processing

---

## 💡 Tips for Best Results

### 1. Line Placement
- Draw line **perpendicular** to movement direction
- Place in area with good lighting
- Avoid areas where people stop/linger

### 2. Camera Position
- Mount camera to capture full body
- Angle should show faces clearly
- Avoid backlighting

### 3. Known Faces
- Use multiple photos per person (if adding more)
- Ensure good quality images
- Similar lighting to deployment environment

### 4. Multi-Embedding Fusion
- System gets **more accurate** after tracking for a few frames
- First detection might be uncertain, but MEF improves it
- Works best when person is tracked continuously

---

## 📝 Example Workflow

### Complete Session

```bash
# 1. Check setup
python check_setup.py

# 2. Run attendance tracking
python entry_exit_attendance.py --source 0 --output today.mp4

# 3. Draw red line when prompted
#    - Click 2 points
#    - Press 's'

# 4. System tracks entry/exit
#    - Watch the screen
#    - Press 'q' when done

# 5. View results
python view_attendance.py --summary

# 6. Export for records
python view_attendance.py --export attendance_$(date +%Y%m%d).csv

# 7. Done!
```

---

## 🔐 Database Schema

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_name TEXT NOT NULL,      -- e.g., "MIJU"
    event_type TEXT NOT NULL,       -- "ENTRY" or "EXIT"
    timestamp DATETIME NOT NULL,    -- "2026-01-21 18:30:45"
    confidence REAL,                -- 0.45 to 1.0
    track_id INTEGER                -- YOLO tracking ID
);
```

### Example Query - Get Today's Attendance
```sql
SELECT 
    person_name,
    COUNT(CASE WHEN event_type = 'ENTRY' THEN 1 END) as entries,
    COUNT(CASE WHEN event_type = 'EXIT' THEN 1 END) as exits
FROM attendance
WHERE DATE(timestamp) = DATE('now')
GROUP BY person_name;
```

---

## 📞 Support

If you encounter issues:

1. Run setup check: `python check_setup.py`
2. Check logs in terminal output
3. Verify all models exist in `weights/` folder
4. Ensure known faces are in `known_faces/` as `.npy` files

---

## 🎓 Technical Details

### Multi-Embedding Fusion (MEF)

The system uses MEF to improve accuracy:

1. **Buffers** last 5 face embeddings per person
2. **Weights** recent embeddings higher: [0.4, 0.3, 0.2, 0.08, 0.02]
3. **Fuses** embeddings using weighted average
4. **Benefits**: More robust to pose/lighting variations

### Line Crossing Algorithm

1. Tracks centroid (bottom-center of bbox)
2. Determines which side of line person is on
3. Detects when side changes
4. Records event based on direction
5. Applies cooldown to prevent duplicates

### Face Recognition Pipeline

1. YOLO detects person
2. Crop upper 60% of person bbox
3. SCRFD detects face in crop
4. ArcFace generates 512-D embedding
5. Compare with known faces using cosine similarity
6. MEF fusion for final decision

---

## ✨ Features Summary

✅ Real-time person detection with YOLO  
✅ Face recognition with SCRFD + ArcFace  
✅ Multi-Embedding Fusion for accuracy  
✅ Red line crossing detection  
✅ Entry/Exit event tracking  
✅ SQLite database with timestamps  
✅ Track-based identification  
✅ Cooldown to prevent duplicates  
✅ Visual feedback and statistics  
✅ CSV export capability  
✅ Works with webcam or video files  

---

## 🏁 You're Ready!

Everything is configured and working. Start tracking attendance:

```bash
./run_attendance.sh
```

or

```bash
python entry_exit_attendance.py --source 0
```

**Good luck with your attendance tracking! 🎉**
