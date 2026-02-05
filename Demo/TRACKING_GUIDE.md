# Face Recognition & Persistent Tracking Guide

## 🎯 How It Works

The system does **exactly** what you requested:

### **Step 1: Face Recognition (Authorization)**
```
1. Person enters camera view
2. EfficientDet-D0 detects their BODY
3. SCRFD detects their FACE (if visible)
4. ArcFace compares face to known_faces/
5. If match found → Person is AUTHORIZED with their NAME
```

### **Step 2: Persistent Body Tracking**
```
6. Name is attached to the BODY track
7. System tracks person by FULL BODY (not face)
8. Name PERSISTS even if face is not visible
9. Tracking continues until person leaves frame
```

### **Step 3: Entry/Exit Logging**
```
10. When authorized person crosses ENTRY line → Log to database
11. When authorized person crosses EXIT line → Log to database
12. Only AUTHORIZED persons trigger events
```

---

## 📺 Visual Display

### **Recognized Person (Known Face)**
```
┌─────────────────────────┐
│ ████ John Doe ████      │ ← Green background with WHITE text
├─────────────────────────┤
│                         │
│    ┏━━━━━━━━━━━━━┓     │
│    ┃             ┃     │ ← THICK GREEN box (3px)
│    ┃   PERSON    ┃     │
│    ┃             ┃     │
│    ┗━━━━━━━━━━━━━┛     │
│ ID:1 | Conf:0.72       │ ← ID and confidence below
└─────────────────────────┘
```

**What you see:**
- ✅ **THICK GREEN box** around full body
- ✅ **Person's NAME** in large white text on green background
- ✅ **Stable ID** and **Confidence** below the box
- ✅ Name stays visible even if face turns away

### **Unknown Person (Not in Database)**
```
┌─────────────────────────┐
│ Unknown (ID:5)          │ ← Orange text
├─────────────────────────┤
│                         │
│    ┌─────────────┐     │
│    │             │     │ ← THIN ORANGE box (2px)
│    │   PERSON    │     │
│    │             │     │
│    └─────────────┘     │
└─────────────────────────┘
```

**What you see:**
- ⚠️ **THIN ORANGE box** around full body
- ⚠️ **"Unknown"** label with temporary track ID
- ⚠️ No entry/exit events logged

---

## 🔍 How Face Recognition Works

### **Known Faces Directory Structure**
```
known_faces/
├── John_Doe.npy        ← Face embeddings for John Doe
├── Jane_Smith.npy      ← Face embeddings for Jane Smith
├── Bob_Johnson.npy     ← Face embeddings for Bob Johnson
└── ...
```

### **Recognition Process**
1. **Face Detection:** SCRFD finds face in frame
2. **Embedding Extraction:** ArcFace converts face to 512-dim vector
3. **Similarity Comparison:** Compare with all known embeddings
4. **Threshold Check:** If similarity > 0.40 → MATCH
5. **Authorization:** Attach name to body track

### **Similarity Threshold**
- **0.40** (default) - Balanced
- **0.35** - More lenient (more false positives)
- **0.45** - Stricter (fewer false positives)

---

## 🎮 Testing the System

### **Test 1: Face Recognition Only**
```bash
python test_face_recognition.py
```

**What to expect:**
- Green boxes around recognized faces
- Red boxes around unknown faces
- Names and confidence scores displayed
- Console shows: `Face 1: RECOGNIZED - John (0.72)`

### **Test 2: Full Attendance System**
```bash
python attendance_efficientnetdet.py --source 0
```

**What to expect:**
1. **Draw entry line** (green) - click 2 points, press 's'
2. **Draw exit line** (red) - click 2 points, press 's'
3. **System starts:**
   - Bodies detected with orange boxes (unknown)
   - When face visible → Recognition attempt
   - If recognized → Box turns GREEN, name appears
   - Name PERSISTS even if face turns away
   - Cross entry line → "ENTRY" logged
   - Cross exit line → "EXIT" logged

### **Console Output**
```
INFO - Loaded 11 known persons.
INFO - ✓ Authorized: John (ID:1, conf:0.65)
INFO - ENTRY: John (ID:1) at 2026-02-02 17:45:00 [conf: 0.65]
INFO - ✓ Authorized: Jane (ID:2, conf:0.72)
INFO - ENTRY: Jane (ID:2) at 2026-02-02 17:45:15 [conf: 0.72]
INFO - EXIT: John (ID:1) at 2026-02-02 17:46:00 [conf: 0.65]
```

---

## 🔧 Configuration

### **Adjust Recognition Sensitivity**
```bash
# More lenient (recognize more people)
python attendance_efficientnetdet.py --source 0 --similarity-threshold 0.35

# Stricter (fewer false positives)
python attendance_efficientnetdet.py --source 0 --similarity-threshold 0.45
```

### **Adjust Face Detection**
```bash
# Detect more faces (lower confidence)
python attendance_efficientnetdet.py --source 0 --face-conf 0.3

# Detect only clear faces (higher confidence)
python attendance_efficientnetdet.py --source 0 --face-conf 0.6
```

### **Adjust Body Tracking**
```bash
# Keep tracks longer without detection
python attendance_efficientnetdet.py --source 0 --track-max-age 30

# Remove tracks faster
python attendance_efficientnetdet.py --source 0 --track-max-age 5
```

---

## ✅ Key Features (Already Implemented)

### ✅ **Persistent Identity**
- Once recognized, name stays attached to body
- Works even if person turns around
- Works even if face becomes occluded
- Identity only removed when person leaves frame for extended period

### ✅ **No Identity Reassignment**
- Once a person is authorized, their identity NEVER changes
- Exception: If a different person is detected with much higher confidence

### ✅ **Detection-Driven Tracking**
- Body tracking uses ONLY EfficientDet-D0 detections
- NO motion prediction
- NO Kalman filtering
- Track position updates ONLY when body is detected

### ✅ **Face for Authorization Only**
- Face recognition used ONLY to identify person
- After authorization, face is NOT needed
- Body tracking is independent of face

### ✅ **Entry/Exit State Machine**
- Prevents duplicate events
- Cooldown period between events
- Direction-aware (entry vs exit)
- Only authorized persons logged

---

## 📊 Database Records

### **Check Attendance Records**
```bash
# View recent entries
sqlite3 attendance.db "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 10;"

# Count entries per person
sqlite3 attendance.db "SELECT person_name, COUNT(*) FROM attendance GROUP BY person_name;"

# View today's records
sqlite3 attendance.db "SELECT * FROM attendance WHERE DATE(timestamp) = DATE('now');"
```

### **Check Known Faces**
```bash
# List all known persons
sqlite3 attendance.db "SELECT DISTINCT person_name, stable_id FROM face_embeddings;"

# Count embeddings per person
sqlite3 attendance.db "SELECT person_name, COUNT(*) FROM face_embeddings GROUP BY person_name;"
```

---

## 🎨 Visual Improvements

### **New Display Features:**
1. **Recognized persons:**
   - ✅ Thick green box (3px) around body
   - ✅ Name in large white text on green background
   - ✅ ID and confidence below the box
   - ✅ More prominent and easier to see

2. **Unknown persons:**
   - ⚠️ Thin orange box (2px)
   - ⚠️ "Unknown" label
   - ⚠️ Less prominent

3. **Entry/Exit counters:**
   - Green "Entry: X" at top left
   - Red "Exit: X" below it
   - "EfficientDet-D0" label at bottom

---

## 🚀 Quick Start

```bash
# 1. Test face recognition
python test_face_recognition.py

# 2. If faces recognized correctly, run full system
python attendance_efficientnetdet.py --source 0

# 3. Draw entry line (green) - press 's' to save
# 4. Draw exit line (red) - press 's' to save

# 5. Watch the magic happen:
#    - Bodies detected
#    - Faces recognized
#    - Names displayed
#    - Tracking persists
#    - Entry/exit logged
```

---

## ❓ Troubleshooting

### **Problem: Faces not recognized**
**Solution:** Lower similarity threshold
```bash
python attendance_efficientnetdet.py --source 0 --similarity-threshold 0.30
```

### **Problem: Wrong person recognized**
**Solution:** Increase similarity threshold
```bash
python attendance_efficientnetdet.py --source 0 --similarity-threshold 0.50
```

### **Problem: Name disappears when face turns away**
**Solution:** This should NOT happen! Check console for errors.
The system is designed to keep the name attached to the body.

### **Problem: No entry/exit events**
**Checklist:**
- ✅ Is person recognized? (green box with name?)
- ✅ Did person cross the line? (bottom-center of box)
- ✅ Is there a cooldown? (wait ~1 second between events)

---

**Status:** ✅ System ready to track known persons by full body with persistent identity!
