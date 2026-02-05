#!/usr/bin/env python3
"""
Diagnostic script to help debug entry/exit tracking issues
"""
import os
import sqlite3

print("=" * 80)
print("ENTRY/EXIT TRACKING DIAGNOSTIC")
print("=" * 80)

# Check if required files exist
print("\n1. Checking Required Files:")
print("-" * 80)

files_to_check = [
    ("v4.py", "Main script"),
    ("attendance.db", "Database"),
    ("known_faces/", "Known faces directory"),
    ("weights/face_detection/det_10g.onnx", "Face detection model"),
    ("weights/face_recognition/w600k_r50.onnx", "Face recognition model"),
]

all_good = True
for file_path, description in files_to_check:
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {file_path}")
    if not exists:
        all_good = False

# Check known faces
print("\n2. Checking Known Faces:")
print("-" * 80)

if os.path.exists("known_faces/"):
    npy_files = [f for f in os.listdir("known_faces/") if f.endswith('.npy')]
    if npy_files:
        print(f"  ✅ Found {len(npy_files)} known face(s):")
        for npy_file in npy_files:
            name = npy_file.replace('.npy', '')
            print(f"     - {name}")
    else:
        print("  ❌ No .npy files found in known_faces/")
        print("     You need to add face embeddings for recognition!")
        all_good = False
else:
    print("  ❌ known_faces/ directory not found")
    all_good = False

# Check database
print("\n3. Checking Database:")
print("-" * 80)

if os.path.exists("attendance.db"):
    try:
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance")
        count = cursor.fetchone()[0]
        print(f"  ✅ Database connected")
        print(f"  📊 Current records: {count}")
        conn.close()
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        all_good = False
else:
    print("  ⚠️  Database doesn't exist yet (will be created on first run)")

# Provide guidance
print("\n" + "=" * 80)
print("TESTING GUIDANCE")
print("=" * 80)

if not all_good:
    print("\n⚠️  Some requirements are missing. Please fix the issues above first.\n")
else:
    print("""
✅ All requirements met! Here's how to test:

1. Run the system:
   python v4.py --source <video_file_or_0_for_webcam>

2. Draw the entry/exit line when prompted

3. Watch the console logs for these key messages:

   📍 Position tracking (shows person position and which side of line)
   🔍 Crossing detection (shows when someone crosses the line)
   ✅ Entry recorded (person entered)
   ❌ Exit recorded (person exited)

4. Common issues to check:

   a) Person not recognized?
      - Make sure their face is in known_faces/ as .npy file
      - Check logs for "✨ New person tracked: <name>"
   
   b) Person recognized but no crossing detected?
      - Check the 📍 logs to see their position and side
      - Make sure they actually cross the line (side changes from -1 to 1 or vice versa)
      - Check for "⏳ In cooldown" messages (30 frame cooldown after each event)
   
   c) Crossing detected but not recorded?
      - Check for "⚠️  already INSIDE/OUTSIDE" messages
      - This means the state machine prevented duplicate recording
      - The person needs to be in the correct state for the event to record

5. After testing, check the database:
   python view_db.py

6. To enable even more detailed logging, edit v4.py line 80:
   Change: level=logging.INFO
   To:     level=logging.DEBUG
""")

print("=" * 80)
