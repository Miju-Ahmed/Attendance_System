#!/usr/bin/env python3
"""
Test Face-Based Attendance System
===================================
Simpler, faster system - NO YOLO, just pure face detection!
"""

import sys
import os
from pathlib import Path

# Available test videos
test_videos = [
    "/run/media/miju_chowdhury/Miju/WorkSpace/Presentation_Slide/Week_51/id_face_logo_2.mp4",
    "/run/media/miju_chowdhury/Miju/WorkSpace/Apex_All_Project_Demo/People-counting-system_good/input/merged_trimmed_D01_60609_3.mp4",
]

print("="*70)
print("🎭 FACE-BASED Attendance System (NO YOLO - Faster!)")
print("="*70)
print()
print("✨ What's New:")
print("  - NO YOLO person detection needed")
print("  - Direct face detection with SCRFD")
print("  - Faster and simpler")
print("  - Tracks faces directly across the line")
print()
print("="*70)
print()

# Find available videos
available = []
for video in test_videos:
    if os.path.exists(video):
        size_mb = os.path.getsize(video) / (1024*1024)
        print(f"✅ {Path(video).name} ({size_mb:.1f} MB)")
        available.append(video)

if not available:
    print("❌ No test videos found!")
    print("Please provide a video file:")
    print("  python entry_exit_attendance.py --source /path/to/video.mp4")
    sys.exit(1)

print()
print("="*70)
print(f"Testing with: {Path(available[0]).name}")
print("="*70)
print()

# Run the system
import subprocess

cmd = [
    "python", "entry_exit_attendance.py",
    "--source", available[0],
    "--output", "face_attendance_output.mp4"
]

print("Command:")
print(" ".join(cmd))
print()
print("="*70)
print("🎬 Instructions:")
print("  1. Draw red line (2 clicks)")
print("  2. Press 's' to start")
print("  3. Watch faces being tracked!")
print("  4. Press 'q' to quit")
print("="*70)
print()

try:
    subprocess.run(cmd)
    print("\n✅ Processing complete!")
    print("\nView results:")
    print("  python view_attendance.py --summary")
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
