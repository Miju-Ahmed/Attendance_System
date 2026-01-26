#!/bin/bash

# Entry/Exit Attendance System - Quick Start Script
# ==================================================

echo "==========================================="
echo "Entry/Exit Attendance System"
echo "==========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "It's recommended to use a virtual environment"
    echo ""
fi

# Check for required models
echo "Checking for required models..."

if [ ! -f "weights/face_detection/det_10g.onnx" ]; then
    echo "❌ Face detection model not found: weights/face_detection/det_10g.onnx"
    exit 1
fi

if [ ! -f "weights/face_recognition/w600k_r50.onnx" ]; then
    echo "❌ Face recognition model not found: weights/face_recognition/w600k_r50.onnx"
    exit 1
fi

echo "✅ Models found"
echo ""

# Check for known faces
echo "Checking for known faces..."
if [ ! -d "known_faces" ]; then
    echo "❌ known_faces directory not found"
    exit 1
fi

num_faces=$(ls -1 known_faces/*.npy 2>/dev/null | wc -l)
if [ "$num_faces" -eq 0 ]; then
    echo "⚠️  Warning: No .npy face embeddings found in known_faces/"
    echo "The system will track people but won't recognize them"
else
    echo "✅ Found $num_faces known face(s)"
fi
echo ""

# Ask for video source
echo "Select video source:"
echo "1) Webcam (default)"
echo "2) Video file"
echo ""
read -p "Enter choice (1 or 2): " choice

SOURCE="0"
LOOP_ARG=""
if [ "$choice" == "2" ]; then
    read -p "Enter video file path: " video_path
    if [ ! -f "$video_path" ]; then
        echo "❌ Video file not found: $video_path"
        exit 1
    fi
    SOURCE="$video_path"
    LOOP_ARG="--loop-video"
fi

# Ask for output video
read -p "Save output video? (y/N): " save_output
OUTPUT_ARG=""
if [[ "$save_output" =~ ^[Yy]$ ]]; then
    mkdir -p outputs
    OUTPUT_FILE="output_attendance_$(date +%Y%m%d_%H%M%S).mp4"
    OUTPUT_ARG="--output outputs/$OUTPUT_FILE"
    echo "Will save to: outputs/$OUTPUT_FILE"
fi

echo ""
echo "==========================================="
echo "Starting Attendance System..."
echo "==========================================="
echo ""
echo "Instructions:"
echo "1. Draw a red line by clicking 2 points"
echo "2. Press 's' to start tracking"
echo "3. Press 'q' to quit"
echo ""
echo "==========================================="
echo ""

# Run the system
python entry_exit_attendance_v3.py \
    --source "$SOURCE" \
    $LOOP_ARG \
    $OUTPUT_ARG

echo ""
echo "==========================================="
echo "Attendance tracking completed"
echo "==========================================="
echo ""

# Ask if user wants to view attendance
read -p "View attendance records? (y/N): " view_records
if [[ "$view_records" =~ ^[Yy]$ ]]; then
    echo ""
    python view_attendance.py --summary
fi

echo ""
echo "Done!"
