#!/bin/bash

# Quick start script for RTSP camera with optimized performance

RTSP_URL="rtsp://admin:deepminD1@192.168.0.2:554/Streaming/Channels/101"

echo "=========================================="
echo "Entry/Exit Attendance - RTSP Camera"
echo "=========================================="
echo ""
echo "Camera: $RTSP_URL"
echo "Performance: Optimized for real-time"
echo ""
echo "Instructions:"
echo "1. Draw the red line (2 points)"
echo "2. Press 's' to start tracking"
echo "3. Press 'q' to quit"
echo ""
echo "=========================================="
echo ""

# Run with balanced settings (good detection + good performance)
python3 entry_exit_attendance_v3.py \
    --source "$RTSP_URL" \
    --process-every-n 2 \
    --resize-width 800

echo ""
echo "Done!"
