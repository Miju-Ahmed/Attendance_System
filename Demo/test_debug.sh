#!/bin/bash

# Quick test script with debug logging enabled
# This will help diagnose why entry/exit detection isn't working

echo "=========================================="
echo "Entry/Exit Detection Debug Test"
echo "=========================================="
echo ""

# Find video file
VIDEO_FILE=""
if [ -f "domain/miju.mp4" ]; then
    VIDEO_FILE="domain/miju.mp4"
elif [ -f "outputs/*.mp4" ]; then
    VIDEO_FILE=$(ls outputs/*.mp4 | head -1)
else
    echo "Please specify video file path:"
    read VIDEO_FILE
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Video file not found: $VIDEO_FILE"
    exit 1
fi

echo "Using video: $VIDEO_FILE"
echo ""
echo "Instructions:"
echo "1. Draw the red line (2 points)"
echo "2. Press 's' to start"
echo "3. Watch the terminal for debug logs"
echo "4. Press 'q' to quit"
echo ""
echo "=========================================="
echo ""

# Run with debug logging (already enabled in the code)
python3 entry_exit_attendance_v3.py \
    --source "$VIDEO_FILE" \
    --loop-video \
    2>&1 | tee debug_output.log

echo ""
echo "=========================================="
echo "Debug log saved to: debug_output.log"
echo "=========================================="
echo ""
echo "Checking for key events in log..."
grep -E "(Initialized|ENTERED|EXITED|Skipped|Processing line crossing)" debug_output.log | tail -20
