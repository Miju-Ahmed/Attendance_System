#!/usr/bin/env python3
"""
Simple RTSP Camera Test Script
Press 'q' to quit
"""

import cv2
import sys

# RTSP stream URL
RTSP_URL = "rtsp://admin:deepminD1@192.168.0.2:554/Streaming/Channels/101"

def main():
    print("=" * 60)
    print("RTSP Camera Test")
    print("=" * 60)
    print(f"Connecting to: {RTSP_URL}")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Open video stream
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ Failed to connect to RTSP stream")
        print("\nTroubleshooting:")
        print("1. Check if camera IP is correct: 192.168.0.2")
        print("2. Verify username/password: admin/deepminD1")
        print("3. Ensure camera is on the same network")
        print("4. Try pinging the camera: ping 192.168.0.2")
        sys.exit(1)
    
    print("✅ Connected successfully!")
    
    # Get stream properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nStream Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  Failed to read frame, reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            frame_count += 1
            
            # Add frame counter overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow("RTSP Camera Test", frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"✓ Frames received: {frame_count}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames received: {frame_count}")
        print("Done!")

if __name__ == "__main__":
    main()
