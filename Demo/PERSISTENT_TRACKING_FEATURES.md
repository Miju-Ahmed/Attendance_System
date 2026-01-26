# Persistent Tracking Features - entry_exit_attendance_v2.py

## Overview
The enhanced `entry_exit_attendance_v2.py` script now includes **persistent face tracking** functionality that maintains person identity even during temporary detection failures or occlusions.

## Key Features Added

### 1. **Persistent Identity Tracking**
Once a person's face is recognized, the system maintains their identity even if they temporarily become unrecognized:
- **Confirmed Identity**: After initial recognition, the system stores a `confirmed_name` for each track
- **Identity Persistence**: If a face becomes temporarily unrecognized (e.g., due to poor angle or lighting), the system maintains the last confirmed identity for up to 2 seconds (60 frames)
- **Best Confidence Tracking**: Stores the highest confidence score achieved for each person

### 2. **Embedding-Based Re-identification**
Advanced matching algorithm that uses both spatial proximity and facial embeddings:
- **Dual Matching**: Combines spatial distance (60% weight) and embedding similarity (40% weight)
- **Lost Track Recovery**: Can re-identify persons who temporarily left the frame and returned
- **Embedding History**: Maintains a buffer of the last 10 embeddings per track for robust matching
- **Average Embedding**: Uses averaged embeddings for more stable re-identification

### 3. **Track History & Movement Tracking**
Comprehensive tracking of person movement and behavior:
- **Position History**: Stores last 30 positions for movement trail visualization
- **Embedding History**: Maintains last 10 face embeddings for better matching
- **Detection Count**: Tracks total number of times a person was detected
- **Identification Rate**: Monitors how often the person was successfully identified

### 4. **Enhanced Entry/Exit Calculation**
Improved entry/exit event tracking with detailed information:
- **Duration Tracking**: Calculates how long each person stayed inside
- **Event History**: Records all entry/exit events with timestamps and confidence scores
- **State Machine**: Prevents duplicate entries/exits with INSIDE/OUTSIDE state tracking
- **Quality Metrics**: Tracks tracking quality score (0-1) based on identification rate and confidence

### 5. **Comprehensive Statistics**
Detailed per-person statistics displayed at the end:
- **Total Tracks Created**: Number of unique tracks generated
- **Persistent Tracks**: Tracks with confirmed identities
- **Per-Person Metrics**:
  - Track IDs associated with each person
  - Total number of entries and exits
  - Total time spent inside
  - Current state (INSIDE/OUTSIDE)
  - Tracking quality score

### 6. **Enhanced Visualization**
Improved on-screen display with more information:
- **Persistent Track Indicator**: ★ symbol for confirmed identities
- **Color Coding**:
  - Green: Persistent tracks (confirmed identity)
  - Yellow-green: New identifications
  - Orange: Unknown faces
- **State Markers**: 🏠 (inside) or 🚪 (outside)
- **Quality Bar**: Visual indicator of tracking quality below each face box
- **Movement Trail**: Shows recent movement path for each tracked person
- **Label Background**: Improved readability with background boxes

## Technical Implementation

### FaceTrack Class Enhancements
```python
# New attributes added:
- confirmed_name: Persistent identity once confirmed
- best_confidence: Highest confidence score achieved
- position_history: Last 30 positions (deque)
- embedding_history: Last 10 embeddings (deque)
- frames_since_identified: Counter for persistence timeout
- total_identifications: Total successful identifications
- is_persistent: Flag for confirmed identity
- entry_events: List of (timestamp, confidence) tuples
- exit_events: List of (timestamp, confidence) tuples
- first_seen_time: When track was created
- last_event_time: Last entry/exit event time
```

### New Methods
```python
# FaceTrack methods:
- get_average_embedding(): Returns averaged embedding from history
- record_entry(timestamp, confidence): Records entry event
- record_exit(timestamp, confidence): Records exit event
- get_tracking_quality(): Calculates quality score (0-1)

# FaceBasedAttendanceSystem methods:
- get_tracking_statistics(): Returns detailed stats dictionary
- print_tracking_summary(): Prints comprehensive summary
```

### Association Algorithm
The new `associate_detections_to_tracks()` method uses a two-pass approach:

**Pass 1: Recent Tracks**
- Computes spatial distances between detections and recent tracks
- Calculates embedding similarities using cosine distance
- Combines both metrics with weighted average (60% spatial, 40% embedding)
- Performs greedy matching with combined cost

**Pass 2: Lost Track Recovery**
- Attempts to match unmatched detections with lost tracks
- Uses embedding similarity and name matching
- Can recover tracks lost for up to 3 seconds (90 frames)

## Usage

The script works exactly the same as before, but with enhanced tracking:

```bash
# Run with webcam
python entry_exit_attendance_v2.py

# Run with video file
python entry_exit_attendance_v2.py --source /path/to/video.mp4

# Save output video
python entry_exit_attendance_v2.py --source video.mp4 --output output.mp4
```

## Benefits

1. **Robustness**: Maintains identity through temporary occlusions or poor angles
2. **Accuracy**: Better re-identification using embedding similarity
3. **Insights**: Detailed per-person statistics including duration inside
4. **Debugging**: Quality metrics help identify tracking issues
5. **Visualization**: Enhanced display makes it easier to monitor system performance

## Example Output

```
============================================================
DETAILED TRACKING SUMMARY
============================================================
Total Tracks Created: 5
Active Tracks: 3
Persistent Tracks: 2
Total Entries: 4
Total Exits: 3
Total Frames: 1250
------------------------------------------------------------
PER-PERSON STATISTICS:
------------------------------------------------------------

👤 John_Doe:
   Track IDs: [1, 4]
   Entries: 2
   Exits: 2
   Total Time Inside: 45.3s
   Current State: OUTSIDE
   Tracking Quality: 0.92

👤 Jane_Smith:
   Track IDs: [2]
   Entries: 2
   Exits: 1
   Total Time Inside: 23.7s
   Current State: INSIDE
   Tracking Quality: 0.88
============================================================
```

## Configuration Parameters

Key parameters that control persistent tracking:

```python
# In entry_exit_attendance_v2.py
MAX_TRACK_DISTANCE = 100  # Max pixels for spatial matching
MIN_TRACK_CONFIDENCE = 3  # Min detections before recording (lowered for persistent tracks)
MEF_BUFFER_SIZE = 5  # Embeddings for Multi-Embedding Fusion
COOLDOWN_FRAMES = 30  # Frames between entry/exit events

# In FaceTrack.update()
frames_since_identified < 60  # 2 seconds tolerance for identity persistence

# In associate_detections_to_tracks()
last_seen_frame < 30  # Recent tracks (1 second)
last_seen_frame < 90  # Lost track recovery (3 seconds)
embedding_similarity > 0.5  # Threshold for re-identification
```

## Notes

- The system automatically handles track creation and deletion
- Old tracks (not seen for 60 frames) are automatically cleaned up
- Persistent identity is maintained even if confidence drops temporarily
- Entry/exit events are only recorded for persistent tracks with confirmed identities
- The quality score helps identify problematic tracks that may need manual review
