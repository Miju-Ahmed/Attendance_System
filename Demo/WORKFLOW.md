# Entry/Exit Attendance System - Workflow Documentation

## System Architecture

```mermaid
graph TB
    Start([Start System]) --> Init[Initialize Components]
    Init --> LoadModels[Load AI Models]
    LoadModels --> CheckGPU{GPU Available?}
    CheckGPU -->|Yes| GPUMode[Use CUDA Acceleration]
    CheckGPU -->|No| CPUMode[Use CPU Processing]
    GPUMode --> DrawLine[Interactive Line Drawing]
    CPUMode --> DrawLine
    DrawLine --> MainLoop[Main Processing Loop]
    
    MainLoop --> ReadFrame[Read Video Frame]
    ReadFrame --> SkipCheck{Skip Frame?}
    SkipCheck -->|Yes| Display1[Display Frame]
    SkipCheck -->|No| Resize[Resize Frame]
    
    Resize --> FaceDetect[SCRFD Face Detection]
    FaceDetect --> HasFaces{Faces Detected?}
    HasFaces -->|No| Display1
    HasFaces -->|Yes| ExtractEmbed[ArcFace Embeddings]
    
    ExtractEmbed --> DeepSORT[DeepSORT Tracking]
    DeepSORT --> MatchFaces[Match with Known Faces]
    MatchFaces --> AssignID[Assign Stable IDs]
    
    AssignID --> LineCheck[Line Crossing Detection]
    LineCheck --> CheckState{Check State Machine}
    CheckState -->|Entry| RecordEntry[Record Entry Event]
    CheckState -->|Exit| RecordExit[Record Exit Event]
    CheckState -->|No Event| Display2[Draw Visualizations]
    
    RecordEntry --> SaveDB[(Save to Database)]
    RecordExit --> SaveDB
    SaveDB --> Display2
    Display2 --> CheckQuit{Press 'q'?}
    
    CheckQuit -->|No| MainLoop
    CheckQuit -->|Yes| Cleanup[Cleanup Resources]
    Cleanup --> End([End])
```

## Detailed Processing Pipeline

```mermaid
flowchart LR
    subgraph Input
        Camera[Camera/Video] --> Frame[Raw Frame]
    end
    
    subgraph Preprocessing
        Frame --> Skip{Skip Frame?}
        Skip -->|Process| Resize[Resize to 800px]
        Skip -->|Skip| Track[Use Existing Tracks]
    end
    
    subgraph Detection
        Resize --> SCRFD[SCRFD Detector]
        SCRFD --> Faces[Face Bboxes + Landmarks]
    end
    
    subgraph Recognition
        Faces --> Align[Face Alignment]
        Align --> ArcFace[ArcFace Model]
        ArcFace --> Embeddings[512D Embeddings]
    end
    
    subgraph Tracking
        Embeddings --> DeepSORT[DeepSORT Tracker]
        Track --> DeepSORT
        DeepSORT --> Tracks[Track IDs]
    end
    
    subgraph Identification
        Tracks --> MEF[Multi-Embedding Fusion]
        MEF --> Compare[Compare with Known Faces]
        Compare --> Identity[Person Name + Stable ID]
    end
    
    subgraph Attendance
        Identity --> Position[Get Position]
        Position --> LineSide[Calculate Line Side]
        LineSide --> StateMachine{State Machine}
        StateMachine -->|OUTSIDE→INSIDE| Entry[Entry Event]
        StateMachine -->|INSIDE→OUTSIDE| Exit[Exit Event]
        Entry --> DB[(Database)]
        Exit --> DB
    end
    
    subgraph Output
        DB --> Visualize[Draw Bboxes + Labels]
        Visualize --> Display[Display Frame]
    end
```

## State Machine Diagram

```mermaid
stateDiagram-v2
    [*] --> Detecting: Person appears
    Detecting --> Recognized: Face matched
    Recognized --> OUTSIDE: Initialize state
    Recognized --> INSIDE: Initialize state
    
    OUTSIDE --> INSIDE: Cross line (Entry)
    INSIDE --> OUTSIDE: Cross line (Exit)
    
    OUTSIDE --> Cooldown1: Entry recorded
    INSIDE --> Cooldown2: Exit recorded
    
    Cooldown1 --> INSIDE: Wait 30 frames
    Cooldown2 --> OUTSIDE: Wait 30 frames
    
    INSIDE --> [*]: Person leaves (60 frames)
    OUTSIDE --> [*]: Person leaves (60 frames)
    Detecting --> [*]: Not recognized (10 frames)
```

## Component Interaction

```mermaid
graph LR
    subgraph Models
        SCRFD[SCRFD<br/>Face Detection]
        ArcFace[ArcFace<br/>Face Recognition]
    end
    
    subgraph Tracking
        DeepSORT[DeepSORT<br/>Multi-Object Tracker]
        MEF[MEF<br/>Embedding Fusion]
    end
    
    subgraph Logic
        KnownFaces[Known Faces<br/>Database]
        LineCross[Line Crossing<br/>Detector]
        StateMachine[State Machine<br/>INSIDE/OUTSIDE]
    end
    
    subgraph Storage
        SQLite[(SQLite<br/>Attendance DB)]
    end
    
    SCRFD --> DeepSORT
    ArcFace --> MEF
    MEF --> KnownFaces
    KnownFaces --> DeepSORT
    DeepSORT --> LineCross
    LineCross --> StateMachine
    StateMachine --> SQLite
```

## Data Flow

```mermaid
flowchart TD
    subgraph Frame Processing
        F1[Frame N] --> D1{Skip?}
        D1 -->|No| P1[Process]
        D1 -->|Yes| T1[Track Only]
        P1 --> Detect[Face Detection]
        Detect --> Embed[Extract Embeddings]
    end
    
    subgraph Tracking System
        Embed --> Track[Update Tracks]
        T1 --> Track
        Track --> Match[Match to Known]
        Match --> Stable[Assign Stable ID]
    end
    
    subgraph Attendance Logic
        Stable --> Pos[Calculate Position]
        Pos --> Side[Determine Side]
        Side --> State{Check State}
        State -->|Valid Transition| Event[Record Event]
        State -->|Invalid| Skip[Skip]
    end
    
    subgraph Persistence
        Event --> DB[(Database)]
        DB --> Log[Log Entry/Exit]
    end
```

## Performance Optimization Flow

```mermaid
graph TB
    Input[Input Frame<br/>1920x1080 @ 30 FPS]
    
    Input --> Opt1{Frame Skip}
    Opt1 -->|Process every 2nd| Half[15 FPS Processing]
    Opt1 -->|Skip| Display
    
    Half --> Opt2{Resize}
    Opt2 -->|Resize to 800px| Small[800x450 Processing]
    Opt2 -->|No resize| Full[Full Resolution]
    
    Small --> GPU{GPU Available?}
    Full --> GPU
    
    GPU -->|CUDA| Fast[Fast Processing<br/>~10ms/frame]
    GPU -->|CPU| Slow[Slow Processing<br/>~150ms/frame]
    
    Fast --> Track[DeepSORT Tracking]
    Slow --> Track
    
    Track --> Scale[Scale Back to Original]
    Scale --> Display[Display 1920x1080]
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `PROCESS_EVERY_N_FRAMES` | 2 | Skip frames for speed |
| `RESIZE_WIDTH` | 800 | Processing resolution |
| `FACE_CONFIDENCE` | 0.5 | Detection threshold |
| `FACE_SIMILARITY_THRESHOLD` | 0.40 | Recognition threshold |
| `MIN_TRACK_CONFIDENCE` | 2 | Detections before attendance |
| `ACTIVE_TRACK_GRACE` | 10 | Frames to keep unknown tracks |
| `RECOGNIZED_TRACK_GRACE` | 60 | Frames to keep known tracks |
| `TRACKER_MAX_AGE` | 30 | DeepSORT prediction window |
| `COOLDOWN_FRAMES` | 30 | Frames between entry/exit |

## Database Schema

```
attendance
├── id (INTEGER PRIMARY KEY)
├── person_name (TEXT)
├── event_type (TEXT: 'ENTRY' or 'EXIT')
├── timestamp (DATETIME)
├── confidence (REAL)
└── track_id (INTEGER: Stable ID)
```
