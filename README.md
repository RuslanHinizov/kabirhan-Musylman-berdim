# Race Vision

Real-time horse race position tracking system using computer vision. Detects jockey colors from camera feeds and displays live race standings on a web dashboard.

## Architecture

```
Analytics Cameras (RTSP/video)          PTZ Cameras (broadcast)
         |                                       |
   api/server.py                           PublicDisplay
   (YOLO + CNN detection)                 (viewers see clean feed)
         |                                       |
         +--- MJPEG /stream/cam{N} ---> CameraGrid (operator only)
         |
         +--- WebSocket /ws ----------> Frontend (React)
              (rankings, positions)        |
                                     OperatorPanel + PublicDisplay
                                     (animated jockey icons)
```

**Two camera types, strict separation:**
- **Analytics cameras** — processed by backend (YOLO + CNN). Video shown **only** in operator's CameraGrid. Never visible to public.
- **PTZ cameras** — broadcast-quality feed for public display. No detection overlays.

Analytics affects PublicDisplay **only** through ranking data (icons + positions), never through video.

## Detection Pipeline

1. **YOLOv8s** detects persons in each frame
2. **SimpleColorCNN** classifies torso crop into 5 colors: red, blue, green, yellow, purple
3. **4-layer filtering** removes false detections:
   - F1: Classifier confidence (>= 0.75)
   - F2: CNN + HSV color agreement
   - F3: Speed constraint (no teleportation)
   - F4: Temporal confirmation (2/5 frame window)
4. **Per-video voting**: filtered complete frames vote on X-sorted order
5. Best order sent to frontend when each video ends

## Quick Start

### Backend
```bash
# With video files (sequential playback, loops)
.venv\Scripts\python.exe api/server.py --video data/videos/exp10_cam1.mp4 data/videos/exp10_cam2.mp4 data/videos/exp10_cam3.mp4 --auto-start

# With RTSP stream (GPU decode)
.venv\Scripts\python.exe api/server.py --url "rtsp://admin:pass@ip:554/stream" --gpu
```

### Frontend
```bash
cd Kabirhan-Frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173/` — Public display (viewers)
- `http://localhost:5173/operator` — Operator panel (race control)

## Project Structure

```
race_vision/
  api/
    server.py                  # FastAPI backend (WebSocket + MJPEG + detection)
  Kabirhan-Frontend/           # React frontend (Vite + Zustand + Framer Motion)
    src/
      pages/
        PublicDisplay.tsx       # Viewer page: PTZ video + animated ranking bar
        OperatorPanel.tsx       # Operator page: cameras, track, settings
      components/
        operator/
          CameraGrid.tsx        # Analytics camera grid (3 cameras, MJPEG)
          CameraSettings.tsx    # Camera configuration (analytics + PTZ)
          PTZControlPanel.tsx   # PTZ camera selection + preview
          Track2DView.tsx       # 2D oval track visualization
          RaceSettings.tsx      # Race config form
        public-display/
          PTZCameraDisplay.tsx  # Active PTZ camera feed
          RankingBoard.tsx      # Horse standings table
        MJPEGPlayer.tsx         # MJPEG stream player component
      config/
        cameras.ts             # Camera configs: ANALYTICS_CAMERAS + PTZ_CAMERAS
      store/
        raceStore.ts           # Race state + horse rankings (Zustand)
        cameraStore.ts         # Camera state: analytics + PTZ (Zustand)
      services/
        backendConnection.ts   # WebSocket client (auto-reconnect, heartbeat)
    public/assets/silks/       # Jockey silk SVG icons (silk_1..silk_10)
  tools/
    test_race_count.py         # Detection pipeline (RaceTracker, ColorClassifier)
    test_rtsp.py               # RTSP/video reader (FFmpegReader)
    train_color_classifier.py  # CNN training script
    archive/                   # Old/unused scripts
  models/
    color_classifier.pt        # Trained CNN (5 classes, 64x64 input)
    yolov8s.pt                 # Person detection model
  data/
    videos/                    # Race video files (exp10_cam1..cam3)
    torso_crops_new/           # Training data for classifier
  results/                     # Output videos and logs
```

## Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `ws://localhost:8000/ws` | WebSocket | Real-time ranking updates + race control |
| `http://localhost:8000/stream/cam{1-3}` | MJPEG | Per-camera annotated video stream |

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | `rtsp://...` | RTSP stream URL |
| `--video` | — | Local video file(s), played sequentially |
| `--gpu` | off | Enable GPU decode (hevc_cuvid) |
| `--host` | `0.0.0.0` | Server bind host |
| `--port` | `8000` | Server bind port |
| `--auto-start` | off | Auto-start race on launch |

## Requirements

- Python 3.9+ with CUDA GPU
- Node.js 18+
- `pip install fastapi uvicorn[standard] ultralytics torch torchvision opencv-python`
