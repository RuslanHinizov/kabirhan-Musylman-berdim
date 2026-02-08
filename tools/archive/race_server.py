"""
Race Vision — FastAPI Backend Server

Bridges the detection pipeline (YOLO + ColorClassifier) with the React frontend
via WebSocket (ranking updates) and MJPEG (annotated video streams).

Architecture:
    RTSP → FrameGrabber thread → SharedState.frame
                                      ↓
                DetectionLoop thread → SharedState.rankings + annotated_frame
                     ↓                    ↓
           WebSocket /ws          MJPEG /stream/cam{1-4}
           (ranking_update)       (multipart/x-mixed-replace)

Usage:
    python tools/race_server.py
    python tools/race_server.py --url "rtsp://admin:pass@ip:554/stream"
    python tools/race_server.py --video data/videos/exp10_cam1.mp4
    python tools/race_server.py --gpu
"""

import cv2
import sys
import time
import json
import asyncio
import logging
import argparse
import threading
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Import from our existing modules
from tools.test_race_count import (
    RaceTracker, ColorClassifier, SimpleColorCNN, draw,
    COLORS_BGR, REQUIRED_COLORS
)
from tools.test_rtsp import (
    FFmpegReader, FFMPEG, detect_codec, parse_resolution
)

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554//stream"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Detection loop rate (seconds between updates)
DETECTION_INTERVAL = 0.10  # ~10 fps

# WebSocket broadcast rate
BROADCAST_INTERVAL = 0.20  # 5 Hz

# MJPEG settings
MJPEG_QUALITY = 75
MJPEG_FPS = 25

# Track mapping: pixel X range → distanceCovered (0-2500)
TRACK_LENGTH = 2500

# ============================================================
# COLOR → HORSE MAPPING (matches frontend SILK_COLORS)
# ============================================================

COLOR_TO_HORSE = {
    "red":    {"id": "horse-1", "number": 1, "name": "Red Runner",     "silkId": 1, "color": "#DC2626", "jockeyName": "Jockey 1"},
    "blue":   {"id": "horse-2", "number": 2, "name": "Blue Storm",     "silkId": 2, "color": "#2563EB", "jockeyName": "Jockey 2"},
    "green":  {"id": "horse-3", "number": 3, "name": "Green Flash",    "silkId": 3, "color": "#16A34A", "jockeyName": "Jockey 3"},
    "yellow": {"id": "horse-4", "number": 4, "name": "Yellow Thunder", "silkId": 4, "color": "#FBBF24", "jockeyName": "Jockey 4"},
    "purple": {"id": "horse-5", "number": 5, "name": "Purple Reign",   "silkId": 5, "color": "#9333EA", "jockeyName": "Jockey 5"},
}

ALL_COLORS = list(COLOR_TO_HORSE.keys())

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("race_server")

# ============================================================
# SHARED STATE
# ============================================================

class SharedState:
    """Thread-safe state shared between grabber, detector, and server."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.jockeys: list = []          # Latest detection result
        self.rankings: list = []         # Formatted for frontend (last non-empty)
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.race_active: bool = False
        self.detection_fps: float = 0.0
        self.detection_count: int = 0    # Total frames processed
        self.video_index: int = 0        # Which video is playing (incremented by grabber)

    def set_frame(self, frame: np.ndarray):
        with self._lock:
            self.frame = frame
            self.frame_height, self.frame_width = frame.shape[:2]

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def set_detection_result(self, jockeys: list, annotated: np.ndarray, rankings: list, fps: float):
        with self._lock:
            self.jockeys = jockeys
            self.annotated_frame = annotated
            # Only update rankings if we got detections (keep last good result)
            if rankings:
                self.rankings = rankings
            self.detection_fps = fps
            self.detection_count += 1

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.annotated_frame.copy() if self.annotated_frame is not None else None

    def get_rankings(self) -> list:
        with self._lock:
            return list(self.rankings)

    def get_frame_dimensions(self) -> tuple:
        with self._lock:
            return self.frame_width, self.frame_height


state = SharedState()

# ============================================================
# FRAME GRABBER THREAD
# ============================================================

class FrameGrabber(threading.Thread):
    """Reads frames from RTSP or video file(s) in a background thread."""

    def __init__(self, sources: list, use_gpu: bool = False, is_rtsp: bool = True):
        super().__init__(daemon=True)
        self.sources = sources  # list of paths/urls
        self.use_gpu = use_gpu
        self.is_rtsp = is_rtsp
        self.running = False
        self._reader = None
        self._cap = None

    def run(self):
        self.running = True

        if self.is_rtsp:
            self._run_rtsp()
        else:
            self._run_video()

    def _run_rtsp(self):
        """Read from RTSP using FFmpegReader with optional GPU decode."""
        source = self.sources[0]
        while self.running:
            try:
                log.info(f"Connecting to RTSP: {source.split('@')[-1] if '@' in source else source}")

                # Detect codec
                codec = detect_codec(source)
                log.info(f"Detected codec: {codec}")

                # Get resolution
                w, h = parse_resolution(source)
                if w == 0:
                    log.error("Could not detect stream resolution, retrying in 5s...")
                    time.sleep(5)
                    continue

                log.info(f"Stream resolution: {w}x{h}, GPU: {self.use_gpu}")

                reader = FFmpegReader(source, w, h, gpu=self.use_gpu, codec=codec)
                reader.start()
                self._reader = reader

                frame_count = 0
                while self.running:
                    ret, frame = reader.read()
                    if not ret:
                        log.warning("RTSP frame read failed, reconnecting...")
                        break
                    state.set_frame(frame)
                    frame_count += 1

                reader.release()
                self._reader = None

            except Exception as e:
                log.error(f"RTSP error: {e}, retrying in 5s...")

            if self.running:
                time.sleep(5)

    def _run_video(self):
        """Read from local video files sequentially, then loop."""
        while self.running:
            for source in self.sources:
                if not self.running:
                    break

                cap = cv2.VideoCapture(source)
                self._cap = cap

                if not cap.isOpened():
                    log.error(f"Cannot open video: {source}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                delay = 1.0 / fps
                name = Path(source).stem

                # Signal video switch to detection loop
                state.video_index += 1
                log.info(f"Playing: {name} @ {fps:.0f} fps (video #{state.video_index})")

                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        log.info(f"{name} ended")
                        break
                    state.set_frame(frame)
                    time.sleep(delay)

                cap.release()
                self._cap = None

            if self.running:
                log.info("All videos played, looping...")

    def stop(self):
        self.running = False
        if self._reader:
            self._reader.release()
        if self._cap:
            self._cap.release()


# ============================================================
# DETECTION LOOP THREAD
# ============================================================

class DetectionLoop(threading.Thread):
    """Runs YOLO + ColorClassifier on frames with 4-layer filtering.

    Per-video logic:
    1. Every detection passes 4 filters:
       F1: Classifier confidence >= CONF_THRESHOLD
       F2: CNN color == HSV color agreement (skip if CNN conf > HSV_SKIP_CONF)
       F3: Speed constraint — reject teleportation
       F4: Temporal confirmation — color seen >= TEMPORAL_MIN in last TEMPORAL_WINDOW frames
    2. Filtered complete frames (all 5 colors) → X-sorted order voted per position
    3. At video end: most-voted order wins → send to frontend
    """

    # --- Filter thresholds ---
    CONF_THRESHOLD = 0.75       # F1: min classifier confidence
    HSV_SKIP_CONF = 0.92        # F2: skip HSV check if CNN conf > this
    MAX_SPEED_MPS = 120.0       # F3: generous for pixel noise between frames
    TEMPORAL_WINDOW = 5          # F4: sliding window (frames)
    TEMPORAL_MIN = 2             # F4: need 2/5 frames confirmed
    CAMERA_TRACK_M = 100.0       # each camera = 0..100 meters

    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.tracker: Optional[RaceTracker] = None
        self._current_video_index = 0

        # Current stable order for frontend: list of color names, 1st first
        self._current_order: list = []
        # Smoothed center_x per color (EMA) for frontend visualization
        self._smooth_x: dict = {}
        self._smooth_alpha = 0.12

        # --- Per-color tracking with filters ---
        self._det_frames: dict = {}    # color -> list of frame numbers (sliding window)
        self._last_pos: dict = {}      # color -> (pos_m, timestamp) for speed check

        # Per-video voting: collect X-sorted orders from filtered complete frames
        self._video_votes: list = []   # list of order tuples

        # Metrics
        self._order_changes = 0
        self._total_frames = 0
        self._start_time = 0.0
        self._filter_stats = {'total': 0, 'f1_low_conf': 0, 'f2_hsv_mismatch': 0,
                              'f3_speed': 0, 'f4_temporal': 0, 'accepted': 0}

    def run(self):
        self.running = True
        self._start_time = time.time()

        output_dir = Path("results/race_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = RaceTracker(output_dir, save_crops=False)
        log.info("Detection pipeline ready")

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        while self.running:
            frame = state.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            if not state.race_active:
                rankings = self._build_rankings(frame.shape[1])
                state.set_detection_result([], frame.copy(), rankings, 0.0)
                time.sleep(0.1)
                continue

            # Video switch → finalize previous video, reset
            if state.video_index != self._current_video_index:
                self._on_video_switch(state.video_index)

            t0 = time.time()
            jockeys, detections = self.tracker.update(frame)
            self._total_frames += 1
            frame_width = frame.shape[1]

            # Draw annotations on MJPEG
            annotated = draw(frame.copy(), jockeys, self.tracker)

            # Update smoothed positions (unfiltered, for visualization only)
            for j in jockeys:
                color = j['color']
                cx = float(j['center_x'])
                if color in self._smooth_x:
                    self._smooth_x[color] += self._smooth_alpha * (cx - self._smooth_x[color])
                else:
                    self._smooth_x[color] = cx

            # === 4-LAYER FILTERING on raw detections ===
            filtered_dets = []  # detections that pass all 4 filters
            for det in detections:
                color = det['color']
                conf = float(det.get('conf', 0.0))
                hsv = det.get('hsv_guess', '')
                cx = float(det['center_x'])
                pos_m = (cx / max(frame_width, 1)) * self.CAMERA_TRACK_M
                self._filter_stats['total'] += 1

                # F1: Confidence threshold
                if conf < self.CONF_THRESHOLD:
                    self._filter_stats['f1_low_conf'] += 1
                    continue

                # F2: CNN + HSV agreement (skip if CNN very confident)
                if hsv and hsv != color and conf < self.HSV_SKIP_CONF:
                    self._filter_stats['f2_hsv_mismatch'] += 1
                    continue

                # F3: Speed constraint (skip if first detection for this color)
                if color in self._last_pos:
                    last_pos_m, last_time = self._last_pos[color]
                    dt_sec = time.time() - last_time
                    if dt_sec > 0.01:
                        speed = abs(pos_m - last_pos_m) / dt_sec
                        if speed > self.MAX_SPEED_MPS:
                            self._filter_stats['f3_speed'] += 1
                            continue

                # Passed F1+F2+F3 → update position + sliding window
                self._last_pos[color] = (pos_m, time.time())
                if color not in self._det_frames:
                    self._det_frames[color] = []
                self._det_frames[color].append(self._total_frames)
                if len(self._det_frames[color]) > self.TEMPORAL_WINDOW:
                    self._det_frames[color] = self._det_frames[color][-self.TEMPORAL_WINDOW:]

                # F4: Temporal confirmation
                recent_count = sum(
                    1 for f in self._det_frames[color]
                    if f > self._total_frames - self.TEMPORAL_WINDOW
                )
                if recent_count < self.TEMPORAL_MIN:
                    self._filter_stats['f4_temporal'] += 1
                    continue

                self._filter_stats['accepted'] += 1
                filtered_dets.append(det)

            # If 4+ colors passed filters → vote this frame's X-sorted order
            filtered_colors = set(d['color'] for d in filtered_dets)
            MIN_VOTE_COLORS = 4
            if len(filtered_colors) >= MIN_VOTE_COLORS:
                # Sort by -center_x (rightmost = 1st) — same as tracker
                sorted_dets = sorted(filtered_dets, key=lambda d: -d['center_x'])
                # Deduplicate: keep first occurrence of each color
                seen = set()
                order = []
                for d in sorted_dets:
                    if d['color'] not in seen:
                        seen.add(d['color'])
                        order.append(d['color'])
                # Append missing colors at the end
                all_colors = {'red', 'blue', 'green', 'yellow', 'purple'}
                for c in all_colors - seen:
                    order.append(c)
                self._video_votes.append(tuple(order))

            # Build rankings from current order + smoothed positions
            rankings = self._build_rankings(frame_width)

            # FPS
            fps_counter += 1
            elapsed_fps = time.time() - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.time()

            state.set_detection_result(jockeys, annotated, rankings, current_fps)

            # Rate limit
            dt = time.time() - t0
            sleep_time = DETECTION_INTERVAL - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _on_video_switch(self, new_index: int):
        """At video end: pick most-voted order → update → reset."""
        old_idx = self._current_video_index
        self._current_video_index = new_index

        elapsed = time.time() - self._start_time
        s = self._filter_stats
        log.info(
            f"[VIDEO END] #{old_idx} at t={elapsed:.1f}s | "
            f"frames={self._total_frames} voted_frames={len(self._video_votes)}"
        )
        log.info(
            f"  Filters: total={s['total']} → "
            f"F1={s['f1_low_conf']} F2={s['f2_hsv_mismatch']} "
            f"F3={s['f3_speed']} F4={s['f4_temporal']} "
            f"→ accepted={s['accepted']}"
        )

        # Pick most-voted order
        if self._video_votes:
            vote_counts = Counter(self._video_votes)
            for order, cnt in vote_counts.most_common():
                log.info(f"    {' > '.join(c.upper()[:3] for c in order)}: {cnt}x")

            best_order, best_count = vote_counts.most_common(1)[0]
            new_order = list(best_order)

            if new_order != self._current_order:
                old_order = self._current_order
                self._current_order = new_order
                self._order_changes += 1
                log.info(
                    f"[ORDER CHANGE #{self._order_changes}] "
                    f"{' > '.join(c.upper()[:3] for c in new_order)} "
                    f"({best_count}/{len(self._video_votes)} votes)"
                )
                if old_order:
                    changes = []
                    for i, (o, n) in enumerate(zip(old_order, new_order)):
                        if o != n:
                            changes.append(f"P{i+1}: {o[:3]}→{n[:3]}")
                    if changes:
                        log.info(f"  Changes: {', '.join(changes)}")
            else:
                log.info(f"  Same order as before, no change")
        else:
            log.info(f"  No voted frames, keeping previous order")

        # Reset per-video state
        self._video_votes.clear()
        self._det_frames.clear()
        self._last_pos.clear()
        self._filter_stats = {k: 0 for k in self._filter_stats}

        # Reset tracker for next video
        if self.tracker:
            self.tracker.close()
        output_dir = Path("results/race_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = RaceTracker(output_dir, save_crops=False)

        # Reset smoothed positions (different camera angle)
        self._smooth_x.clear()
        log.info(f"[VIDEO START] #{new_index}")

    def _build_rankings(self, frame_width: int) -> list:
        """Build frontend rankings from current order + smoothed positions."""
        if not self._current_order:
            return []

        rankings = []
        for pos, color_name in enumerate(self._current_order):
            horse_info = COLOR_TO_HORSE.get(color_name)
            if not horse_info:
                continue

            # Smoothed X → distanceCovered for track visualization
            sx = self._smooth_x.get(color_name, frame_width * (1.0 - pos * 0.15))
            distance = (float(sx) / max(frame_width, 1)) * TRACK_LENGTH

            # Gap based on pixel distance to leader
            leader_x = self._smooth_x.get(self._current_order[0], float(frame_width))
            gap_px = float(leader_x) - float(sx)
            gap_seconds = abs(gap_px) / max(frame_width, 1) * 8.0

            rankings.append({
                "id": horse_info["id"],
                "number": int(horse_info["number"]),
                "name": horse_info["name"],
                "color": horse_info["color"],
                "jockeyName": horse_info["jockeyName"],
                "silkId": int(horse_info["silkId"]),
                "position": pos + 1,
                "distanceCovered": round(float(distance), 1),
                "currentLap": 1,
                "timeElapsed": 0,
                "speed": 0,
                "gapToLeader": round(float(gap_seconds), 2),
                "lastCameraId": "cam-1",
            })

        return rankings

    def stop(self):
        self.running = False
        if self.tracker:
            self.tracker.close()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Race Vision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected WebSocket clients
ws_clients: set[WebSocket] = set()

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    log.info(f"WebSocket client connected ({len(ws_clients)} total)")

    # Send horses_detected on connect
    horses_msg = {
        "type": "horses_detected",
        "horses": [
            {
                "id": info["id"],
                "number": info["number"],
                "name": info["name"],
                "color": info["color"],
                "jockeyName": info["jockeyName"],
                "silkId": info["silkId"],
            }
            for info in COLOR_TO_HORSE.values()
        ],
    }
    await websocket.send_json(horses_msg)

    # Send race_start if race is active
    if state.race_active:
        await websocket.send_json({
            "type": "race_start",
            "race": {
                "name": "Live Race",
                "totalLaps": 1,
                "trackLength": TRACK_LENGTH,
            },
        })

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg.get("type") == "get_state":
                rankings = state.get_rankings()
                await websocket.send_json({
                    "type": "state",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                        "status": "active" if state.race_active else "pending",
                    },
                    "rankings": rankings,
                })

            elif msg.get("type") == "start_race":
                state.race_active = True
                log.info("Race started (from operator)")
                broadcast_msg = {
                    "type": "race_start",
                    "race": {
                        "name": "Live Race",
                        "totalLaps": 1,
                        "trackLength": TRACK_LENGTH,
                    },
                }
                await broadcast(broadcast_msg)

            elif msg.get("type") == "stop_race":
                state.race_active = False
                log.info("Race stopped (from operator)")
                await broadcast({"type": "race_stop"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        ws_clients.discard(websocket)
        log.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


async def broadcast(msg: dict):
    """Send a message to all connected WebSocket clients."""
    dead = set()
    for client in ws_clients:
        try:
            await client.send_json(msg)
        except Exception:
            dead.add(client)
    ws_clients.difference_update(dead)


# ============================================================
# MJPEG STREAM ENDPOINT
# ============================================================

def mjpeg_generator():
    """Yield MJPEG frames from annotated detection output."""
    delay = 1.0 / MJPEG_FPS

    while True:
        frame = state.get_annotated_frame()
        if frame is None:
            # Show a black frame with "waiting" text if no frame yet
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for video...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(delay)


@app.get("/stream/cam{cam_id}")
async def mjpeg_stream(cam_id: int):
    """MJPEG video stream (all cam IDs serve same stream for now)."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# BACKGROUND BROADCAST TASK
# ============================================================

async def ranking_broadcast_loop():
    """Periodically broadcast ranking updates to all WebSocket clients."""
    last_log_time = 0
    while True:
        await asyncio.sleep(BROADCAST_INTERVAL)

        if not ws_clients or not state.race_active:
            continue

        rankings = state.get_rankings()
        if not rankings:
            continue

        msg = {
            "type": "ranking_update",
            "rankings": rankings,
        }
        await broadcast(msg)

        # Log every 5 seconds
        now = time.time()
        if now - last_log_time > 5.0:
            colors = [r.get("name", "?") for r in rankings]
            log.info(f"Broadcasting {len(rankings)} horses to {len(ws_clients)} clients: {colors}")
            last_log_time = now


@app.on_event("startup")
async def startup():
    asyncio.create_task(ranking_broadcast_loop())
    log.info(f"Race Vision backend running on http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"  WebSocket: ws://localhost:{SERVER_PORT}/ws")
    log.info(f"  MJPEG:     http://localhost:{SERVER_PORT}/stream/cam1")


# ============================================================
# MAIN
# ============================================================

# Global references so we can clean up
_grabber: Optional[FrameGrabber] = None
_detector: Optional[DetectionLoop] = None


def main():
    global _grabber, _detector

    parser = argparse.ArgumentParser(description="Race Vision Backend Server")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--video", nargs="+", default=None,
                        help="Local video file(s) played sequentially (instead of RTSP)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU decode (hevc_cuvid)")
    parser.add_argument("--host", default=SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start race on launch (skip operator start)")
    args = parser.parse_args()

    # Determine video source(s)
    if args.video:
        sources = args.video
        is_rtsp = False
        log.info(f"Video sources: {[Path(s).stem for s in sources]}")
    else:
        sources = [args.url]
        is_rtsp = True
        log.info(f"RTSP source: {args.url.split('@')[-1] if '@' in args.url else args.url}")

    # Start frame grabber
    _grabber = FrameGrabber(sources, use_gpu=args.gpu, is_rtsp=is_rtsp)
    _grabber.start()
    log.info("Frame grabber started")

    # Start detection loop
    _detector = DetectionLoop()
    _detector.start()
    log.info("Detection loop started")

    # Auto-start race if requested
    if args.auto_start:
        state.race_active = True
        log.info("Race auto-started")

    # Run FastAPI server
    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        log.info("Shutting down...")
        _grabber.stop()
        _detector.stop()


if __name__ == "__main__":
    main()
