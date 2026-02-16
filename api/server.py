"""
Race Vision — FastAPI Backend Server (Multi-Camera Architecture)

Supports 25 analytics cameras (YOLO detection) + 4 PTZ cameras (broadcast only).
Uses GPU-accelerated RTSP decode (NVDEC) and WebRTC streaming.

Architecture:
    25 Analytics RTSP → CameraReader threads → MultiCameraManager
    4 PTZ RTSP       → CameraReader threads → MultiCameraManager
                                                    ↓
    SmartDetectionScheduler → MultiDetectionLoop → per-camera state
                                                    ↓
    RankingMerger → combined rankings → WebSocket broadcast
                                                    ↓
    WebRTC/MJPEG streams → Frontend (operator + public display)

Usage:
    python api/server.py                                    # RTSP mode (configure cameras via API)
    python api/server.py --video video/cam1.mp4 video/cam2.mp4  # Video file test mode
    python api/server.py --gpu                              # Enable GPU decode
    python api/server.py --auto-start                       # Auto-start race
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
from contextlib import asynccontextmanager
from collections import Counter
from copy import deepcopy

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows Proactor loop can spam WinError 10054 on transient websocket closes.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel

class CameraUpdate(BaseModel):
    rtspUrl: str

# Import from our existing modules
from tools.test_race_count import (
    RaceTracker, ColorClassifier, SimpleColorCNN, draw,
    COLORS_BGR, REQUIRED_COLORS
)

# Import new multi-camera modules
from api.camera_manager import MultiCameraManager, CameraReader, VideoFileReader
from api.smart_detection import CameraDetectionState, SmartDetectionScheduler
from api.ranking_merger import RankingMerger
from api.webrtc_server import setup_webrtc_routes, WEBRTC_AVAILABLE

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RTSP_URL = "rtsp://admin:Qaz445566@192.168.18.59:554//stream"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Detection loop rate (seconds between updates)
DETECTION_INTERVAL = 0.10  # ~10 fps
# Lightweight preview detection when race is not started yet
PREVIEW_DETECTION_INTERVAL = 0.35
PREVIEW_MAX_CAMERAS_PER_CYCLE = 5
PREVIEW_ACCESS_FRESHNESS_SEC = 8.0

# WebSocket broadcast rate
BROADCAST_INTERVAL = 0.20  # 5 Hz

# MJPEG settings (fallback)
MJPEG_QUALITY = 75
MJPEG_FPS = 25
MAX_ANNOTATED_FRAME_AGE_SEC = 0.9
PTZ_GPU_ANALYTICS_THRESHOLD = 8

# Track mapping
TRACK_LENGTH = 2500
NUM_ANALYTICS_CAMERAS = 25
NUM_PTZ_CAMERAS = 4
CAMERA_TRACK_M = 100.0  # Each analytics camera covers 100m

# GPU settings
USE_GPU = False  # Set via --gpu flag

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
DEFAULT_COLOR_TO_HORSE = deepcopy(COLOR_TO_HORSE)

RACE_SETTINGS = {
    "name": "Live Race",
    "totalLaps": 1,
    "trackLength": TRACK_LENGTH,
    "startFinishPosition": 0,
}
RACE_STATUS = "pending"

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("race_server")


def configure_clean_logging():
    """Reduce noisy third-party logs while keeping race_server logs visible."""
    noisy_loggers = {
        "aioice": logging.ERROR,            # ICE candidate pair spam
        "aioice.ice": logging.ERROR,        # ICE checks/candidates
        "aioice.stun": logging.ERROR,       # STUN retry chatter
        "aiortc": logging.ERROR,            # WebRTC internals
        "av": logging.WARNING,              # ffmpeg/pyav details
        "uvicorn.access": logging.ERROR,    # per-request access lines
    }
    for logger_name, level in noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def install_asyncio_exception_filter():
    """Suppress noisy transient Windows socket-reset callbacks."""
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def _handler(loop_obj, context):
        exc = context.get("exception")
        msg = str(context.get("message", ""))

        if isinstance(exc, ConnectionResetError):
            if getattr(exc, "winerror", None) == 10054:
                return

        if "_ProactorBasePipeTransport._call_connection_lost" in msg:
            return

        if previous_handler is not None:
            previous_handler(loop_obj, context)
        else:
            loop_obj.default_exception_handler(context)

    loop.set_exception_handler(_handler)


# Apply once at import time (works for `uvicorn api.server:app` mode too).
configure_clean_logging()

# ============================================================
# SHARED STATE (Multi-Camera Aware)
# ============================================================

class SharedState:
    """Thread-safe state shared between camera manager, detector, and server."""

    def __init__(self):
        self._lock = threading.Lock()

        # Per-camera annotated frames (cam_id string → numpy array)
        self.annotated_frames: dict[str, np.ndarray] = {}
        self.annotated_frame_times: dict[str, float] = {}

        # Per-camera detection results
        self.per_camera_detections: dict[str, list] = {}

        # Combined rankings from all cameras
        self.combined_rankings: list = []

        # Which horses are visible on which camera
        self.camera_horse_presence: dict[str, set] = {}
        self.stream_access_times: dict[str, float] = {}

        # Race state
        self.race_active: bool = False
        self.detection_fps: float = 0.0
        self.detection_count: int = 0

        # Legacy compatibility for video mode
        self.video_index: int = 0
        self.active_cam_id: str = ""

    def set_camera_detection(self, cam_id: str, jockeys: list,
                              annotated: np.ndarray, horse_colors: set):
        """Store detection results for a specific camera."""
        now = time.time()
        with self._lock:
            self.per_camera_detections[cam_id] = jockeys
            self.annotated_frames[cam_id] = annotated
            self.annotated_frame_times[cam_id] = now
            self.camera_horse_presence[cam_id] = horse_colors
            self.detection_count += 1

    def get_annotated_frame(self, cam_id: str,
                            max_age_sec: Optional[float] = None) -> Optional[np.ndarray]:
        """Get annotated frame for a specific camera by cam_id string."""
        now = time.time()
        with self._lock:
            frame = self.annotated_frames.get(cam_id)
            ts = self.annotated_frame_times.get(cam_id, 0.0)
            if frame is None:
                return None
            if max_age_sec is not None and ts > 0 and (now - ts) > max_age_sec:
                return None
            return frame.copy()

    def set_combined_rankings(self, rankings: list):
        """Store the merged ranking list from all cameras."""
        with self._lock:
            self.combined_rankings = list(rankings) if rankings is not None else []

    def get_combined_rankings(self) -> list:
        """Get the current combined rankings."""
        with self._lock:
            return list(self.combined_rankings)

    # Legacy alias
    def get_rankings(self) -> list:
        return self.get_combined_rankings()

    def set_detection_fps(self, fps: float):
        with self._lock:
            self.detection_fps = fps

    def mark_stream_access(self, cam_id: str):
        """Record when an analytics camera stream/snapshot was requested."""
        if not cam_id.startswith("analytics-"):
            return
        with self._lock:
            self.stream_access_times[cam_id] = time.time()

    def get_recent_stream_cameras(self, max_count: int = 4,
                                   freshness_sec: float = 8.0) -> list[str]:
        """Return recently requested analytics cameras, newest first."""
        now = time.time()
        with self._lock:
            recent = [
                (cam_id, ts)
                for cam_id, ts in self.stream_access_times.items()
                if now - ts <= freshness_sec
            ]
        recent.sort(key=lambda item: item[1], reverse=True)
        return [cam_id for cam_id, _ in recent[:max_count]]


state = SharedState()

# ============================================================
# GLOBAL MULTI-CAMERA MANAGER
# ============================================================

camera_manager = MultiCameraManager()

# RTSP URLs configured by operator (persisted in memory)
CUSTOM_CAMERA_URLS: dict[str, str] = {}


def _gpu_for_camera_type(cam_type: str) -> bool:
    """Select GPU decode policy for camera types.

    Analytics cameras keep priority. PTZ can use GPU only when analytics
    load is still low, which helps 4K HEVC PTZ feeds connect reliably.
    """
    if not USE_GPU:
        return False
    if cam_type == "analytics":
        return True

    # PTZ policy: allow GPU only while analytics load is modest.
    running_analytics = len(camera_manager.get_analytics_cameras())
    return running_analytics <= PTZ_GPU_ANALYTICS_THRESHOLD


def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, ivalue))


def get_horses_payload() -> list[dict]:
    """Build horse payload ordered by detector color channels."""
    payload = []
    for color in ALL_COLORS:
        info = COLOR_TO_HORSE[color]
        payload.append({
            "id": str(info.get("id", f"horse-{info.get('number', 1)}")),
            "number": int(info.get("number", 1)),
            "name": str(info.get("name", f"Horse {info.get('number', 1)}")),
            "color": str(info.get("color", "#FFFFFF")),
            "jockeyName": str(info.get("jockeyName", f"Jockey {info.get('number', 1)}")),
            "silkId": int(info.get("silkId", 1)),
        })
    return payload


def get_race_payload() -> dict:
    race_status = "active" if state.race_active else RACE_STATUS
    return {
        "name": RACE_SETTINGS["name"],
        "totalLaps": int(RACE_SETTINGS["totalLaps"]),
        "trackLength": int(RACE_SETTINGS["trackLength"]),
        "startFinishPosition": int(RACE_SETTINGS["startFinishPosition"]),
        "status": race_status,
    }


def build_state_payload() -> dict:
    return {
        "type": "state",
        "race": get_race_payload(),
        "rankings": state.get_rankings(),
        "horses": get_horses_payload(),
    }


def apply_race_settings(update: dict) -> bool:
    """Apply race settings from operator input. Returns True if changed."""
    if not isinstance(update, dict):
        return False

    changed = False

    if "name" in update:
        name = str(update.get("name", "")).strip()
        if name and name != RACE_SETTINGS["name"]:
            RACE_SETTINGS["name"] = name
            changed = True

    if "trackLength" in update:
        new_track_len = _clamp_int(update.get("trackLength"), RACE_SETTINGS["trackLength"], 100, 10000)
        if new_track_len != RACE_SETTINGS["trackLength"]:
            RACE_SETTINGS["trackLength"] = new_track_len
            changed = True
            # Keep start/finish marker within track bounds
            clamped = min(RACE_SETTINGS["startFinishPosition"], new_track_len)
            if clamped != RACE_SETTINGS["startFinishPosition"]:
                RACE_SETTINGS["startFinishPosition"] = clamped

    if "totalLaps" in update:
        new_laps = _clamp_int(update.get("totalLaps"), RACE_SETTINGS["totalLaps"], 1, 100)
        if new_laps != RACE_SETTINGS["totalLaps"]:
            RACE_SETTINGS["totalLaps"] = new_laps
            changed = True

    if "startFinishPosition" in update:
        track_len = int(RACE_SETTINGS["trackLength"])
        new_pos = _clamp_int(update.get("startFinishPosition"), RACE_SETTINGS["startFinishPosition"], 0, track_len)
        if new_pos != RACE_SETTINGS["startFinishPosition"]:
            RACE_SETTINGS["startFinishPosition"] = new_pos
            changed = True

    return changed


def apply_horses_update(horses: list) -> bool:
    """Apply horse roster from frontend.

    Important: detector color channels are fixed (red/blue/green/yellow/purple).
    We keep the channel color as the horse color in backend outputs so UI color
    always matches what analytics cameras actually detect.
    """
    if not isinstance(horses, list) or not horses:
        return False

    normalized: list[dict] = []
    for i, raw in enumerate(horses):
        if not isinstance(raw, dict):
            continue
        fallback_num = i + 1
        horse_id = str(raw.get("id") or f"horse-{fallback_num}")
        number = _clamp_int(raw.get("number"), fallback_num, 1, 999)
        name = str(raw.get("name") or f"Horse {number}")
        jockey_name = str(raw.get("jockeyName") or f"Jockey {number}")
        color = str(raw.get("color") or "#FFFFFF")
        silk_id = _clamp_int(raw.get("silkId"), number, 1, 999)
        normalized.append({
            "id": horse_id,
            "number": number,
            "name": name,
            "jockeyName": jockey_name,
            "color": color,
            "silkId": silk_id,
        })

    if not normalized:
        return False

    if len(normalized) > len(ALL_COLORS):
        log.warning(
            f"Received {len(normalized)} horses, detector supports {len(ALL_COLORS)} colors; trimming extras"
        )

    # Keep mapping stable by horse number (not transient UI row order).
    ordered = sorted(normalized, key=lambda h: h["number"])

    changed = False
    for i, det_color in enumerate(ALL_COLORS):
        if i < len(ordered):
            src = ordered[i]
            new_info = {
                "id": src["id"],
                "number": src["number"],
                "name": src["name"],
                "silkId": src["silkId"],
                # Always expose detector channel color (what cameras detect).
                "color": DEFAULT_COLOR_TO_HORSE[det_color]["color"],
                "jockeyName": src["jockeyName"],
            }
        else:
            new_info = deepcopy(DEFAULT_COLOR_TO_HORSE[det_color])

        if COLOR_TO_HORSE.get(det_color) != new_info:
            COLOR_TO_HORSE[det_color] = new_info
            changed = True

    return changed


def reset_runtime_state():
    """Reset runtime race state and cached rankings."""
    global RACE_STATUS
    state.race_active = False
    RACE_STATUS = "pending"
    with state._lock:
        state.combined_rankings = []
        state.camera_horse_presence.clear()
        state.per_camera_detections.clear()
        state.annotated_frames.clear()
        state.annotated_frame_times.clear()

    detector = globals().get("_detector")
    if detector is not None:
        try:
            detector.merger.reset()
        except Exception as e:
            log.warning(f"Failed to reset merger state: {e}")

# ============================================================
# MULTI-CAMERA DETECTION LOOP
# ============================================================

class MultiDetectionLoop(threading.Thread):
    """Runs YOLO + ColorClassifier across multiple cameras with smart scheduling.

    For each detection cycle:
    1. Ask SmartDetectionScheduler which cameras need processing
    2. For each camera: grab frame → YOLO → color classify → 4-layer filter
    3. Update per-camera CameraDetectionState
    4. Merge rankings from all cameras via RankingMerger
    5. Update SharedState for WebSocket broadcast

    In video mode: works with single video source (backward compatible).
    """

    def __init__(self):
        super().__init__(daemon=True, name="MultiDetectionLoop")
        self.running = False
        self.tracker: Optional[RaceTracker] = None

        # Per-camera detection states (analytics cameras only)
        self.camera_states: dict[str, CameraDetectionState] = {}
        for i in range(NUM_ANALYTICS_CAMERAS):
            cam_id = f"analytics-{i + 1}"
            self.camera_states[cam_id] = CameraDetectionState(cam_id, i)

        # Smart scheduler
        self.scheduler = SmartDetectionScheduler(self.camera_states)

        # Ranking merger
        self.merger = RankingMerger(COLOR_TO_HORSE, ALL_COLORS)

        # Race timing
        self._race_start_time: float = 0.0

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = 0.0
        self._current_fps = 0.0

        # Video mode tracking
        self._current_video_index = 0
        self._video_mode = False
        self._preview_rr_index = 0

    def run(self):
        self.running = True
        self._fps_timer = time.time()

        output_dir = Path("results/race_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = RaceTracker(output_dir, save_crops=False)
        log.info("Multi-camera detection pipeline ready")

        while self.running:
            if not state.race_active:
                self._race_start_time = 0.0
                # Keep a lightweight preview detector alive so operators can
                # verify horses before pressing "Start race".
                if self._video_mode:
                    self._process_video_mode()
                else:
                    self._process_preview_mode()
                time.sleep(PREVIEW_DETECTION_INTERVAL)
                continue

            # Start race timer on first active frame
            if self._race_start_time == 0.0:
                self._race_start_time = time.time()
                self.merger.set_race_start_time(self._race_start_time)

            t0 = time.time()

            if self._video_mode:
                self._process_video_mode()
            else:
                self._process_multi_camera()

            # FPS tracking
            self._fps_counter += 1
            elapsed_fps = time.time() - self._fps_timer
            if elapsed_fps >= 1.0:
                self._current_fps = self._fps_counter / elapsed_fps
                self._fps_counter = 0
                self._fps_timer = time.time()
                state.set_detection_fps(self._current_fps)

            # Rate limit
            dt = time.time() - t0
            sleep_time = DETECTION_INTERVAL - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_preview_mode(self):
        """Run low-rate detection preview before race start."""
        cameras_to_process = self._select_preview_cameras(PREVIEW_MAX_CAMERAS_PER_CYCLE)
        if not cameras_to_process:
            return

        frame_widths = {}

        for cam_id in cameras_to_process:
            frame = camera_manager.get_frame(cam_id)
            if frame is None:
                continue

            cam_state = self.camera_states.get(cam_id)
            if cam_state is None:
                continue

            cam_state.last_process_time = time.time()
            cam_state.frame_number += 1
            frame_width = frame.shape[1]
            frame_widths[cam_id] = frame_width

            jockeys, detections = self.tracker.update(frame)
            annotated = draw(frame.copy(), jockeys, self.tracker)

            cam_state.evict_stale_colors()
            filtered = self._apply_filters(detections, cam_state, frame_width)

            for det in filtered:
                color = str(det.get("color", ""))
                if not color:
                    continue
                cx = float(det.get("center_x", 0.0))
                cam_state.update_smooth_x(color, cx)
                cam_state.color_confidence[color] = float(det.get("conf", 0.5))

            filtered_colors = {str(d.get("color")) for d in filtered if d.get("color")}
            if len(filtered_colors) >= 3:
                sorted_dets = sorted(filtered, key=lambda d: -float(d.get("center_x", 0.0)))
                seen = set()
                order = []
                for d in sorted_dets:
                    color = str(d.get("color", ""))
                    if color and color not in seen:
                        seen.add(color)
                        order.append(color)
                for c in ALL_COLORS:
                    if c not in seen:
                        order.append(c)
                cam_state.update_live_votes(order)

            now_t = time.time()
            cam_state.horses_present = {
                c for c, t in cam_state.color_last_seen.items()
                if now_t - t < 2.0
            }

            state.set_camera_detection(cam_id, jockeys, annotated, cam_state.horses_present)

        if not frame_widths:
            return

        all_frame_widths = {}
        for cam_id in self.camera_states:
            w, _ = camera_manager.get_frame_dimensions(cam_id)
            if w > 0:
                all_frame_widths[cam_id] = w
        all_frame_widths.update(frame_widths)

        combined = self.merger.merge(self.camera_states, all_frame_widths)
        state.set_combined_rankings(combined)

    def _select_preview_cameras(self, max_count: int) -> list[str]:
        """Prefer recently viewed analytics cameras, then round-robin scan."""
        if max_count <= 0:
            return []

        active = camera_manager.get_analytics_cameras()
        if not active:
            return []

        chosen: list[str] = []
        recent = state.get_recent_stream_cameras(
            max_count=max_count,
            freshness_sec=PREVIEW_ACCESS_FRESHNESS_SEC,
        )
        for cam_id in recent:
            if cam_id in active and cam_id not in chosen:
                chosen.append(cam_id)
            if len(chosen) >= max_count:
                return chosen

        remaining = [cam_id for cam_id in active if cam_id not in chosen]
        if not remaining:
            return chosen

        start = self._preview_rr_index % len(remaining)
        ordered = remaining[start:] + remaining[:start]
        needed = max_count - len(chosen)
        picked = ordered[:needed]
        chosen.extend(picked)
        self._preview_rr_index = (start + len(picked)) % len(remaining)
        return chosen

    def _process_multi_camera(self):
        """Process multiple RTSP cameras using smart scheduling."""
        # Process only currently running analytics cameras.
        active_cameras = camera_manager.get_analytics_cameras()
        if not active_cameras:
            return

        # If active camera count fits the per-cycle budget, process all of them
        # every cycle for stable low-latency analytics (common in 2-5 camera races).
        max_per_cycle = self.scheduler.MAX_CAMERAS_PER_CYCLE
        if len(active_cameras) <= max_per_cycle:
            cameras_to_process = sorted(active_cameras)
        else:
            active_set = set(active_cameras)
            scheduled = self.scheduler.get_processing_queue()
            cameras_to_process = [cam_id for cam_id in scheduled if cam_id in active_set]
            if not cameras_to_process:
                cameras_to_process = sorted(active_cameras)[:max_per_cycle]

        frame_widths = {}

        for cam_id in cameras_to_process:
            frame = camera_manager.get_frame(cam_id)
            if frame is None:
                continue

            cam_state = self.camera_states.get(cam_id)
            if cam_state is None:
                continue

            cam_state.last_process_time = time.time()
            cam_state.frame_number += 1
            frame_width = frame.shape[1]
            frame_widths[cam_id] = frame_width

            # Run YOLO + color classification
            jockeys, detections = self.tracker.update(frame)

            # Draw annotations
            annotated = draw(frame.copy(), jockeys, self.tracker)

            # Evict stale colors that haven't been seen recently
            cam_state.evict_stale_colors()

            # Apply 4-layer filtering
            filtered = self._apply_filters(detections, cam_state, frame_width)

            # Update smoothed positions only from filtered detections
            for d in filtered:
                color = str(d.get("color", ""))
                if not color:
                    continue
                cx = float(d.get("center_x", 0.0))
                cam_state.update_smooth_x(color, cx)
                cam_state.color_confidence[color] = float(d.get("conf", 0.5))

            # Vote on order if enough colors passed
            filtered_colors = {d['color'] for d in filtered}
            if len(filtered_colors) >= 4:
                sorted_dets = sorted(filtered, key=lambda d: -d['center_x'])
                seen = set()
                order = []
                for d in sorted_dets:
                    if d['color'] not in seen:
                        seen.add(d['color'])
                        order.append(d['color'])
                for c in ALL_COLORS:
                    if c not in seen:
                        order.append(c)
                cam_state.update_live_votes(order)

            # Update horse presence based on per-color last-seen time (not smooth_x keys)
            now_t = time.time()
            cam_state.horses_present = {
                c for c, t in cam_state.color_last_seen.items()
                if now_t - t < 2.0
            }

            # Store results
            state.set_camera_detection(cam_id, jockeys, annotated, cam_state.horses_present)

        # Update scheduler priorities
        self.scheduler.update_priorities(self.camera_states, frame_widths)

        # Merge rankings from all cameras
        all_frame_widths = {}
        for cam_id in self.camera_states:
            w, _ = camera_manager.get_frame_dimensions(cam_id)
            if w > 0:
                all_frame_widths[cam_id] = w
        # Use last known widths for cameras not processed this cycle
        for cam_id, fw in frame_widths.items():
            all_frame_widths[cam_id] = fw

        combined = self.merger.merge(self.camera_states, all_frame_widths)
        state.set_combined_rankings(combined)

    def _process_video_mode(self):
        """Process frames from video file reader (backward compatible)."""
        vr = camera_manager.get_video_reader()
        if vr is None:
            time.sleep(0.1)
            return

        active_cam_id = vr.get_active_cam_id()
        if not active_cam_id:
            time.sleep(0.05)
            return

        # Detect video switch
        vid_idx = vr.get_video_index()
        if vid_idx != self._current_video_index:
            self._on_video_switch(vid_idx, active_cam_id)

        frame = vr.get_frame(active_cam_id)
        if frame is None:
            time.sleep(0.05)
            return

        cam_state = self.camera_states.get(active_cam_id)
        if cam_state is None:
            # Create state dynamically for video mode
            cam_idx = int(active_cam_id.split('-')[1]) - 1
            cam_state = CameraDetectionState(active_cam_id, cam_idx)
            self.camera_states[active_cam_id] = cam_state

        cam_state.last_process_time = time.time()
        cam_state.frame_number += 1
        frame_width = frame.shape[1]

        # Run detection
        jockeys, detections = self.tracker.update(frame)
        annotated = draw(frame.copy(), jockeys, self.tracker)

        # Evict stale colors
        cam_state.evict_stale_colors()

        # Apply filters
        filtered = self._apply_filters(detections, cam_state, frame_width)

        # Update smoothed positions from filtered detections only
        for d in filtered:
            color = str(d.get("color", ""))
            if not color:
                continue
            cx = float(d.get("center_x", 0.0))
            cam_state.update_smooth_x(color, cx)
            cam_state.color_confidence[color] = float(d.get("conf", 0.5))

        # Vote
        filtered_colors = {d['color'] for d in filtered}
        if len(filtered_colors) >= 4:
            sorted_dets = sorted(filtered, key=lambda d: -d['center_x'])
            seen = set()
            order = []
            for d in sorted_dets:
                if d['color'] not in seen:
                    seen.add(d['color'])
                    order.append(d['color'])
            for c in ALL_COLORS:
                if c not in seen:
                    order.append(c)
            cam_state.update_live_votes(order)

        # Horse presence based on per-color last-seen time
        now_t = time.time()
        cam_state.horses_present = {
            c for c, t in cam_state.color_last_seen.items()
            if now_t - t < 2.0
        }

        state.set_camera_detection(active_cam_id, jockeys, annotated, cam_state.horses_present)
        state.active_cam_id = active_cam_id

        # Build rankings using merger
        fw_dict = {active_cam_id: frame_width}
        combined = self.merger.merge(self.camera_states, fw_dict)
        state.set_combined_rankings(combined)

    def _apply_filters(self, detections: list, cam_state: CameraDetectionState,
                       frame_width: int) -> list:
        """Apply 4-layer filtering to raw detections. Returns filtered list."""
        filtered = []
        for det in detections:
            color = det['color']
            conf = float(det.get('conf', 0.0))
            hsv = det.get('hsv_guess', '')
            cx = float(det['center_x'])
            pos_m = (cx / max(frame_width, 1)) * CAMERA_TRACK_M
            cam_state.filter_stats['total'] += 1

            # F1: Confidence threshold
            if conf < cam_state.CONF_THRESHOLD:
                cam_state.filter_stats['pass_f1'] += 1
                continue

            # F2: CNN + HSV agreement
            if hsv and hsv != color and conf < cam_state.HSV_SKIP_CONF:
                cam_state.filter_stats['pass_f2'] += 1
                continue

            # F3: Speed constraint
            if color in cam_state.last_pos:
                last_pos_m, last_time = cam_state.last_pos[color]
                dt_sec = time.time() - last_time
                if dt_sec > 0.01:
                    speed = abs(pos_m - last_pos_m) / dt_sec
                    if speed > cam_state.MAX_SPEED_MPS:
                        cam_state.filter_stats['pass_f3'] += 1
                        continue

            # Compute speed before updating position
            if color in cam_state.last_pos:
                prev_pos_m, prev_time = cam_state.last_pos[color]
                dt_speed = time.time() - prev_time
                if dt_speed > 0.01:
                    raw_speed = abs(pos_m - prev_pos_m) / dt_speed
                    cam_state.update_speed(color, raw_speed)

            # Update position tracking
            cam_state.last_pos[color] = (pos_m, time.time())
            cam_state.add_temporal_detection(color)

            # F4: Temporal confirmation
            if not cam_state.check_temporal(color):
                cam_state.filter_stats['pass_f4'] += 1
                continue

            cam_state.filter_stats['total'] += 0  # accepted
            filtered.append(det)

        return filtered

    def _on_video_switch(self, new_index: int, new_cam_id: str):
        """Handle video file switch (reset per-camera state)."""
        old_idx = self._current_video_index
        old_cam_id = f"analytics-{old_idx + 1}"
        self._current_video_index = new_index

        log.info(f"[VIDEO SWITCH] #{old_idx} → #{new_index} ({new_cam_id})")

        # Reset tracker for new camera perspective
        if self.tracker:
            self.tracker.close()
        output_dir = Path("results/race_server")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = RaceTracker(output_dir, save_crops=False)

        # Clear old camera's smooth_x so merger doesn't pick up stale data
        old_state = self.camera_states.get(old_cam_id)
        if old_state:
            old_state.reset()

        # Reset state for the new camera
        cam_idx = int(new_cam_id.split('-')[1]) - 1
        self.camera_states[new_cam_id] = CameraDetectionState(new_cam_id, cam_idx)

    def stop(self):
        self.running = False
        if self.tracker:
            self.tracker.close()


# ============================================================
# FASTAPI APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Re-apply in startup in case runner reconfigured loggers.
    configure_clean_logging()
    install_asyncio_exception_filter()
    ranking_task = asyncio.create_task(ranking_broadcast_loop())
    camera_detection_task = asyncio.create_task(camera_detection_broadcast_loop())

    log.info(f"Race Vision backend running on http://{SERVER_HOST}:{SERVER_PORT}")
    log.info(f"  WebSocket: ws://localhost:{SERVER_PORT}/ws")
    log.info(f"  MJPEG:     http://localhost:{SERVER_PORT}/stream/cam1")
    log.info(f"  WebRTC:    {'enabled' if WEBRTC_AVAILABLE else 'disabled (install aiortc)'}")
    log.info(f"  Health:    http://localhost:{SERVER_PORT}/api/system/health")

    try:
        yield
    finally:
        ranking_task.cancel()
        camera_detection_task.cancel()
        await asyncio.gather(ranking_task, camera_detection_task, return_exceptions=True)


app = FastAPI(title="Race Vision Backend", lifespan=lifespan)

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
    global RACE_STATUS
    await websocket.accept()
    ws_clients.add(websocket)
    log.info(f"WebSocket client connected ({len(ws_clients)} total)")

    try:
        # Send initial state packets
        horses_msg = {
            "type": "horses_detected",
            "horses": get_horses_payload(),
        }
        await websocket.send_json(horses_msg)

        if state.race_active:
            await websocket.send_json({
                "type": "race_start",
                "race": get_race_payload(),
            })

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg.get("type") == "get_state":
                await websocket.send_json(build_state_payload())

            elif msg.get("type") == "set_race_config":
                race_update = msg.get("race", {})
                if apply_race_settings(race_update):
                    log.info(
                        "[RACE CONFIG] "
                        f"name={RACE_SETTINGS['name']}, "
                        f"laps={RACE_SETTINGS['totalLaps']}, "
                        f"track={RACE_SETTINGS['trackLength']}, "
                        f"startFinish={RACE_SETTINGS['startFinishPosition']}"
                    )
                    await broadcast(build_state_payload())

            elif msg.get("type") == "set_horses":
                if apply_horses_update(msg.get("horses", [])):
                    horses_msg = {
                        "type": "horses_detected",
                        "horses": get_horses_payload(),
                    }
                    log.info(f"[HORSES UPDATE] Applied {len(horses_msg['horses'])} horses")
                    await broadcast(horses_msg)
                    await broadcast(build_state_payload())

            elif msg.get("type") == "start_race":
                state.race_active = True
                RACE_STATUS = "active"
                log.info("Race started (from operator)")
                broadcast_msg = {
                    "type": "race_start",
                    "race": get_race_payload(),
                }
                await broadcast(broadcast_msg)

            elif msg.get("type") == "stop_race":
                state.race_active = False
                RACE_STATUS = "finished"
                log.info("Race stopped (from operator)")
                await broadcast({"type": "race_stop"})

            elif msg.get("type") == "reset_race":
                log.info("Race reset (from operator)")
                reset_runtime_state()
                await broadcast({"type": "race_stop"})
                await broadcast(build_state_payload())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f"WebSocket session ended: {e}")
    finally:
        ws_clients.discard(websocket)
        log.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


async def broadcast(msg: dict):
    """Send a message to all connected WebSocket clients."""
    dead = set()
    for client in list(ws_clients):
        try:
            await client.send_json(msg)
        except Exception:
            dead.add(client)
    ws_clients.difference_update(dead)


# ============================================================
# CAMERA MANAGEMENT API (Real implementations)
# ============================================================

@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, config: CameraUpdate):
    """Update RTSP URL for a camera. Restart only if URL changed."""
    previous_url = CUSTOM_CAMERA_URLS.get(camera_id)
    new_url = config.rtspUrl
    url_changed = previous_url != new_url
    CUSTOM_CAMERA_URLS[camera_id] = new_url
    if url_changed:
        log.info(f"[CAMERA UPDATE] {camera_id} → {config.rtspUrl}")

    restarted = False

    # Restart active/connecting/error readers when URL changes so new creds apply immediately.
    current = camera_manager.get_status().get(camera_id, {})
    current_state = current.get("state") if isinstance(current, dict) else None
    if url_changed and current_state in {
        CameraReader.RUNNING,
        CameraReader.CONNECTING,
        CameraReader.ERROR,
    }:
        cam_type = "ptz" if camera_id.startswith("ptz") else "analytics"
        camera_manager.start_camera(camera_id, new_url,
                                     use_gpu=_gpu_for_camera_type(cam_type), cam_type=cam_type)
        log.info(f"[CAMERA RESTART] {camera_id} restarted with new URL")
        restarted = True
    elif not url_changed:
        log.debug(f"[CAMERA UPDATE] {camera_id} unchanged, restart skipped")

    return {
        "id": camera_id,
        "status": "updated",
        "urlChanged": url_changed,
        "restarted": restarted,
    }


@app.post("/api/cameras/{camera_id}/start")
async def start_camera_endpoint(camera_id: str):
    """Start an RTSP camera."""
    url = CUSTOM_CAMERA_URLS.get(camera_id)
    if not url:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"No RTSP URL configured for {camera_id}"}
        )

    current = camera_manager.get_status().get(camera_id, {})
    current_state = current.get("state") if isinstance(current, dict) else None
    if current_state in {CameraReader.RUNNING, CameraReader.CONNECTING}:
        return {"status": "already_active", "id": camera_id, "state": current_state}

    cam_type = "ptz" if camera_id.startswith("ptz") else "analytics"
    camera_manager.start_camera(camera_id, url, use_gpu=_gpu_for_camera_type(cam_type), cam_type=cam_type)
    log.info(f"[CAMERA START] {camera_id} ({cam_type})")
    return {"status": "starting", "id": camera_id}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_camera_endpoint(camera_id: str):
    """Stop an RTSP camera."""
    camera_manager.stop_camera(camera_id)
    log.info(f"[CAMERA STOP] {camera_id}")
    return {"status": "stopped", "id": camera_id}


@app.get("/api/streams/status")
async def get_stream_status():
    """Get real status of all cameras."""
    status = camera_manager.get_status()

    # Ensure all 25 analytics cameras appear in status
    for i in range(NUM_ANALYTICS_CAMERAS):
        cam_id = f"analytics-{i + 1}"
        if cam_id not in status:
            status[cam_id] = {
                "state": "idle",
                "fps": 0,
                "type": "analytics",
            }

    # Ensure PTZ cameras appear
    for i in range(NUM_PTZ_CAMERAS):
        cam_id = f"ptz-{i + 1}"
        if cam_id not in status:
            status[cam_id] = {
                "state": "idle",
                "fps": 0,
                "type": "ptz",
            }

    return status


@app.post("/api/cameras/start-all")
async def start_all_cameras():
    """Start all configured cameras at once."""
    started = []
    for cam_id, url in CUSTOM_CAMERA_URLS.items():
        if not camera_manager.is_running(cam_id):
            cam_type = "ptz" if cam_id.startswith("ptz") else "analytics"
            camera_manager.start_camera(
                cam_id, url, use_gpu=_gpu_for_camera_type(cam_type), cam_type=cam_type
            )
            started.append(cam_id)
    log.info(f"[START ALL] Started {len(started)} cameras")
    return {"status": "ok", "started": started}


@app.post("/api/cameras/stop-all")
async def stop_all_cameras():
    """Stop all cameras."""
    camera_manager.stop_all()
    log.info("[STOP ALL] All cameras stopped")
    return {"status": "ok"}


# ============================================================
# SYSTEM HEALTH MONITORING
# ============================================================

@app.get("/api/system/health")
async def system_health():
    """Get system health information."""
    import torch

    cam_status = camera_manager.get_status()
    running_count = sum(1 for s in cam_status.values()
                       if isinstance(s, dict) and s.get("state") == "running")

    health = {
        "cameras": {
            "total_configured": len(CUSTOM_CAMERA_URLS),
            "running": running_count,
            "status": cam_status,
        },
        "detection": {
            "fps": round(state.detection_fps, 1),
            "total_frames": state.detection_count,
            "race_active": state.race_active,
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0,
        },
    }
    return health


# ============================================================
# MJPEG STREAM ENDPOINT (Fallback)
# ============================================================

def _get_frame_for_stream(cam_id: str) -> np.ndarray:
    """Get best available frame for a camera, with placeholder fallback."""
    state.mark_stream_access(cam_id)
    frame = state.get_annotated_frame(cam_id, max_age_sec=MAX_ANNOTATED_FRAME_AGE_SEC)
    if frame is None:
        frame = camera_manager.get_frame(cam_id)
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        label = f"{cam_id} - waiting..."
        cv2.putText(frame, label, (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def _encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
    if not ret:
        return None
    return jpeg.tobytes()


def mjpeg_generator(cam_id: str):
    """Yield MJPEG frames for a specific camera (by cam_id string)."""
    delay = 1.0 / MJPEG_FPS
    last_access_mark = 0.0

    while True:
        now = time.time()
        if now - last_access_mark >= 1.0:
            state.mark_stream_access(cam_id)
            last_access_mark = now

        # Try annotated frame first (analytics cameras with detection overlay)
        frame = state.get_annotated_frame(cam_id, max_age_sec=MAX_ANNOTATED_FRAME_AGE_SEC)

        if frame is None:
            # Try raw frame from camera manager
            frame = camera_manager.get_frame(cam_id)

        if frame is None:
            # Show placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            label = f"{cam_id} — waiting..."
            cv2.putText(frame, label, (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ret:
            time.sleep(delay)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(delay)


@app.get("/stream/cam{cam_id}")
async def mjpeg_stream(cam_id: str):
    """MJPEG video stream. Supports both numeric (cam1) and string (analytics-1) IDs."""
    # Support both "1" (old format) and "analytics-1" (new format)
    if cam_id.isdigit():
        cam_id_str = f"analytics-{cam_id}"
    else:
        cam_id_str = cam_id

    return StreamingResponse(
        mjpeg_generator(cam_id_str),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/{cam_id_full}")
async def mjpeg_stream_full(cam_id_full: str):
    """MJPEG stream using full cam_id (e.g. analytics-1, ptz-2)."""
    return StreamingResponse(
        mjpeg_generator(cam_id_full),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _snapshot_response(cam_id: str) -> Response:
    """Return a single JPEG frame for lightweight polling previews."""
    frame = _get_frame_for_stream(cam_id)
    jpeg_bytes = _encode_jpeg(frame)
    if jpeg_bytes is None:
        return Response(status_code=503)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/snapshot/cam{cam_id}")
async def snapshot_numeric(cam_id: str):
    """Single JPEG snapshot for numeric camera ids."""
    if cam_id.isdigit():
        cam_id_str = f"analytics-{cam_id}"
    else:
        cam_id_str = cam_id
    return _snapshot_response(cam_id_str)


@app.get("/snapshot/{cam_id_full}")
async def snapshot_full(cam_id_full: str):
    """Single JPEG snapshot for full camera ids."""
    return _snapshot_response(cam_id_full)


# ============================================================
# BACKGROUND BROADCAST TASKS
# ============================================================

async def ranking_broadcast_loop():
    """Periodically broadcast ranking updates to all WebSocket clients."""
    last_log_time = 0
    while True:
        await asyncio.sleep(BROADCAST_INTERVAL)

        if not ws_clients:
            continue

        rankings = state.get_combined_rankings()

        msg = {
            "type": "ranking_update",
            "rankings": rankings,
        }
        await broadcast(msg)

        # Log every 5 seconds
        now = time.time()
        if now - last_log_time > 5.0:
            names = [r.get("name", "?") for r in rankings]
            log.info(f"Broadcasting {len(rankings)} horses to {len(ws_clients)} clients: {names}")
            last_log_time = now


async def camera_detection_broadcast_loop():
    """Broadcast which cameras are detecting horses (every 1 second)."""
    while True:
        await asyncio.sleep(1.0)

        if not ws_clients:
            continue

        # Build camera → horses map
        presence = {}
        with state._lock:
            for cam_id, horses in state.camera_horse_presence.items():
                if horses:
                    presence[cam_id] = list(horses)

        if presence:
            msg = {
                "type": "camera_detection",
                "cameras": presence,
            }
            await broadcast(msg)


# Register WebRTC routes
setup_webrtc_routes(app, camera_manager, state)


# ============================================================
# MAIN
# ============================================================

# Global references for cleanup
_detector: Optional[MultiDetectionLoop] = None


def main():
    global _detector, USE_GPU, RACE_STATUS, SERVER_HOST, SERVER_PORT

    # Ensure selector loop policy is active in the runtime process on Windows.
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Race Vision Backend Server")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--video", nargs="+", default=None,
                        help="Local video file(s) played sequentially (test mode)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU decode (NVDEC)")
    parser.add_argument("--host", default=SERVER_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start race on launch")
    args = parser.parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    configure_clean_logging()

    USE_GPU = args.gpu
    camera_manager.set_gpu(USE_GPU)

    # Start detection loop
    _detector = MultiDetectionLoop()

    if args.video:
        # Video file test mode
        sources = args.video
        log.info(f"Video sources: {[Path(s).stem for s in sources]}")
        camera_manager.start_video_mode(sources)
        _detector._video_mode = True
    else:
        # RTSP mode — cameras are configured via API
        log.info("RTSP mode — configure cameras via /api/cameras/{id}")
        log.info(f"  Default RTSP: {args.url.split('@')[-1] if '@' in args.url else args.url}")
        _detector._video_mode = False

    _detector.start()
    log.info("Detection loop started")

    # Auto-start race if requested
    if args.auto_start:
        state.race_active = True
        RACE_STATUS = "active"
        log.info("Race auto-started")

    # Run FastAPI server
    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
    finally:
        log.info("Shutting down...")
        camera_manager.stop_all()
        _detector.stop()


if __name__ == "__main__":
    main()
