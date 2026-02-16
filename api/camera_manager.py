"""
Multi-Camera RTSP Manager

Manages 28 simultaneous RTSP camera readers (25 analytics + 3 PTZ).
Each camera runs in its own daemon thread with GPU-accelerated decode (NVDEC).

Architecture:
    CameraReader(Thread) — one per physical camera
        ↓ FFmpegReader (subprocess ffmpeg → raw BGR24 pipe)
        ↓ thread-safe frame buffer

    MultiCameraManager — manages all readers
        ↓ start/stop/get_frame per camera
        ↓ status tracking + auto-reconnect
"""

import time
import logging
import threading
import cv2
import numpy as np
from typing import Optional

from tools.test_rtsp import FFmpegReader, detect_codec, parse_resolution

log = logging.getLogger("race_server")


# ============================================================
# CAMERA READER — One thread per physical camera
# ============================================================

class CameraReader(threading.Thread):
    """Reads frames from a single RTSP camera in a background thread.

    Features:
        - GPU decode via NVDEC (h264_cuvid / hevc_cuvid)
        - Thread-safe frame buffer (latest frame always available)
        - Auto-reconnect with exponential backoff
        - State tracking: IDLE → CONNECTING → RUNNING → ERROR/STOPPED
    """

    # States
    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

    # Output caps keep 25+ camera mode stable under heavy RTSP loads.
    MAX_WIDTH_ANALYTICS = 1280
    MAX_WIDTH_PTZ = 1600

    def __init__(self, cam_id: str, rtsp_url: str, use_gpu: bool = True,
                 cam_type: str = "analytics"):
        super().__init__(daemon=True, name=f"CameraReader-{cam_id}")
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.use_gpu = use_gpu
        self.cam_type = cam_type  # "analytics" or "ptz"

        # Thread control
        self.running = False
        self._stop_event = threading.Event()

        # Thread-safe frame buffer
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_width: int = 0
        self._frame_height: int = 0

        # State
        self._state: str = self.IDLE
        self._state_lock = threading.Lock()

        # Reconnect settings
        self._base_delay = 1.0
        self._max_delay = 30.0
        self._current_delay = self._base_delay

        # FPS tracking
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start_time: float = 0.0
        self._last_frame_time: float = 0.0

        # Reader reference (for cleanup)
        self._reader: Optional[FFmpegReader] = None

        # Cache stream metadata so reconnect is fast and doesn't re-probe every time
        self._cached_codec: Optional[str] = None
        self._cached_source_resolution: Optional[tuple[int, int]] = None
        self._cached_output_resolution: Optional[tuple[int, int]] = None

    def run(self):
        """Main thread loop — connect, read frames, auto-reconnect."""
        self.running = True
        self._set_state(self.CONNECTING)

        while self.running and not self._stop_event.is_set():
            try:
                self._connect_and_read()
            except Exception as e:
                log.error(f"[{self.cam_id}] Unexpected error: {e}")
                self._set_state(self.ERROR)

            # Auto-reconnect with backoff
            if self.running and not self._stop_event.is_set():
                log.info(f"[{self.cam_id}] Reconnecting in {self._current_delay:.1f}s...")
                self._stop_event.wait(self._current_delay)
                self._current_delay = min(self._current_delay * 2, self._max_delay)

        self._set_state(self.STOPPED)
        log.info(f"[{self.cam_id}] Reader thread stopped")

    def _connect_and_read(self):
        """Single connection attempt: probe → connect → read loop."""
        self._set_state(self.CONNECTING)
        url_display = self.rtsp_url.split('@')[-1] if '@' in self.rtsp_url else self.rtsp_url
        log.info(f"[{self.cam_id}] Connecting to: {url_display}")

        # Detect codec (cached after first success)
        if self._cached_codec:
            codec = self._cached_codec
        else:
            codec = detect_codec(self.rtsp_url)
            self._cached_codec = codec
        log.info(f"[{self.cam_id}] Codec: {codec}")

        # Get source resolution (cached after first success)
        if self._cached_source_resolution:
            w, h = self._cached_source_resolution
        else:
            w, h = parse_resolution(self.rtsp_url)
            if w == 0 or h == 0:
                log.error(f"[{self.cam_id}] Could not detect resolution via probe, trying OpenCV direct...")
                # Last-resort: try OpenCV to grab one frame and get resolution
                try:
                    import cv2
                    cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            log.info(f"[{self.cam_id}] OpenCV direct got resolution: {w}x{h}")
                    cap.release()
                except Exception as e:
                    log.error(f"[{self.cam_id}] OpenCV direct failed: {e}")

        if w == 0 or h == 0:
            log.error(f"[{self.cam_id}] Could not detect resolution - all methods failed")
            self._set_state(self.ERROR)
            return

        out_w, out_h = self._choose_output_resolution(w, h)
        self._cached_source_resolution = (w, h)
        self._cached_output_resolution = (out_w, out_h)

        if (out_w, out_h) != (w, h):
            log.info(
                f"[{self.cam_id}] Resolution: {w}x{h} -> {out_w}x{out_h}, "
                f"GPU: {self.use_gpu}, Type: {self.cam_type}"
            )
        else:
            log.info(f"[{self.cam_id}] Resolution: {w}x{h}, GPU: {self.use_gpu}, Type: {self.cam_type}")

        # Create FFmpegReader
        reader = FFmpegReader(self.rtsp_url, out_w, out_h, gpu=self.use_gpu, codec=codec)
        reader.start()
        self._reader = reader

        # Verify first frame comes through
        first_ret, first_frame = reader.read()
        if not first_ret or first_frame is None:
            log.error(f"[{self.cam_id}] FFmpegReader started but first frame failed, "
                      f"resolution might be wrong ({w}x{h})")
            reader.release()
            self._reader = None
            # GPU decode can fail under heavy multi-stream load (NVDEC session pressure).
            # Fall back to OpenCV decoder before giving up.
            if self.use_gpu:
                log.warning(f"[{self.cam_id}] GPU decode first-frame failure; trying OpenCV fallback")
                self._connect_and_read_opencv(out_w, out_h)
                return
            self._set_state(self.ERROR)
            return

        # Some RTSP streams may decode as green/garbled with ffmpeg raw pipe.
        # Fallback to OpenCV reader for those cameras.
        if self._is_obviously_corrupted(first_frame):
            log.warning(f"[{self.cam_id}] FFmpeg frame appears corrupted; falling back to OpenCV decoder")
            reader.release()
            self._reader = None
            self._connect_and_read_opencv()
            return

        # Store first frame
        with self._frame_lock:
            self._frame = first_frame
            self._frame_width = out_w
            self._frame_height = out_h
            self._last_frame_time = time.time()

        self._set_state(self.RUNNING)
        self._current_delay = self._base_delay  # Reset backoff on success
        self._fps_start_time = time.time()
        self._frame_count = 1  # Already got first frame

        log.info(f"[{self.cam_id}] ✅ Connected and streaming ({out_w}x{out_h})")

        # Read loop
        bad_frames = 0
        while self.running and not self._stop_event.is_set():
            ret, frame = reader.read()
            if not ret:
                log.warning(f"[{self.cam_id}] Frame read failed, reconnecting...")
                break

            if self._is_obviously_corrupted(frame):
                bad_frames += 1
                if bad_frames >= 8:
                    log.warning(f"[{self.cam_id}] Repeated corrupted frames in ffmpeg mode; switching decoder")
                    break
                continue
            bad_frames = 0

            # Store frame (thread-safe)
            with self._frame_lock:
                self._frame = frame
                self._frame_width = out_w
                self._frame_height = out_h
                self._last_frame_time = time.time()

            # FPS tracking
            self._frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.time()

        # Cleanup
        reader.release()
        self._reader = None
        if self.running:
            if bad_frames >= 8:
                self._connect_and_read_opencv(out_w, out_h)
            elif self.use_gpu:
                log.warning(f"[{self.cam_id}] FFmpeg read loop failed; trying OpenCV fallback")
                self._connect_and_read_opencv(out_w, out_h)
            else:
                self._set_state(self.ERROR)

    def _connect_and_read_opencv(self, target_w: Optional[int] = None, target_h: Optional[int] = None):
        """Fallback decoder path using OpenCV VideoCapture."""
        self._set_state(self.CONNECTING)
        log.info(f"[{self.cam_id}] OpenCV fallback decoder starting")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)

        if not cap.isOpened():
            log.error(f"[{self.cam_id}] OpenCV fallback failed to open stream")
            cap.release()
            self._set_state(self.ERROR)
            return

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            log.error(f"[{self.cam_id}] OpenCV fallback first frame failed")
            cap.release()
            self._set_state(self.ERROR)
            return

        if target_w and target_h and (first_frame.shape[1] != target_w or first_frame.shape[0] != target_h):
            first_frame = cv2.resize(first_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if self._is_obviously_corrupted(first_frame):
            log.error(f"[{self.cam_id}] OpenCV fallback also produced corrupted frame")
            cap.release()
            self._set_state(self.ERROR)
            return

        h, w = first_frame.shape[:2]
        with self._frame_lock:
            self._frame = first_frame
            self._frame_width = w
            self._frame_height = h
            self._last_frame_time = time.time()

        self._set_state(self.RUNNING)
        self._current_delay = self._base_delay
        self._fps_start_time = time.time()
        self._frame_count = 1
        log.info(f"[{self.cam_id}] ✅ OpenCV fallback streaming ({w}x{h})")

        bad_frames = 0
        while self.running and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning(f"[{self.cam_id}] OpenCV fallback frame read failed, reconnecting...")
                break

            if target_w and target_h and (frame.shape[1] != target_w or frame.shape[0] != target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if self._is_obviously_corrupted(frame):
                bad_frames += 1
                if bad_frames >= 8:
                    log.warning(f"[{self.cam_id}] OpenCV fallback produced repeated corrupted frames")
                    break
                continue
            bad_frames = 0

            with self._frame_lock:
                self._frame = frame
                self._frame_width = frame.shape[1]
                self._frame_height = frame.shape[0]
                self._last_frame_time = time.time()

            self._frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.time()

        cap.release()
        if self.running:
            self._set_state(self.ERROR)

    def stop(self):
        """Signal this reader to stop."""
        self.running = False
        self._stop_event.set()
        if self._reader:
            self._reader.release()
            self._reader = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe). Returns None if no frame available."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_frame_dimensions(self) -> tuple:
        """Get current frame dimensions (width, height)."""
        with self._frame_lock:
            return self._frame_width, self._frame_height

    def get_state(self) -> str:
        """Get current connection state."""
        with self._state_lock:
            return self._state

    def get_fps(self) -> float:
        """Get measured input FPS."""
        return self._fps

    def get_last_frame_age(self) -> float:
        """Seconds since the last good frame was received."""
        with self._frame_lock:
            ts = self._last_frame_time
        if ts <= 0:
            return float("inf")
        return max(0.0, time.time() - ts)

    def _set_state(self, new_state: str):
        """Update state (thread-safe)."""
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            log.debug(f"[{self.cam_id}] State: {old} → {new_state}")

    @staticmethod
    def _is_obviously_corrupted(frame: Optional[np.ndarray]) -> bool:
        """Heuristic detector for green/garbled decode artifacts."""
        if frame is None or frame.size == 0:
            return True
        if frame.ndim != 3 or frame.shape[2] != 3:
            return True
        if frame.dtype != np.uint8:
            return True

        mean_bgr = frame.reshape(-1, 3).mean(axis=0)
        b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])

        # Typical corruption observed: almost pure green frame
        if g > 150.0 and b < 20.0 and r < 20.0:
            return True

        # If green dominates heavily and there is almost no texture, likely bad decode.
        texture = float(frame.std())
        if g > (b + r) * 2.2 and texture < 12.0:
            return True

        return False

    def _choose_output_resolution(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Downscale oversized streams to keep backend stable with many cameras."""
        max_w = self.MAX_WIDTH_ANALYTICS if self.cam_type == "analytics" else self.MAX_WIDTH_PTZ
        if src_w <= max_w:
            out_w, out_h = src_w, src_h
        else:
            scale = max_w / float(src_w)
            out_w = int(src_w * scale)
            out_h = int(src_h * scale)

        # Keep dimensions even for downstream encoders and OpenCV operations
        out_w = max(2, out_w - (out_w % 2))
        out_h = max(2, out_h - (out_h % 2))
        return out_w, out_h


# ============================================================
# VIDEO FILE READER — For testing with local video files
# ============================================================

class VideoFileReader(threading.Thread):
    """Reads frames from local video files sequentially (for testing).

    Mimics a camera by writing frames to the same interface as CameraReader.
    Loops through video files, switching cam_id for each.
    """

    def __init__(self, sources: list, cam_id_prefix: str = "analytics"):
        super().__init__(daemon=True, name="VideoFileReader")
        self.sources = sources
        self.cam_id_prefix = cam_id_prefix
        self.running = False

        # Per-camera frame buffers (same as MultiCameraManager expects)
        self._frames: dict[str, Optional[np.ndarray]] = {}
        self._frame_lock = threading.Lock()
        self._active_cam_id: str = ""
        self._video_index: int = 0
        self._frame_widths: dict[str, int] = {}
        self._frame_heights: dict[str, int] = {}

    def run(self):
        import cv2
        from pathlib import Path

        self.running = True

        while self.running:
            for cam_idx, source in enumerate(self.sources):
                if not self.running:
                    break

                cam_id = f"{self.cam_id_prefix}-{cam_idx + 1}"
                cap = cv2.VideoCapture(source)

                if not cap.isOpened():
                    log.error(f"Cannot open video: {source}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                delay = 1.0 / fps
                name = Path(source).stem
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                self._video_index += 1
                self._active_cam_id = cam_id
                log.info(f"[VideoFile] Playing: {name} as {cam_id} @ {fps:.0f} fps (video #{self._video_index})")

                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        log.info(f"[VideoFile] {name} ended")
                        break

                    with self._frame_lock:
                        self._frames[cam_id] = frame
                        self._frame_widths[cam_id] = w
                        self._frame_heights[cam_id] = h

                    time.sleep(delay)

                cap.release()

            if self.running:
                log.info("[VideoFile] All videos played, looping...")

    def stop(self):
        self.running = False

    def get_frame(self, cam_id: str) -> Optional[np.ndarray]:
        with self._frame_lock:
            frame = self._frames.get(cam_id)
            return frame.copy() if frame is not None else None

    def get_active_cam_id(self) -> str:
        return self._active_cam_id

    def get_video_index(self) -> int:
        return self._video_index

    def get_frame_dimensions(self, cam_id: str) -> tuple:
        with self._frame_lock:
            return self._frame_widths.get(cam_id, 0), self._frame_heights.get(cam_id, 0)


# ============================================================
# MULTI-CAMERA MANAGER — Manages all camera readers
# ============================================================

class MultiCameraManager:
    """Manages multiple RTSP camera readers simultaneously.

    Provides a unified interface to start/stop cameras and access their frames.
    Supports both RTSP cameras and video file fallback.
    """

    def __init__(self):
        self._readers: dict[str, CameraReader] = {}
        self._lock = threading.Lock()
        self._use_gpu: bool = True  # Default GPU decode

        # Video file fallback (for --video mode)
        self._video_reader: Optional[VideoFileReader] = None

    def set_gpu(self, use_gpu: bool):
        """Set whether new cameras should use GPU decode."""
        self._use_gpu = use_gpu

    # --- RTSP Camera Management ---

    def start_camera(self, cam_id: str, rtsp_url: str,
                     use_gpu: Optional[bool] = None,
                     cam_type: str = "analytics"):
        """Start an RTSP camera reader. Restarts if already running."""
        gpu = use_gpu if use_gpu is not None else self._use_gpu

        with self._lock:
            # Stop existing reader if any
            if cam_id in self._readers:
                old = self._readers[cam_id]
                old.stop()
                old.join(timeout=5)
                del self._readers[cam_id]

            # Create and start new reader
            reader = CameraReader(cam_id, rtsp_url, use_gpu=gpu, cam_type=cam_type)
            self._readers[cam_id] = reader
            reader.start()
            log.info(f"[Manager] Started camera: {cam_id} ({cam_type})")

    def stop_camera(self, cam_id: str):
        """Stop a specific camera reader."""
        with self._lock:
            reader = self._readers.pop(cam_id, None)
        if reader:
            reader.stop()
            reader.join(timeout=5)
            log.info(f"[Manager] Stopped camera: {cam_id}")

    def stop_all(self):
        """Stop all camera readers."""
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()

        for reader in readers:
            reader.stop()
        for reader in readers:
            reader.join(timeout=5)

        # Also stop video reader if any
        if self._video_reader:
            self._video_reader.stop()
            self._video_reader = None

        log.info("[Manager] All cameras stopped")

    def is_running(self, cam_id: str) -> bool:
        """Check if a camera is currently running."""
        with self._lock:
            reader = self._readers.get(cam_id)
            return reader is not None and reader.get_state() == CameraReader.RUNNING

    # --- Video File Fallback ---

    def start_video_mode(self, sources: list, cam_id_prefix: str = "analytics"):
        """Start video file reader (for --video testing mode)."""
        self._video_reader = VideoFileReader(sources, cam_id_prefix)
        self._video_reader.start()
        log.info(f"[Manager] Video mode started with {len(sources)} files")

    # --- Frame Access ---

    def get_frame(self, cam_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a specific camera."""
        # Try RTSP reader first
        with self._lock:
            reader = self._readers.get(cam_id)
        if reader:
            return reader.get_frame()

        # Fall back to video file reader
        if self._video_reader:
            return self._video_reader.get_frame(cam_id)

        return None

    def get_frame_dimensions(self, cam_id: str) -> tuple:
        """Get frame dimensions for a specific camera (width, height)."""
        with self._lock:
            reader = self._readers.get(cam_id)
        if reader:
            return reader.get_frame_dimensions()

        if self._video_reader:
            return self._video_reader.get_frame_dimensions(cam_id)

        return 0, 0

    def get_all_frames(self) -> dict:
        """Get latest frames from all cameras that have frames available."""
        frames = {}

        with self._lock:
            reader_list = list(self._readers.items())

        for cam_id, reader in reader_list:
            frame = reader.get_frame()
            if frame is not None:
                frames[cam_id] = frame

        # Include video file frames
        if self._video_reader:
            active = self._video_reader.get_active_cam_id()
            if active:
                frame = self._video_reader.get_frame(active)
                if frame is not None:
                    frames[active] = frame

        return frames

    # --- Status ---

    def get_status(self) -> dict:
        """Get status of all cameras."""
        status = {}
        stale_threshold_sec = 3.0

        with self._lock:
            for cam_id, reader in self._readers.items():
                state = reader.get_state()
                age = reader.get_last_frame_age()
                # Running but no fresh frames means stream is effectively reconnecting.
                if state == CameraReader.RUNNING and age > stale_threshold_sec:
                    state = CameraReader.CONNECTING

                status[cam_id] = {
                    "state": state,
                    "fps": round(reader.get_fps(), 1),
                    "type": reader.cam_type,
                    "frameAgeSec": round(age, 2) if age != float("inf") else None,
                }

        # Video mode status
        if self._video_reader:
            active = self._video_reader.get_active_cam_id()
            if active:
                status[active] = {
                    "state": "running",
                    "fps": 25.0,
                    "type": "video",
                }

        return status

    def get_active_cameras(self) -> list:
        """Get list of cam_ids in RUNNING state."""
        active = []
        with self._lock:
            for cam_id, reader in self._readers.items():
                if reader.get_state() == CameraReader.RUNNING:
                    active.append(cam_id)

        if self._video_reader:
            active_vid = self._video_reader.get_active_cam_id()
            if active_vid:
                active.append(active_vid)

        return active

    def get_analytics_cameras(self) -> list:
        """Get list of analytics camera IDs (for detection scheduling)."""
        cameras = []
        with self._lock:
            for cam_id, reader in self._readers.items():
                if reader.cam_type == "analytics" and reader.get_state() == CameraReader.RUNNING:
                    cameras.append(cam_id)
        return cameras

    def get_video_reader(self) -> Optional[VideoFileReader]:
        """Get the video file reader (for backward compatibility)."""
        return self._video_reader
