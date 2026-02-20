"""
Centralized Configuration

Loads settings from .env file (if present) with sensible defaults.
All hardcoded constants from server.py are centralized here.
"""

import os
from pathlib import Path

# Try loading .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on environment or defaults


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


# ============================================================
# NETWORK
# ============================================================
DEFAULT_RTSP_URL = _env("RTSP_DEFAULT_URL",
                        "rtsp://admin:admin@192.168.1.100:554/stream")
SERVER_HOST = _env("SERVER_HOST", "0.0.0.0")
SERVER_PORT = _env_int("SERVER_PORT", 8000)

# ============================================================
# MODELS
# ============================================================
MODEL_PATH_YOLO = _env("MODEL_PATH_YOLO", "yolov8s.pt")
MODEL_PATH_COLOR = _env("MODEL_PATH_COLOR", "models/color_classifier.pt")

# ============================================================
# DETECTION
# ============================================================
DETECTION_INTERVAL = _env_float("DETECTION_INTERVAL", 0.08)          # ~12.5 fps (tuned for RTX 3060)
PREVIEW_DETECTION_INTERVAL = _env_float("PREVIEW_DETECTION_INTERVAL", 0.35)
PREVIEW_MAX_CAMERAS_PER_CYCLE = _env_int("PREVIEW_MAX_CAMERAS_PER_CYCLE", 5)
PREVIEW_ACCESS_FRESHNESS_SEC = _env_float("PREVIEW_ACCESS_FRESHNESS_SEC", 8.0)
DETECTION_MODE = _env("DETECTION_MODE", "internal")  # "internal" or "external"

# ============================================================
# BROADCAST
# ============================================================
BROADCAST_INTERVAL = _env_float("BROADCAST_INTERVAL", 0.20)          # 5 Hz
MJPEG_QUALITY = _env_int("MJPEG_QUALITY", 75)
MJPEG_FPS = _env_int("MJPEG_FPS", 25)
MAX_ANNOTATED_FRAME_AGE_SEC = _env_float("MAX_ANNOTATED_FRAME_AGE_SEC", 0.9)

# ============================================================
# CAMERAS
# ============================================================
NUM_ANALYTICS_CAMERAS = _env_int("NUM_ANALYTICS_CAMERAS", 25)
NUM_PTZ_CAMERAS = _env_int("NUM_PTZ_CAMERAS", 4)
PTZ_GPU_ANALYTICS_THRESHOLD = _env_int("PTZ_GPU_ANALYTICS_THRESHOLD", 8)

# ============================================================
# TRACK
# ============================================================
TRACK_LENGTH = _env_int("TRACK_LENGTH", 2500)
CAMERA_TRACK_M = _env_float("CAMERA_TRACK_M", 100.0)

# ============================================================
# GPU
# ============================================================
USE_GPU = _env_bool("USE_GPU", False)  # Overridden by --gpu CLI flag

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = _env("LOG_LEVEL", "INFO")
