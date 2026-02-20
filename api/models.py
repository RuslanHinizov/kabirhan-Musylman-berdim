"""
API Data Models (Pydantic)

Defines the message contracts for:
- External analytics pipeline -> Backend (detection ingest)
- Backend -> Frontend (ranking, camera status, alerts)
"""

from typing import Optional
from pydantic import BaseModel


# ============================================================
# Analytics -> Backend (Detection Ingest)
# ============================================================

class DetectionItem(BaseModel):
    """Single detection from analytics pipeline."""
    participant_id: str          # e.g., "horse-3"
    color_class: str             # e.g., "green"
    bbox: list[float]            # [x1, y1, x2, y2]
    confidence: float            # CNN classifier confidence
    center_x: float              # Horizontal center of bbox (pixels)
    speed_mps: Optional[float] = None  # Estimated speed in m/s


class DetectionPayload(BaseModel):
    """Batch of detections from a single camera frame."""
    camera_id: str               # e.g., "analytics-7"
    frame_id: int                # Frame sequence number
    timestamp: float             # Unix timestamp with fractional seconds
    detections: list[DetectionItem]


# ============================================================
# Backend -> Frontend (WebSocket Messages)
# ============================================================

class RankingItem(BaseModel):
    """Single horse ranking entry."""
    type: str = "ranking_update"
    place: int
    participant_id: str
    gap: float                   # Gap to leader in meters
    confidence: float


class CameraStatusItem(BaseModel):
    """Camera health status."""
    type: str = "camera_status"
    camera_id: str
    online: bool
    fps: float
    latency_ms: float


class AlertItem(BaseModel):
    """System alert."""
    type: str = "alert"
    alert_type: str              # e.g., "camera_reconnect", "gpu_overload"
    message: str
