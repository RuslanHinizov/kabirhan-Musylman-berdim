"""
WebRTC Streaming Server

Provides low-latency H.264 video streaming via WebRTC, replacing MJPEG.
Uses aiortc for Python WebRTC implementation.

Features:
    - Per-camera WebRTC tracks (analytics: annotated, PTZ: raw high-quality)
    - HTTP signaling (POST /api/webrtc/offer → SDP answer)
    - Auto-cleanup on disconnect
    - NVENC hardware encoding when available

Usage:
    Endpoints are registered on the FastAPI app via setup_webrtc_routes().
"""

import asyncio
import fractions
import logging
import time
import cv2
import numpy as np
from typing import Optional
from fastapi import Request

log = logging.getLogger("race_server")
MAX_ANNOTATED_FRAME_AGE_SEC = 0.9

# Try importing aiortc (optional dependency)
try:
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
    from aiortc.contrib.media import MediaRelay
    WEBRTC_AVAILABLE = True
    log.info("WebRTC (aiortc) available")
except ImportError:
    WEBRTC_AVAILABLE = False
    log.warning("aiortc not installed — WebRTC disabled, using MJPEG fallback")


# Active peer connections (for cleanup)
peer_connections: set = set()


class CameraVideoTrack(MediaStreamTrack if WEBRTC_AVAILABLE else object):
    """WebRTC video track that serves frames from a specific camera.

    For analytics cameras: serves annotated frames (with detection overlays)
    For PTZ cameras: serves raw frames (high quality, no processing delay)
    """

    kind = "video"

    def __init__(self, cam_id: str, camera_manager, shared_state, fps: int = 25):
        if WEBRTC_AVAILABLE:
            super().__init__()
        self.cam_id = cam_id
        self.camera_manager = camera_manager
        self.shared_state = shared_state
        self._fps = fps
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 90000)
        self._start_time = time.time()
        self._max_width = 960 if cam_id.startswith("analytics") else 1280

    async def recv(self):
        """Called by aiortc when it needs the next frame."""
        # Wait for next frame timing
        pts_time = self._timestamp / 90000
        real_time = time.time() - self._start_time
        wait = pts_time - real_time
        if wait > 0:
            await asyncio.sleep(wait)

        frame = None

        # For analytics cameras: try annotated frame first (detection overlays)
        if self.cam_id.startswith("analytics"):
            frame = self.shared_state.get_annotated_frame(
                self.cam_id,
                max_age_sec=MAX_ANNOTATED_FRAME_AGE_SEC,
            )

        # For PTZ cameras or if no annotated frame: use raw camera frame
        if frame is None:
            frame = self.camera_manager.get_frame(self.cam_id)

        # Fallback: black placeholder
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = self._normalize_frame(frame)

        # Convert numpy BGR to av.VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._timestamp
        video_frame.time_base = self._time_base
        self._timestamp += int(90000 / self._fps)

        return video_frame

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame shape/dtype before handing it to aiortc encoder.

        This avoids green/garbled output caused by odd dimensions, wrong channels,
        or non-uint8 frames coming from different decoder paths.
        """
        if frame is None or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Ensure 3-channel BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Keep dimensions even for YUV420 encoders
        even_w = w - (w % 2)
        even_h = h - (h % 2)
        if even_w != w or even_h != h:
            frame = frame[:even_h, :even_w]
            h, w = frame.shape[:2]

        # Limit output size to keep many parallel WebRTC streams stable
        if w > self._max_width:
            scale = self._max_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w -= new_w % 2
            new_h -= new_h % 2
            new_w = max(new_w, 2)
            new_h = max(new_h, 2)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        return frame


def setup_webrtc_routes(app, camera_manager, shared_state):
    """Register WebRTC signaling endpoints on the FastAPI app.

    Args:
        app: FastAPI application instance
        camera_manager: MultiCameraManager instance
        shared_state: SharedState instance
    """
    if not WEBRTC_AVAILABLE:
        # Register a fallback endpoint that tells clients WebRTC is unavailable
        @app.post("/api/webrtc/offer")
        async def webrtc_unavailable():
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"error": "WebRTC not available. Install aiortc: pip install aiortc"}
            )

        @app.get("/api/webrtc/status")
        async def webrtc_status():
            return {"available": False, "reason": "aiortc not installed"}

        return

    async def cleanup_stale_peers() -> int:
        stale_states = {"failed", "closed", "disconnected"}
        stale = [pc for pc in list(peer_connections) if pc.connectionState in stale_states]
        for pc in stale:
            peer_connections.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
        return len(stale)

    @app.post("/api/webrtc/offer")
    async def webrtc_offer(request: Request):
        """Handle WebRTC offer from client. Returns SDP answer."""
        await cleanup_stale_peers()
        body = await request.json()
        cam_id = body.get("camId", "analytics-1")
        offer_sdp = body.get("sdp", "")
        offer_type = body.get("type", "offer")

        log.debug(f"[WebRTC] Offer received for {cam_id}")

        # Create peer connection
        pc = RTCPeerConnection()
        peer_connections.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            if state == "connected":
                log.debug(f"[WebRTC] {cam_id} connection state: {state}")
            elif state in ("failed", "closed", "disconnected"):
                log.warning(f"[WebRTC] {cam_id} connection state: {state}")
            if state in ("failed", "closed", "disconnected"):
                peer_connections.discard(pc)
                await pc.close()

        # Add camera video track
        fps = 10 if cam_id.startswith("analytics") else 20
        track = CameraVideoTrack(cam_id, camera_manager, shared_state, fps=fps)
        pc.addTrack(track)

        # Set remote description (client's offer)
        offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        log.debug(f"[WebRTC] Answer sent for {cam_id}")

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    @app.get("/api/webrtc/status")
    async def webrtc_status():
        """Get WebRTC server status."""
        await cleanup_stale_peers()
        return {
            "available": True,
            "active_connections": len(peer_connections),
        }

    @app.post("/api/webrtc/close-all")
    async def webrtc_close_all():
        """Close all WebRTC connections."""
        count = len(peer_connections)
        for pc in list(peer_connections):
            await pc.close()
        peer_connections.clear()
        log.info(f"[WebRTC] Closed {count} connections")
        return {"status": "ok", "closed": count}

    log.info("WebRTC signaling endpoints registered")
