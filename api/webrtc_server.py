"""
WebRTC Streaming Server

Provides low-latency H.264 video streaming via WebRTC, replacing MJPEG.
Uses aiortc for Python WebRTC implementation.

Features:
    - Per-camera WebRTC tracks (analytics: annotated, PTZ: raw high-quality)
    - MediaRelay: single encode per camera shared across all viewers
    - HTTP signaling (POST /api/webrtc/offer -> SDP answer)
    - DataChannel for low-latency PTZ control commands
    - Auto-cleanup on disconnect

Usage:
    Endpoints are registered on the FastAPI app via setup_webrtc_routes().
"""

import asyncio
import fractions
import json
import logging
import time
import cv2
import numpy as np
from typing import Optional
from fastapi import Request

log = logging.getLogger("race_server")
MAX_ANNOTATED_FRAME_AGE_SEC = 0.9
# PTZ frames older than this are considered stale (camera likely moved)
MAX_PTZ_FRAME_AGE_SEC = 0.15

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

# MediaRelay: share one encode pipeline across multiple viewers per camera
_media_relay: Optional["MediaRelay"] = None
_source_tracks: dict[str, "CameraVideoTrack"] = {}


def _get_relay():
    """Lazy-init the global MediaRelay instance."""
    global _media_relay
    if _media_relay is None and WEBRTC_AVAILABLE:
        _media_relay = MediaRelay()
    return _media_relay


class CameraVideoTrack(MediaStreamTrack if WEBRTC_AVAILABLE else object):
    """WebRTC video track that serves frames from a specific camera.

    For analytics cameras: serves annotated frames (with detection overlays)
    For PTZ cameras: serves raw frames directly (no annotation lookup delay)
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
        self._is_ptz = cam_id.startswith("ptz")
        # PTZ: full resolution for public display quality
        # Analytics: thumbnails, lower quality acceptable
        self._max_width = 1920 if self._is_ptz else 960
        # Track previous frame to detect stale/duplicate frames from RTSP buffer
        self._prev_frame_id = None
        self._last_good_frame = None
        self._keyframe_counter = 0

    async def recv(self):
        """Called by aiortc when it needs the next frame."""
        # Wait for next frame timing
        pts_time = self._timestamp / 90000
        real_time = time.time() - self._start_time
        wait = pts_time - real_time
        if wait > 0:
            await asyncio.sleep(wait)

        frame = None

        if self._is_ptz:
            # PTZ: always raw camera frame — skip annotation lookup for minimum latency
            frame = self.camera_manager.get_frame(self.cam_id)
        else:
            # Analytics: try annotated frame first (detection overlays)
            frame = self.shared_state.get_annotated_frame(
                self.cam_id,
                max_age_sec=MAX_ANNOTATED_FRAME_AGE_SEC,
            )
            if frame is None:
                frame = self.camera_manager.get_frame(self.cam_id)

        # Fallback: black placeholder
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = self._normalize_frame(frame)

        # Cache good frame for PTZ
        if self._is_ptz and frame is not None and frame.size > 0:
            self._last_good_frame = frame

        # Convert numpy BGR to av.VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._timestamp
        video_frame.time_base = self._time_base
        self._timestamp += int(90000 / self._fps)

        # Force keyframe more often for PTZ to recover quickly from movement blur
        if self._is_ptz:
            self._keyframe_counter += 1
            if self._keyframe_counter >= self._fps:  # Every ~1 second
                video_frame.key_frame = True
                self._keyframe_counter = 0

        return video_frame

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame shape/dtype before handing it to aiortc encoder."""
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


def _get_or_create_relay_track(cam_id: str, camera_manager, shared_state):
    """Get a relayed track for a camera. Creates the source track on first request.

    MediaRelay shares the encoding work: one CameraVideoTrack produces frames,
    and all viewers receive copies of the encoded packets without re-encoding.
    """
    relay = _get_relay()
    if relay is None:
        # Fallback: no relay, create direct track
        fps = 10 if cam_id.startswith("analytics") else 15
        return CameraVideoTrack(cam_id, camera_manager, shared_state, fps=fps)

    # Create source track if it doesn't exist or has been stopped
    if cam_id not in _source_tracks or _source_tracks[cam_id]._start_time == 0:
        # PTZ: 25fps for smooth live view, Analytics: 10fps (detection overlays)
        fps = 25 if cam_id.startswith("ptz") else 10
        source = CameraVideoTrack(cam_id, camera_manager, shared_state, fps=fps)
        _source_tracks[cam_id] = source
        log.info(f"[WebRTC] Created source track for {cam_id} at {fps} fps")

    # Subscribe returns a new track that shares the encoded output
    return relay.subscribe(_source_tracks[cam_id])


def setup_webrtc_routes(app, camera_manager, shared_state):
    """Register WebRTC signaling endpoints on the FastAPI app."""
    if not WEBRTC_AVAILABLE:
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

    # PTZ command handler (placeholder — extend with ONVIF/Pelco integration)
    ptz_command_handler = None
    try:
        from api.ptz_controller import handle_ptz_command
        ptz_command_handler = handle_ptz_command
    except ImportError:
        pass

    async def cleanup_stale_peers() -> int:
        stale_states = {"failed", "closed", "disconnected"}
        stale = [pc for pc in list(peer_connections) if pc.connectionState in stale_states]
        for pc in stale:
            peer_connections.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
        if stale:
            log.debug(f"[WebRTC] Cleaned up {len(stale)} stale peers")
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
            conn_state = pc.connectionState
            if conn_state == "connected":
                log.debug(f"[WebRTC] {cam_id} connected")
            if conn_state in ("failed", "closed", "disconnected"):
                log.warning(f"[WebRTC] {cam_id} state: {conn_state}")
                peer_connections.discard(pc)
                await pc.close()

        # DataChannel for PTZ commands (low-latency control path)
        if cam_id.startswith("ptz"):
            @pc.on("datachannel")
            async def on_datachannel(channel):
                log.info(f"[WebRTC] DataChannel opened for PTZ: {cam_id}")

                @channel.on("message")
                async def on_message(message):
                    try:
                        cmd = json.loads(message)
                        cmd["camera_id"] = cam_id
                        if ptz_command_handler:
                            result = await ptz_command_handler(cmd)
                            channel.send(json.dumps({"ack": True, **result}))
                        else:
                            log.debug(f"[PTZ] Command received (no handler): {cmd}")
                            channel.send(json.dumps({"ack": True, "status": "no_handler"}))
                    except Exception as e:
                        log.error(f"[PTZ] DataChannel error: {e}")
                        channel.send(json.dumps({"ack": False, "error": str(e)}))

        # Add camera video track via MediaRelay (shared encoding)
        track = _get_or_create_relay_track(cam_id, camera_manager, shared_state)
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
            "relay_tracks": list(_source_tracks.keys()),
        }

    @app.post("/api/webrtc/close-all")
    async def webrtc_close_all():
        """Close all WebRTC connections."""
        count = len(peer_connections)
        for pc in list(peer_connections):
            await pc.close()
        peer_connections.clear()
        _source_tracks.clear()
        log.info(f"[WebRTC] Closed {count} connections, cleared relay tracks")
        return {"status": "ok", "closed": count}

    log.info("WebRTC signaling endpoints registered (MediaRelay enabled)")
