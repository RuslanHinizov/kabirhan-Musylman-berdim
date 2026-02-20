"""
PTZ Camera Control

Translates frontend PTZ commands to camera control protocol calls.
Supports ONVIF (future) and direct HTTP/serial protocols.

Commands arrive via WebRTC DataChannel or WebSocket for minimum latency.
"""

import asyncio
import logging
import time
from typing import Optional

log = logging.getLogger("race_server")

# Debounce: ignore commands arriving faster than this interval (seconds)
COMMAND_DEBOUNCE_SEC = 0.05  # 50ms

_last_command_time: dict[str, float] = {}


async def handle_ptz_command(cmd: dict) -> dict:
    """Handle a PTZ command from the frontend.

    Args:
        cmd: dict with keys:
            - camera_id: str (e.g., "ptz-1")
            - action: str ("move", "stop", "preset", "zoom")
            - pan: float (-1.0 to 1.0, for "move")
            - tilt: float (-1.0 to 1.0, for "move")
            - zoom: float (-1.0 to 1.0, for "move" or "zoom")
            - preset_id: int (for "preset")

    Returns:
        dict with status info
    """
    camera_id = cmd.get("camera_id", "unknown")
    action = cmd.get("action", "unknown")

    # Debounce rapid commands
    now = time.time()
    last = _last_command_time.get(camera_id, 0)
    if now - last < COMMAND_DEBOUNCE_SEC:
        return {"status": "debounced", "camera_id": camera_id}
    _last_command_time[camera_id] = now

    if action == "move":
        pan = cmd.get("pan", 0.0)
        tilt = cmd.get("tilt", 0.0)
        zoom = cmd.get("zoom", 0.0)
        log.debug(f"[PTZ] {camera_id} move: pan={pan}, tilt={tilt}, zoom={zoom}")
        # TODO: Send ONVIF ContinuousMove command
        return {"status": "ok", "action": "move", "camera_id": camera_id}

    elif action == "stop":
        log.debug(f"[PTZ] {camera_id} stop")
        # TODO: Send ONVIF Stop command
        return {"status": "ok", "action": "stop", "camera_id": camera_id}

    elif action == "preset":
        preset_id = cmd.get("preset_id", 1)
        log.debug(f"[PTZ] {camera_id} goto preset {preset_id}")
        # TODO: Send ONVIF GotoPreset command
        return {"status": "ok", "action": "preset", "camera_id": camera_id}

    elif action == "zoom":
        zoom = cmd.get("zoom", 0.0)
        log.debug(f"[PTZ] {camera_id} zoom: {zoom}")
        # TODO: Send zoom command
        return {"status": "ok", "action": "zoom", "camera_id": camera_id}

    else:
        log.warning(f"[PTZ] Unknown action: {action}")
        return {"status": "error", "message": f"Unknown action: {action}"}
