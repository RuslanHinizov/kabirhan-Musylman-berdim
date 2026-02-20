"""
Smart Multi-Camera Detection System

Manages per-camera detection state and intelligently schedules which
cameras to process each cycle based on horse presence and movement.

Detection is ONLY performed on analytics cameras (25 cameras).
PTZ cameras are purely for broadcast — no detection.

Scheduling priorities:
    HIGH  — Horses currently detected or expected → 10 fps
    LOW   — Adjacent cameras to horse-active ones → 2 fps
    IDLE  — No horses nearby → 0.5 fps (scan)

Horse handoff:
    When horses approach edge of camera N frame → camera N+1 promoted to HIGH.
"""

import time
import logging
from typing import Optional
from collections import Counter

log = logging.getLogger("race_server")


# ============================================================
# PER-CAMERA DETECTION STATE
# ============================================================

class CameraDetectionState:
    """Maintains all tracking state for a single analytics camera.

    This contains the per-camera equivalent of what the old DetectionLoop
    kept globally: smooth_x, speed, filter state, voting, etc.
    """

    # Filter thresholds — tuned for accuracy over speed
    CONF_THRESHOLD = 0.55       # F1: min classifier confidence (was 0.45)
    HSV_SKIP_CONF = 0.85        # F2: skip HSV check only with very high CNN conf (was 0.80)
    MAX_SPEED_MPS = 160.0       # F3: tolerant for real RTSP jitter
    TEMPORAL_WINDOW = 6          # F4: sliding window frames (was 4)
    TEMPORAL_MIN = 2             # F4: require 2+ frames confirmation (was 1)
    CAMERA_TRACK_M = 100.0       # each analytics camera = 100 meters

    # How long (seconds) a color stays valid in smooth_x after last detection
    STALE_TIMEOUT = 3.0

    def __init__(self, cam_id: str, cam_index: int):
        self.cam_id = cam_id
        self.cam_index = cam_index          # 0-24
        self.track_start_m = cam_index * self.CAMERA_TRACK_M    # e.g. 0, 100, 200...
        self.track_end_m = (cam_index + 1) * self.CAMERA_TRACK_M

        # Smoothed X positions per color (EMA, alpha=0.12)
        self.smooth_x: dict[str, float] = {}
        self.smooth_alpha = 0.12

        # Per-color last-seen timestamp (for staleness eviction)
        self.color_last_seen: dict[str, float] = {}

        # Speed tracking per color (m/s, EMA, alpha=0.08 for smoother output)
        self.speed: dict[str, float] = {}
        self.speed_alpha = 0.08

        # Last known position per color: (pos_m, timestamp)
        self.last_pos: dict[str, tuple] = {}

        # Temporal confirmation: frame numbers where color was detected
        self.det_frames: dict[str, list] = {}

        # Detection frame counter
        self.frame_number: int = 0

        # Voting for color order — larger window for stability
        self.live_votes: list = []
        self.LIVE_VOTE_WINDOW = 20     # was 15
        self.LIVE_MIN_VOTES = 8        # was 5
        self.current_order: list = []
        self.order_changes: int = 0

        # Filter statistics
        self.filter_stats = {
            "total": 0, "pass_f1": 0, "pass_f2": 0,
            "pass_f3": 0, "pass_f4": 0
        }

        # Scheduling
        self.priority: str = "idle"       # "high" | "low" | "idle"
        self.horses_present: set = set()  # colors currently detected
        self.expected_horses: set = set() # colors expected from adjacent camera handoff
        self.last_detection_time: float = 0.0
        self.last_process_time: float = 0.0

        # Confidence per color (for ranking merger)
        self.color_confidence: dict[str, float] = {}

    def reset(self):
        """Reset state (e.g., when camera reconnects)."""
        self.smooth_x.clear()
        self.color_last_seen.clear()
        self.speed.clear()
        self.last_pos.clear()
        self.det_frames.clear()
        self.frame_number = 0
        self.live_votes.clear()
        self.current_order.clear()
        self.filter_stats = {
            "total": 0, "pass_f1": 0, "pass_f2": 0,
            "pass_f3": 0, "pass_f4": 0
        }
        self.horses_present.clear()
        self.expected_horses.clear()
        self.color_confidence.clear()

    def update_smooth_x(self, color: str, raw_x: float):
        """Update smoothed X position for a color."""
        if color in self.smooth_x:
            self.smooth_x[color] += self.smooth_alpha * (raw_x - self.smooth_x[color])
        else:
            self.smooth_x[color] = raw_x
        self.color_last_seen[color] = time.time()

    def evict_stale_colors(self):
        """Remove colors not seen for STALE_TIMEOUT seconds.

        Prevents old smooth_x entries from leaking into RankingMerger
        after a horse has left this camera's view.
        """
        now = time.time()
        stale = [
            c for c, t in self.color_last_seen.items()
            if now - t > self.STALE_TIMEOUT
        ]
        for c in stale:
            self.smooth_x.pop(c, None)
            self.speed.pop(c, None)
            self.last_pos.pop(c, None)
            self.det_frames.pop(c, None)
            self.color_last_seen.pop(c, None)
            self.color_confidence.pop(c, None)

    def update_speed(self, color: str, raw_speed: float):
        """Update EMA speed for a color."""
        if color in self.speed:
            self.speed[color] += self.speed_alpha * (raw_speed - self.speed[color])
        else:
            self.speed[color] = raw_speed

    def add_temporal_detection(self, color: str):
        """Record that this color was detected in current frame (F4)."""
        if color not in self.det_frames:
            self.det_frames[color] = []
        self.det_frames[color].append(self.frame_number)
        # Keep only last TEMPORAL_WINDOW entries
        self.det_frames[color] = self.det_frames[color][-self.TEMPORAL_WINDOW:]

    def check_temporal(self, color: str) -> bool:
        """Check if color passes F4: temporal confirmation."""
        frames = self.det_frames.get(color, [])
        recent = [f for f in frames if f >= self.frame_number - self.TEMPORAL_WINDOW]
        return len(recent) >= self.TEMPORAL_MIN

    def update_live_votes(self, order: list):
        """Add a vote and update live order.

        Tie-breaking: when two orders have the same vote count, keep the
        current order (stability) or prefer the one that appeared more
        recently to avoid non-deterministic flickering.
        """
        self.live_votes.append(tuple(order))
        if len(self.live_votes) > self.LIVE_VOTE_WINDOW:
            self.live_votes = self.live_votes[-self.LIVE_VOTE_WINDOW:]

        if len(self.live_votes) >= self.LIVE_MIN_VOTES:
            counts = Counter(self.live_votes)
            top_count = counts.most_common(1)[0][1]
            # All orders tied at the top count
            tied = [o for o, c in counts.items() if c == top_count]

            if len(tied) == 1:
                best = list(tied[0])
            else:
                # Tie: prefer current order if it's among the tied; otherwise
                # pick the most recently voted order for stability
                current_tuple = tuple(self.current_order) if self.current_order else None
                if current_tuple in tied:
                    best = list(current_tuple)
                else:
                    # Pick the last-voted order among tied candidates
                    for vote in reversed(self.live_votes):
                        if vote in tied:
                            best = list(vote)
                            break
                    else:
                        best = list(tied[0])

            if best != self.current_order:
                self.current_order = best
                self.order_changes += 1


# ============================================================
# SMART DETECTION SCHEDULER
# ============================================================

class SmartDetectionScheduler:
    """Decides which analytics cameras to process each detection cycle.

    Algorithm:
    1. Track horse positions across cameras
    2. Classify cameras into HIGH/LOW/IDLE priority
    3. Build processing queue respecting GPU budget
    4. Handle horse handoff between adjacent cameras
    """

    # Configuration — tuned for RTX 3060 with TensorRT (~10-12ms/frame)
    try:
        from api.config import _env_int
        MAX_CAMERAS_PER_CYCLE = _env_int("MAX_CAMERAS_PER_CYCLE", 8)
    except ImportError:
        MAX_CAMERAS_PER_CYCLE = 8

    HIGH_PRIORITY_INTERVAL = 0.08  # 12.5 fps (tuned for RTX 3060)
    LOW_PRIORITY_INTERVAL = 0.35   # ~3 fps
    IDLE_SCAN_INTERVAL = 1.5       # ~0.7 fps

    # Handoff: when horse X position > this fraction of frame width → expect at next camera
    HANDOFF_THRESHOLD = 0.85

    # Grace period: keep camera HIGH for this long after losing detection
    GRACE_PERIOD = 3.0

    # Max time expected_horses stays valid without actual detection
    EXPECTED_TIMEOUT = 5.0

    def __init__(self, camera_states: dict[str, CameraDetectionState]):
        self.camera_states = camera_states  # cam_id → CameraDetectionState
        self._last_scan_index = 0  # Round-robin for IDLE scanning
        # Track when expected_horses was set: cam_id → timestamp
        self._expected_set_time: dict[str, float] = {}

    def get_processing_queue(self) -> list:
        """Get ordered list of cam_ids to process this cycle.

        Returns at most MAX_CAMERAS_PER_CYCLE cameras, prioritized:
        1. All HIGH priority cameras (if overdue for detection)
        2. LOW priority cameras (round-robin if budget allows)
        3. One IDLE camera (scan)
        """
        now = time.time()
        high_queue = []
        low_queue = []
        idle_queue = []

        for cam_id, cam_state in self.camera_states.items():
            elapsed = now - cam_state.last_process_time

            if cam_state.priority == "high":
                if elapsed >= self.HIGH_PRIORITY_INTERVAL:
                    high_queue.append(cam_id)
            elif cam_state.priority == "low":
                if elapsed >= self.LOW_PRIORITY_INTERVAL:
                    low_queue.append(cam_id)
            else:  # idle
                if elapsed >= self.IDLE_SCAN_INTERVAL:
                    idle_queue.append(cam_id)

        # Build final queue respecting budget
        queue = []

        # Add all HIGH cameras (these are critical)
        queue.extend(high_queue[:self.MAX_CAMERAS_PER_CYCLE])

        # Fill remaining budget with LOW cameras
        remaining = self.MAX_CAMERAS_PER_CYCLE - len(queue)
        if remaining > 0:
            queue.extend(low_queue[:remaining])

        # Add one IDLE camera for scanning (round-robin)
        remaining = self.MAX_CAMERAS_PER_CYCLE - len(queue)
        if remaining > 0 and idle_queue:
            # Round-robin through idle cameras
            self._last_scan_index = self._last_scan_index % len(idle_queue)
            queue.append(idle_queue[self._last_scan_index])
            self._last_scan_index += 1

        return queue

    def update_priorities(self, camera_states: dict[str, CameraDetectionState],
                          frame_widths: dict[str, int]):
        """Update camera priorities based on detection results.

        Called after each detection cycle to recalculate priorities.
        """
        now = time.time()

        # First pass: determine which cameras have horses
        cameras_with_horses = set()
        for cam_id, state in camera_states.items():
            if state.horses_present:
                cameras_with_horses.add(cam_id)

        # Clear stale expected_horses (timeout protection)
        for cam_id, state in camera_states.items():
            if state.expected_horses:
                set_time = self._expected_set_time.get(cam_id, 0.0)
                if now - set_time > self.EXPECTED_TIMEOUT:
                    state.expected_horses.clear()
                    self._expected_set_time.pop(cam_id, None)

        # Second pass: set priorities
        for cam_id, state in camera_states.items():
            cam_idx = state.cam_index

            if state.horses_present:
                # Camera has active detections → HIGH
                state.priority = "high"
                state.last_detection_time = now

                # Handoff complete: if we expected horses and now see them, clear
                if state.expected_horses:
                    detected_expected = state.horses_present & state.expected_horses
                    state.expected_horses -= detected_expected
                    if not state.expected_horses:
                        self._expected_set_time.pop(cam_id, None)

                # Check for handoff: horses near right edge
                fw = frame_widths.get(cam_id, 1280)
                for color, sx in state.smooth_x.items():
                    # Only handoff colors that are currently present
                    if color not in state.horses_present:
                        continue
                    if sx / max(fw, 1) > self.HANDOFF_THRESHOLD:
                        # Horse approaching exit → promote next camera
                        next_cam_id = f"analytics-{cam_idx + 2}"
                        next_state = camera_states.get(next_cam_id)
                        if next_state:
                            next_state.expected_horses.add(color)
                            if next_cam_id not in self._expected_set_time:
                                self._expected_set_time[next_cam_id] = now
                            next_state.priority = "high"
                            log.debug(f"[Handoff] {color} from {cam_id} → {next_cam_id}")

            elif state.expected_horses:
                # Horses expected from adjacent camera → HIGH
                state.priority = "high"

            elif now - state.last_detection_time < self.GRACE_PERIOD:
                # Recently had horses, keep monitoring → LOW
                state.priority = "low"

            elif self._has_adjacent_horses(cam_idx, cameras_with_horses):
                # Adjacent camera has horses → LOW
                state.priority = "low"

            else:
                # No horses anywhere near → IDLE
                state.priority = "idle"
                state.expected_horses.clear()
                self._expected_set_time.pop(cam_id, None)

    def _has_adjacent_horses(self, cam_index: int,
                              cameras_with_horses: set) -> bool:
        """Check if any adjacent camera (±2) has horses."""
        for offset in [-2, -1, 1, 2]:
            adj_id = f"analytics-{cam_index + 1 + offset}"
            if adj_id in cameras_with_horses:
                return True
        return False

    def get_high_priority_cameras(self) -> list:
        """Get list of HIGH priority camera IDs."""
        return [cam_id for cam_id, s in self.camera_states.items()
                if s.priority == "high"]

    def get_priority_summary(self) -> dict:
        """Get count of cameras per priority level."""
        summary = {"high": 0, "low": 0, "idle": 0}
        for state in self.camera_states.values():
            summary[state.priority] = summary.get(state.priority, 0) + 1
        return summary
