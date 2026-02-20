"""
Cross-Camera Ranking Merger

Combines horse position data from analytics cameras into a single
unified ranking (0-2500m track).

Ranking rule:
    - Each analytics camera represents a 100m segment.
    - Ranking advances only after a segment is fully passed.
    - Practically: if a horse is stably observed on camera N,
      we mark camera N-1 as completed and update distance to N * 100m.
"""

import time
import logging
from typing import Optional

log = logging.getLogger("race_server")


class HorseTrackingInfo:
    """Tracks one horse across all cameras."""

    def __init__(self, color: str, horse_info: dict):
        self.color = color
        self.horse_info = horse_info

        # Output metrics
        self.absolute_distance: float = 0.0
        self.speed: float = 0.0
        self.confidence: float = 0.0
        self.last_cam_id: str = ""
        self.last_cam_index: int = 0
        self.local_progress: float = 0.0

        # Tracking state
        self.last_seen_time: float = 0.0
        self.is_tracked: bool = False

        # Segment completion state:
        # -1 means start line (0m completed)
        self.completed_camera_index: int = -1
        self.last_checkpoint_time: float = 0.0

        # Camera transition stability (requires repeated consecutive observations)
        self.last_observed_camera_index: Optional[int] = None
        self.observed_streak: int = 0


class RankingMerger:
    """Merges per-camera detection into unified rankings."""

    CAMERA_TRACK_M = 100.0
    TRACK_LENGTH = 2500.0
    # Keep horse visible in public ranking for 5s after last real detection.
    GRACE_PERIOD = 5.0

    # Transition stability controls
    CAMERA_OBS_MIN_VOTES = 2
    MAX_CAMERA_ADVANCE = 1
    LAST_CAMERA_EXIT_PROGRESS = 0.92
    BEST_CAM_TIME_EPS = 0.20

    def __init__(self, color_to_horse: dict, all_colors: list):
        self.color_to_horse = color_to_horse
        self.all_colors = all_colors

        self.horse_tracking: dict[str, HorseTrackingInfo] = {}
        for color, info in color_to_horse.items():
            self.horse_tracking[color] = HorseTrackingInfo(color, info)

        # Kept for compatibility with existing callsites
        self.known_frame_widths: dict[str, int] = {}

        self.race_start_time: float = 0.0
        self.last_order: list[str] = list(all_colors)

    def set_race_start_time(self, t: float):
        self.race_start_time = t

    def merge(self, camera_states: dict, frame_widths: dict) -> list:
        now = time.time()
        elapsed = now - self.race_start_time if self.race_start_time > 0 else 0.0
        max_cam_index = max((s.cam_index for s in camera_states.values()), default=0)

        # Keep current cache behavior (used by other debug code paths).
        for cam_id, fw in frame_widths.items():
            if fw > 0:
                self.known_frame_widths[cam_id] = fw

        # Step 1: find best camera observation per horse color
        for color in self.all_colors:
            tracking = self.horse_tracking.get(color)
            if tracking is None:
                continue

            best_cam = None
            best_confidence = 0.0
            best_speed = 0.0
            best_time = 0.0
            best_local_progress = 0.0

            for cam_state in camera_states.values():
                if color not in cam_state.smooth_x:
                    continue

                color_seen_time = cam_state.color_last_seen.get(color, 0.0)
                if now - color_seen_time > cam_state.STALE_TIMEOUT:
                    continue

                conf = float(cam_state.color_confidence.get(color, 0.5))
                should_pick = False
                if best_cam is None:
                    should_pick = True
                else:
                    is_newer = color_seen_time > (best_time + self.BEST_CAM_TIME_EPS)
                    is_time_close = abs(color_seen_time - best_time) <= self.BEST_CAM_TIME_EPS
                    prefer_forward_cam = is_time_close and cam_state.cam_index > best_cam.cam_index
                    prefer_conf = (
                        is_time_close
                        and cam_state.cam_index == best_cam.cam_index
                        and conf > best_confidence
                    )
                    should_pick = is_newer or prefer_forward_cam or prefer_conf

                if should_pick:
                    best_cam = cam_state
                    best_confidence = conf
                    best_speed = float(cam_state.speed.get(color, 0.0))
                    best_time = color_seen_time
                    fw = frame_widths.get(
                        cam_state.cam_id,
                        self.known_frame_widths.get(cam_state.cam_id, 1280),
                    )
                    best_local_progress = float(cam_state.smooth_x[color]) / max(float(fw), 1.0)

            if best_cam is not None:
                if tracking.last_observed_camera_index == best_cam.cam_index:
                    tracking.observed_streak += 1
                else:
                    tracking.last_observed_camera_index = best_cam.cam_index
                    tracking.observed_streak = 1

                stable_cam = None
                if tracking.observed_streak >= self.CAMERA_OBS_MIN_VOTES:
                    stable_cam = tracking.last_observed_camera_index

                # Only advance ranking after a full camera is passed.
                # Seeing camera N confirms completion of camera N-1.
                if stable_cam is not None:
                    completed_candidate = stable_cam - 1
                    if (
                        stable_cam == max_cam_index
                        and best_cam.cam_index == stable_cam
                        and best_local_progress >= self.LAST_CAMERA_EXIT_PROGRESS
                    ):
                        # Last camera has no "next camera", so allow direct completion
                        # when horse reaches the right edge zone.
                        completed_candidate = stable_cam
                    if completed_candidate > tracking.completed_camera_index:
                        max_allowed = tracking.completed_camera_index + self.MAX_CAMERA_ADVANCE
                        tracking.completed_camera_index = min(completed_candidate, max_allowed)
                        tracking.last_checkpoint_time = now

                # Distance is based ONLY on completed cameras — ranking changes
                # only when a horse passes to the next camera, not every frame.
                tracking.absolute_distance = min(
                    (tracking.completed_camera_index + 1) * self.CAMERA_TRACK_M,
                    self.TRACK_LENGTH,
                )
                tracking.speed = best_speed
                tracking.confidence = best_confidence
                tracking.last_cam_id = best_cam.cam_id
                tracking.last_cam_index = best_cam.cam_index
                # local_progress used only as tiebreaker for horses at the same distance
                tracking.local_progress = max(0.0, min(1.0, best_local_progress))
                # Use camera's real last-detected timestamp so visibility hold
                # is based on actual detection, not merge-loop clock.
                tracking.last_seen_time = best_time
                tracking.is_tracked = True

            elif now - tracking.last_seen_time < self.GRACE_PERIOD:
                tracking.is_tracked = True
                # Gradually decay speed (gentler than before to avoid sudden jumps)
                tracking.speed *= 0.95
                if abs(tracking.speed) < 0.05:
                    tracking.speed = 0.0

            else:
                tracking.is_tracked = False
                tracking.speed = 0.0

        # Step 2: stable sort for ranking
        tracked_horses = [
            (color, self.horse_tracking[color])
            for color in self.all_colors
            if color in self.color_to_horse and color in self.horse_tracking
        ]

        active = [(c, t) for c, t in tracked_horses if t.is_tracked]

        prev_order_idx = {color: i for i, color in enumerate(self.last_order)}

        active.sort(
            key=lambda x: (
                -x[1].absolute_distance,
                -x[1].local_progress,  # within same camera: further right = ahead
                x[1].last_checkpoint_time if x[1].last_checkpoint_time > 0 else float("inf"),
                prev_order_idx.get(x[0], len(self.all_colors)),
            )
        )

        # Public ranking should reflect currently detected/tracked colors only.
        ordered = active
        if ordered:
            ordered_colors = [color for color, _ in ordered]
            self.last_order = ordered_colors + [
                c for c in self.last_order if c not in ordered_colors
            ]

        # Step 3: output payload
        rankings = []
        leader_distance = ordered[0][1].absolute_distance if ordered else 0.0
        leader_speed = ordered[0][1].speed if ordered and ordered[0][1].speed > 0 else 10.0

        for pos, (color, tracking) in enumerate(ordered):
            horse_info = self.color_to_horse[color]

            if pos == 0:
                gap_seconds = 0.0
            else:
                distance_gap = leader_distance - tracking.absolute_distance
                gap_seconds = abs(distance_gap) / leader_speed

            rankings.append({
                "id": horse_info["id"],
                "number": int(horse_info["number"]),
                "name": horse_info["name"],
                "color": horse_info["color"],
                "jockeyName": horse_info["jockeyName"],
                "silkId": int(horse_info["silkId"]),
                "position": pos + 1,
                "distanceCovered": round(float(tracking.absolute_distance), 1),
                "currentLap": 1,
                "timeElapsed": round(float(elapsed), 1),
                "speed": round(float(tracking.speed), 2),
                "gapToLeader": round(float(gap_seconds), 2),
                "lastCameraId": tracking.last_cam_id or "analytics-1",
            })

        return rankings

    def reset(self):
        for tracking in self.horse_tracking.values():
            tracking.absolute_distance = 0.0
            tracking.speed = 0.0
            tracking.confidence = 0.0
            tracking.last_cam_id = ""
            tracking.last_cam_index = 0
            tracking.last_seen_time = 0.0
            tracking.is_tracked = False
            tracking.completed_camera_index = -1
            tracking.last_checkpoint_time = 0.0
            tracking.last_observed_camera_index = None
            tracking.observed_streak = 0
            tracking.local_progress = 0.0

        self.known_frame_widths.clear()
        self.race_start_time = 0.0
        self.last_order = list(self.all_colors)
