"""
Fast Jockey Tracking v7 - BoT-SORT + Color Uniqueness

Key improvements:
- COLOR UNIQUENESS: Once a color is locked to a track, no other track can claim it
- Early color voting: Start classifying from first samples, not just at lock
- Better duplicate handling: If two tracks compete for same color, keep the better one
- Improved HSV ranges for 5 jockey colors

Pipeline: DET → TRACK (BoT-SORT) → COLOR CLASSIFY → UNIQUENESS CHECK
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# === SETTINGS ===
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1280

# Color detection
MIN_SAMPLES_FOR_COLOR = 5      # Min samples before classifying
LOCK_SAMPLES = 20              # Samples to lock template
COLOR_HIST_BINS = 16

# Torso ROI
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

# 5 Jockey colors with HSV hue ranges
# Note: OpenCV uses H: 0-180, S: 0-255, V: 0-255
JOCKEY_COLORS = {
    "green":  {"bgr": (0, 200, 0),   "hue_range": (35, 85)},
    "red":    {"bgr": (0, 0, 220),   "hue_range": (0, 10, 170, 180)},  # wrap-around
    "yellow": {"bgr": (0, 220, 220), "hue_range": (15, 35)},
    "blue":   {"bgr": (220, 0, 0),   "hue_range": (85, 130)},
    "purple": {"bgr": (180, 0, 180), "hue_range": (130, 165)},
}


def classify_hue(avg_hue: float) -> str:
    """Classify color by average hue value"""
    # Red wraps around (0-10 and 170-180)
    if avg_hue < 10 or avg_hue > 170:
        return "red"
    elif 15 <= avg_hue < 35:
        return "yellow"
    elif 35 <= avg_hue < 85:
        return "green"
    elif 85 <= avg_hue < 130:
        return "blue"
    elif 130 <= avg_hue < 170:
        return "purple"
    return "unknown"


def extract_torso_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], float, float]:
    """
    Extract color from torso region
    Returns: (color_name, avg_hue, avg_saturation)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    # Torso region
    ty1 = y1 + int(h * TORSO_TOP)
    ty2 = y1 + int(h * TORSO_BOTTOM)
    tx1 = x1 + int(w * 0.1)
    tx2 = x2 - int(w * 0.1)

    # Clamp
    ty1 = max(0, ty1)
    ty2 = min(frame.shape[0], ty2)
    tx1 = max(0, tx1)
    tx2 = min(frame.shape[1], tx2)

    if ty2 <= ty1 or tx2 <= tx1:
        return None, 0, 0

    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size < 500:
        return None, 0, 0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])

    # Need minimum saturation
    if avg_sat < 25:
        return None, avg_hue, avg_sat

    color_name = classify_hue(avg_hue)
    return color_name, avg_hue, avg_sat


class ColorTracker:
    """Track with color voting and lock"""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.color_votes: Dict[str, int] = defaultdict(int)
        self.total_samples = 0
        self.locked_color: Optional[str] = None
        self.hue_samples: List[float] = []

    def add_sample(self, color_name: str, hue: float):
        if self.locked_color is not None:
            return  # Already locked

        if color_name and color_name != "unknown":
            self.color_votes[color_name] += 1
            self.hue_samples.append(hue)
        self.total_samples += 1

    def get_best_color(self) -> Tuple[Optional[str], float]:
        """Get best color and confidence"""
        if not self.color_votes:
            return None, 0.0

        best = max(self.color_votes, key=self.color_votes.get)
        total = sum(self.color_votes.values())
        confidence = self.color_votes[best] / total if total > 0 else 0
        return best, confidence

    def can_lock(self) -> bool:
        """Check if ready to lock"""
        if self.locked_color:
            return False
        if self.total_samples < MIN_SAMPLES_FOR_COLOR:
            return False

        best, conf = self.get_best_color()
        # Need at least 60% confidence and min samples
        return best is not None and conf >= 0.6 and self.color_votes[best] >= MIN_SAMPLES_FOR_COLOR

    def lock(self, color: str):
        self.locked_color = color

    @property
    def current_color(self) -> str:
        if self.locked_color:
            return self.locked_color
        best, _ = self.get_best_color()
        return best if best else "unknown"


class FastJockeyTrackerV7:
    def __init__(self):
        print("Loading YOLO with BoT-SORT tracking...")
        self.model = YOLO("models/yolov8n.pt")

        # Track ID -> ColorTracker
        self.trackers: Dict[int, ColorTracker] = {}

        # COLOR UNIQUENESS: Set of locked colors
        self.locked_colors: Set[str] = set()

        # Track ID -> last bbox (for smoothing)
        self.last_bbox: Dict[int, Tuple] = {}

        self.frame_num = 0

    def _smooth_bbox(self, track_id: int, bbox: Tuple) -> Tuple:
        if track_id not in self.last_bbox:
            self.last_bbox[track_id] = bbox
            return bbox

        old = self.last_bbox[track_id]
        alpha = 0.6
        smoothed = (
            int(alpha * bbox[0] + (1-alpha) * old[0]),
            int(alpha * bbox[1] + (1-alpha) * old[1]),
            int(alpha * bbox[2] + (1-alpha) * old[2]),
            int(alpha * bbox[3] + (1-alpha) * old[3]),
        )
        self.last_bbox[track_id] = smoothed
        return smoothed

    def _try_lock_color(self, tracker: ColorTracker) -> bool:
        """Try to lock a color for this tracker (with uniqueness check)"""
        if tracker.locked_color:
            return True

        if not tracker.can_lock():
            return False

        best_color, conf = tracker.get_best_color()
        if best_color is None:
            return False

        # COLOR UNIQUENESS CHECK
        if best_color in self.locked_colors:
            # This color is already taken by another track
            # Try second best color
            votes_copy = dict(tracker.color_votes)
            del votes_copy[best_color]

            if votes_copy:
                second_best = max(votes_copy, key=votes_copy.get)
                total = sum(votes_copy.values())
                second_conf = votes_copy[second_best] / total if total > 0 else 0

                if second_conf >= 0.5 and second_best not in self.locked_colors:
                    tracker.lock(second_best)
                    self.locked_colors.add(second_best)
                    print(f"  [LOCK] Track {tracker.track_id} -> {second_best} (2nd choice, {best_color} taken)")
                    return True

            # Cannot lock - best color taken, no good second choice
            return False

        # Lock the color
        tracker.lock(best_color)
        self.locked_colors.add(best_color)
        print(f"  [LOCK] Track {tracker.track_id} -> {best_color} (conf={conf:.2f})")
        return True

    def update(self, frame: np.ndarray) -> List[Dict]:
        self.frame_num += 1

        # BoT-SORT tracking
        results = self.model.track(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],  # person
            tracker="botsort.yaml",
            persist=True,
            device="cuda:0",
            half=True,
            verbose=False
        )

        output = []

        if results[0].boxes.id is None:
            return output

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        active_tracks = set()

        for bbox, tid, conf in zip(boxes, track_ids, confs):
            active_tracks.add(tid)

            # Smooth bbox
            bbox = self._smooth_bbox(tid, tuple(bbox))

            # Get or create tracker
            if tid not in self.trackers:
                self.trackers[tid] = ColorTracker(tid)
            tracker = self.trackers[tid]

            # Extract color from torso
            color_name, avg_hue, avg_sat = extract_torso_color(frame, bbox)

            # Add sample (if not locked)
            if color_name:
                tracker.add_sample(color_name, avg_hue)

            # Try to lock color
            self._try_lock_color(tracker)

            # Get display color
            display_color = tracker.current_color
            color_bgr = JOCKEY_COLORS.get(display_color, {}).get("bgr", (128, 128, 128))

            output.append({
                "track_id": tid,
                "bbox": bbox,
                "conf": conf,
                "color_name": display_color,
                "color_bgr": color_bgr,
                "locked": tracker.locked_color is not None,
                "samples": tracker.total_samples,
                "votes": dict(tracker.color_votes),
            })

        return output


def draw(frame: np.ndarray, tracks: List[Dict], frame_num: int, locked_colors: Set[str]) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        bgr = t["color_bgr"]
        locked = t["locked"]

        # Box
        thick = 4 if locked else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thick)

        # Torso line
        h = y2 - y1
        ty2 = y1 + int(h * TORSO_BOTTOM)
        cv2.line(frame, (x1, ty2), (x2, ty2), (255, 255, 255), 1)

        # Label
        status = "OK" if locked else f"[{t['samples']}]"
        label = f"T{t['track_id']} {t['color_name']} {status}"
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2)

        # Show votes if not locked
        if not locked and t["votes"]:
            votes_str = " ".join([f"{c[0].upper()}:{v}" for c, v in sorted(t["votes"].items())])
            cv2.putText(frame, votes_str, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Info panel
    locked_count = sum(1 for t in tracks if t["locked"])
    cv2.rectangle(frame, (5, 5), (300, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(tracks)} | Locked: {locked_count}/5", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show locked colors
    y = 80
    for color in ["green", "red", "yellow", "blue", "purple"]:
        is_locked = color in locked_colors
        bgr = JOCKEY_COLORS[color]["bgr"]
        status_color = (0, 255, 0) if is_locked else (100, 100, 100)

        cv2.rectangle(frame, (15, y), (35, y+15), bgr, -1)
        cv2.putText(frame, f"{color}: {'LOCKED' if is_locked else '---'}",
                    (45, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += 18

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FAST JOCKEY TRACKING v7")
    print("BoT-SORT + Color Voting + UNIQUENESS CONSTRAINT")
    print("=" * 60)

    tracker = FastJockeyTrackerV7()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")
    print(f"Min samples: {MIN_SAMPLES_FOR_COLOR}, Lock at: {LOCK_SAMPLES}")
    print()

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_v7.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Track
        tracks = tracker.update(frame)

        # Draw
        frame_draw = draw(frame.copy(), tracks, frame_num, tracker.locked_colors)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.imshow("Jockey Tracking v7", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if frame_num % 100 == 0:
            locked = [(t["track_id"], t["color_name"]) for t in tracks if t["locked"]]
            print(f"Frame {frame_num}/{total} | Locked: {locked} | Colors: {tracker.locked_colors}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Locked colors: {tracker.locked_colors}")
    print()
    for tid, tr in sorted(tracker.trackers.items()):
        if tr.locked_color:
            print(f"  Track {tid}: {tr.locked_color} (votes: {dict(tr.color_votes)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
