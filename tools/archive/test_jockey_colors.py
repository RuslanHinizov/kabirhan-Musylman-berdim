"""
Jockey Color Tracking
Трекинг жокеев по цвету формы

Цвета:
1. Green (Зеленый)
2. Red (Красный)
3. Yellow (Желтый)
4. Blue (Синий)
5. Purple (Фиолетовый)

Usage:
    python tools/test_jockey_colors.py data/videos/exp10_cam1.mp4
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from api.vision.bytetrack import ByteTracker
from api.vision.dtypes import Detection

# Detection settings
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 1920

# 5 Jockey colors (BGR format)
JOCKEY_COLORS = {
    1: {"name": "Green",  "bgr": (0, 255, 0),   "hsv_range": ((35, 50, 50), (85, 255, 255))},
    2: {"name": "Red",    "bgr": (0, 0, 255),   "hsv_range": ((0, 50, 50), (10, 255, 255))},  # + (170, 180)
    3: {"name": "Yellow", "bgr": (0, 255, 255), "hsv_range": ((20, 50, 50), (35, 255, 255))},
    4: {"name": "Blue",   "bgr": (255, 0, 0),   "hsv_range": ((100, 50, 50), (130, 255, 255))},
    5: {"name": "Purple", "bgr": (255, 0, 255), "hsv_range": ((130, 50, 50), (170, 255, 255))},
}


def extract_dominant_color(crop_bgr: np.ndarray) -> tuple:
    """
    Extract dominant color from jockey's torso area
    Returns: (hue, saturation, value, bgr_color)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    # Focus on torso (upper-middle area where uniform is visible)
    y1 = int(h * 0.15)
    y2 = int(h * 0.60)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return None

    # Convert to HSV
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Get average color (excluding very dark/light pixels)
    mask = (hsv[:,:,2] > 30) & (hsv[:,:,2] < 250) & (hsv[:,:,1] > 30)

    if np.sum(mask) < 50:
        # Not enough valid pixels, use all
        avg_h = np.mean(hsv[:,:,0])
        avg_s = np.mean(hsv[:,:,1])
        avg_v = np.mean(hsv[:,:,2])
    else:
        avg_h = np.mean(hsv[:,:,0][mask])
        avg_s = np.mean(hsv[:,:,1][mask])
        avg_v = np.mean(hsv[:,:,2][mask])

    # Convert back to BGR for display
    avg_hsv = np.uint8([[[int(avg_h), int(avg_s), int(avg_v)]]])
    avg_bgr = cv2.cvtColor(avg_hsv, cv2.COLOR_HSV2BGR)[0][0]

    return (avg_h, avg_s, avg_v, tuple(map(int, avg_bgr)))


def classify_jockey_color(hsv_color: tuple) -> int:
    """
    Classify jockey by uniform color
    Returns: jockey_id (1-5) or 0 if unknown
    """
    if hsv_color is None:
        return 0

    h, s, v, _ = hsv_color

    # Need some saturation and brightness to detect color
    if s < 40 or v < 40:
        return 0  # Too dark or desaturated

    # Red (wraps around 0/180)
    if (h < 10 or h > 170) and s > 50:
        return 2  # Red

    # Yellow (20-35)
    if 15 < h < 40 and s > 50:
        return 3  # Yellow

    # Green (35-85)
    if 35 < h < 90 and s > 40:
        return 1  # Green

    # Blue (100-130)
    if 90 < h < 135 and s > 40:
        return 4  # Blue

    # Purple (130-170)
    if 130 < h < 170 and s > 40:
        return 5  # Purple

    return 0  # Unknown


class JockeyColorTracker:
    """Track jockeys by uniform color"""

    def __init__(self):
        self.tracker = ByteTracker(
            track_thresh=0.3,
            track_buffer=60,
            match_thresh=0.7,
            low_thresh=0.1,
            confirm_frames=2
        )

        # track_id -> (jockey_id, color_history)
        self.track_colors = defaultdict(lambda: {"votes": defaultdict(int), "jockey_id": 0})

        # jockey_id -> best track_id (to avoid duplicates)
        self.jockey_tracks = {}

    def update(self, frame, detections):
        """
        Update tracker and classify colors
        Returns: list of (jockey_id, bbox, color_name, bgr_color, confidence)
        """
        # ByteTrack update
        tracks = self.tracker.update(detections)

        results = []
        seen_jockeys = set()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            tid = track.track_id

            # Extract crop
            x1c, y1c = max(0, x1), max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)

            if x2c <= x1c or y2c <= y1c:
                continue

            crop = frame[y1c:y2c, x1c:x2c]

            # Get color
            hsv_color = extract_dominant_color(crop)
            jockey_id = classify_jockey_color(hsv_color)

            if jockey_id > 0:
                # Vote for this color
                self.track_colors[tid]["votes"][jockey_id] += 1

                # Get most voted color for this track
                votes = self.track_colors[tid]["votes"]
                best_jid = max(votes, key=votes.get)
                self.track_colors[tid]["jockey_id"] = best_jid

                # Only one track per jockey
                if best_jid not in seen_jockeys:
                    color_info = JOCKEY_COLORS[best_jid]
                    results.append((
                        best_jid,
                        track.bbox,
                        color_info["name"],
                        color_info["bgr"],
                        track.confidence,
                        hsv_color[3] if hsv_color else (128, 128, 128)  # Actual detected color
                    ))
                    seen_jockeys.add(best_jid)

        return results


def draw_jockeys(frame, jockeys, frame_num):
    """Draw jockeys with their colors"""

    for jid, bbox, color_name, bgr, conf, actual_bgr in jockeys:
        x1, y1, x2, y2 = bbox

        # Draw thick box with jockey color
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)

        # Draw label background
        label = f"J{jid} {color_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 15, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Draw actual detected color swatch
        cv2.rectangle(frame, (x2 - 30, y1), (x2, y1 + 30), actual_bgr, -1)
        cv2.rectangle(frame, (x2 - 30, y1), (x2, y1 + 30), (255, 255, 255), 2)

    # Info panel
    detected = [j[0] for j in jockeys]
    missing = [i for i in range(1, 6) if i not in detected]

    info = f"Frame {frame_num} | Jockeys: {len(jockeys)}/5"
    cv2.rectangle(frame, (5, 5), (500, 55), (0, 0, 0), -1)
    cv2.putText(frame, info, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Legend
    y_offset = 70
    for jid, color_info in JOCKEY_COLORS.items():
        status = "OK" if jid in detected else "?"
        cv2.rectangle(frame, (10, y_offset), (40, y_offset + 25), color_info["bgr"], -1)
        text = f"J{jid} {color_info['name']} [{status}]"
        cv2.putText(frame, text, (50, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

    return frame


def test_jockey_colors(video_path: str, show: bool = True, save: bool = False):
    """Run jockey color tracking test"""

    print(f"\n{'='*60}")
    print(f"JOCKEY COLOR TRACKING")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Colors: Green, Red, Yellow, Blue, Purple")
    print(f"{'='*60}\n")

    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Load YOLO
    print("Loading YOLO model...")
    model = YOLO("models/yolov8n.pt")

    # Create tracker
    tracker = JockeyColorTracker()
    print("Ready!\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Display scale
    display_scale = 0.5 if width > 1920 else 1.0
    display_w, display_h = int(width * display_scale), int(height * display_scale)

    # Writer
    writer = None
    if save:
        output_path = str(Path(video_path).stem) + "_jockey_colors.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving to: {output_path}")

    print("\nProcessing... (Press 'q' to quit, SPACE to pause)\n")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect persons only
        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Parse detections
        detections = []
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            bbox = (int(x1), int(y1), int(x2), int(y2))
            detections.append(Detection(bbox=bbox, confidence=conf))

        # Track and classify
        jockeys = tracker.update(frame, detections)

        # Draw
        frame_drawn = draw_jockeys(frame, jockeys, frame_num)

        if writer:
            writer.write(frame_drawn)

        if show:
            if display_scale != 1.0:
                display_frame = cv2.resize(frame_drawn, (display_w, display_h))
            else:
                display_frame = frame_drawn

            cv2.imshow("Jockey Color Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Paused at frame {frame_num}")
                cv2.waitKey(0)

        if frame_num % 50 == 0:
            jids = sorted([j[0] for j in jockeys])
            print(f"Frame {frame_num}/{total_frames} | Jockeys: {jids}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"DONE - Processed {frame_num} frames")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Jockey Color Tracking")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-show", action="store_true", help="Don't show window")

    args = parser.parse_args()

    test_jockey_colors(
        video_path=args.video,
        show=not args.no_show,
        save=args.save
    )


if __name__ == "__main__":
    main()
