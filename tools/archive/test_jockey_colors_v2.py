"""
Jockey Color Tracking v2
ЦВЕТ = ГЛАВНЫЙ ИДЕНТИФИКАТОР

Логика:
1. Детектим всех людей
2. Для КАЖДОГО определяем цвет формы
3. ID присваивается по цвету (не по трекеру)
4. Трекер только для сглаживания bbox

Цвета:
1. Green (Зеленый)
2. Red (Красный)
3. Yellow (Желтый)
4. Blue (Синий)
5. Purple (Фиолетовый)
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Detection settings
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 1920

# 5 Jockey colors with WIDER HSV ranges
JOCKEY_COLORS = {
    1: {"name": "Green",  "bgr": (0, 200, 0),   "hue_min": 30, "hue_max": 95},   # Wider: includes lime, teal
    2: {"name": "Red",    "bgr": (0, 0, 220),   "hue_min": 0,  "hue_max": 12, "hue_min2": 165, "hue_max2": 180},
    3: {"name": "Yellow", "bgr": (0, 220, 220), "hue_min": 12, "hue_max": 30},   # Narrower to not overlap with green
    4: {"name": "Blue",   "bgr": (220, 0, 0),   "hue_min": 95, "hue_max": 135},  # Wider
    5: {"name": "Purple", "bgr": (180, 0, 180), "hue_min": 135, "hue_max": 165},
}


def get_uniform_color(crop_bgr: np.ndarray, debug=False):
    """
    Extract uniform color from jockey torso
    Returns: (jockey_id, confidence, actual_bgr, avg_hue)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0, 0, (128, 128, 128), 0

    h, w = crop_bgr.shape[:2]
    if h < 30 or w < 15:
        return 0, 0, (128, 128, 128), 0

    # Torso area (uniform visible here)
    y1 = int(h * 0.10)
    y2 = int(h * 0.55)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        return 0, 0, (128, 128, 128), 0

    # Convert to HSV
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Get average hue for debug
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])

    # Count pixels for each color
    color_scores = {}

    for jid, color_info in JOCKEY_COLORS.items():
        # Create mask for this color
        if "hue_min2" in color_info:
            # Red wraps around
            mask1 = cv2.inRange(hsv,
                (color_info["hue_min"], 20, 40),
                (color_info["hue_max"], 255, 255))
            mask2 = cv2.inRange(hsv,
                (color_info["hue_min2"], 20, 40),
                (color_info["hue_max2"], 255, 255))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv,
                (color_info["hue_min"], 20, 40),
                (color_info["hue_max"], 255, 255))

        # Count matching pixels
        pixel_count = cv2.countNonZero(mask)
        total_pixels = torso.shape[0] * torso.shape[1]
        ratio = pixel_count / total_pixels if total_pixels > 0 else 0

        color_scores[jid] = ratio

    # Find best matching color
    if not color_scores:
        return 0, 0, (128, 128, 128), avg_hue

    best_jid = max(color_scores, key=color_scores.get)
    best_score = color_scores[best_jid]

    # Need at least 12% of pixels matching (lowered threshold)
    if best_score < 0.12:
        if debug:
            print(f"  [DEBUG] Unclassified: Hue={avg_hue:.0f}, Sat={avg_sat:.0f}, best={best_jid} score={best_score:.2f}")
        return 0, 0, (128, 128, 128), avg_hue

    # Get average color of torso for display
    avg_bgr = np.mean(torso, axis=(0, 1)).astype(int)

    return best_jid, best_score, tuple(avg_bgr), avg_hue


class ColorBasedTracker:
    """
    Track jockeys by COLOR as primary ID
    No ID switching - color determines identity
    """

    def __init__(self):
        # Last known position for each jockey (for smoothing)
        self.last_bbox = {}  # jockey_id -> bbox
        self.last_seen = {}  # jockey_id -> frame_num
        self.confidence_history = defaultdict(list)  # jockey_id -> [scores]

    def update(self, frame, detections, frame_num):
        """
        Process detections and assign IDs by color
        Returns: list of (jockey_id, bbox, color_name, bgr, confidence, actual_bgr)
        """
        results = []
        seen_jockeys = {}  # jockey_id -> (bbox, score, actual_bgr)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            conf = det.confidence

            # Crop
            x1c, y1c = max(0, x1), max(0, y1)
            x2c = min(frame.shape[1], x2)
            y2c = min(frame.shape[0], y2)

            if x2c <= x1c or y2c <= y1c:
                continue

            crop = frame[y1c:y2c, x1c:x2c]

            # Get color (THIS IS THE ID!)
            jockey_id, color_score, actual_bgr, avg_hue = get_uniform_color(crop, debug=(frame_num % 100 == 1))

            if jockey_id == 0:
                continue

            # If same jockey detected multiple times, keep best score
            if jockey_id in seen_jockeys:
                if color_score > seen_jockeys[jockey_id][1]:
                    seen_jockeys[jockey_id] = (det.bbox, color_score, actual_bgr, conf)
            else:
                seen_jockeys[jockey_id] = (det.bbox, color_score, actual_bgr, conf)

        # Build results
        for jockey_id, (bbox, color_score, actual_bgr, conf) in seen_jockeys.items():
            color_info = JOCKEY_COLORS[jockey_id]

            # Smooth bbox with last known position
            if jockey_id in self.last_bbox and (frame_num - self.last_seen.get(jockey_id, 0)) < 10:
                old_bbox = self.last_bbox[jockey_id]
                # Exponential smoothing
                alpha = 0.7
                smooth_bbox = (
                    int(alpha * bbox[0] + (1-alpha) * old_bbox[0]),
                    int(alpha * bbox[1] + (1-alpha) * old_bbox[1]),
                    int(alpha * bbox[2] + (1-alpha) * old_bbox[2]),
                    int(alpha * bbox[3] + (1-alpha) * old_bbox[3]),
                )
                bbox = smooth_bbox

            self.last_bbox[jockey_id] = bbox
            self.last_seen[jockey_id] = frame_num

            results.append((
                jockey_id,
                bbox,
                color_info["name"],
                color_info["bgr"],
                conf,
                actual_bgr
            ))

        return results


def draw_jockeys(frame, jockeys, frame_num):
    """Draw jockeys with colors and legend"""

    detected_ids = set()

    for jid, bbox, color_name, bgr, conf, actual_bgr in jockeys:
        x1, y1, x2, y2 = bbox
        detected_ids.add(jid)

        # Thick colored box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 5)

        # Label with ID and color name
        label = f"J{jid} {color_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

        # Label background
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 15, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Actual detected color swatch (small square)
        actual_color = tuple(int(c) for c in actual_bgr)
        cv2.rectangle(frame, (x2 - 35, y1 + 5), (x2 - 5, y1 + 35), actual_color, -1)
        cv2.rectangle(frame, (x2 - 35, y1 + 5), (x2 - 5, y1 + 35), (255, 255, 255), 2)

    # === Info Panel ===
    panel_h = 250
    cv2.rectangle(frame, (5, 5), (280, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (280, panel_h), (100, 100, 100), 2)

    # Title
    cv2.putText(frame, f"Frame {frame_num}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Detected: {len(jockeys)}/5", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Legend
    y = 100
    for jid, color_info in JOCKEY_COLORS.items():
        # Color box
        cv2.rectangle(frame, (15, y), (45, y + 25), color_info["bgr"], -1)
        cv2.rectangle(frame, (15, y), (45, y + 25), (255, 255, 255), 1)

        # Status
        status = "DETECTED" if jid in detected_ids else "---"
        status_color = (0, 255, 0) if jid in detected_ids else (128, 128, 128)

        text = f"J{jid} {color_info['name']}"
        cv2.putText(frame, text, (55, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status, (180, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        y += 30

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Jockey Color Tracking v2")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-show", action="store_true", help="Don't show window")

    args = parser.parse_args()
    video_path = args.video

    print(f"\n{'='*60}")
    print(f"JOCKEY COLOR TRACKING v2")
    print(f"Color = Primary ID (no switching!)")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"{'='*60}\n")

    if not Path(video_path).exists():
        print(f"ERROR: Video not found")
        return

    # Load YOLO
    print("Loading YOLO...")
    model = YOLO("models/yolov8n.pt")

    # Tracker
    tracker = ColorBasedTracker()
    print("Ready!\n")

    # Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    scale = 0.5 if w > 1920 else 1.0
    dw, dh = int(w * scale), int(h * scale)

    writer = None
    if args.save:
        out_path = str(Path(video_path).stem) + "_colors_v2.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving: {out_path}")

    print("\nPress 'q' to quit, SPACE to pause\n")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect
        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Parse
        from api.vision.dtypes import Detection
        detections = []
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, _ = det
            detections.append(Detection(bbox=(int(x1), int(y1), int(x2), int(y2)), confidence=conf))

        # Track by color
        jockeys = tracker.update(frame, detections, frame_num)

        # Draw
        frame_drawn = draw_jockeys(frame, jockeys, frame_num)

        if writer:
            writer.write(frame_drawn)

        if not args.no_show:
            disp = cv2.resize(frame_drawn, (dw, dh)) if scale != 1.0 else frame_drawn
            cv2.imshow("Jockey Colors v2", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Paused at {frame_num}")
                cv2.waitKey(0)

        if frame_num % 50 == 0:
            jids = sorted([j[0] for j in jockeys])
            print(f"Frame {frame_num}/{total} | Jockeys: {jids}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone! Processed {frame_num} frames")


if __name__ == "__main__":
    main()
