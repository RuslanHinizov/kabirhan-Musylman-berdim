"""
Fast Jockey Tracking v6 - BoT-SORT + Color Gating

Pipeline: DET → TRACK (BoT-SORT) → SEG (раз в N кадров) → COLOR TEMPLATE

Оптимизации:
- YOLO встроенный BoT-SORT трекинг
- Сегментация только каждые 10 кадров (для обновления цвета)
- Цветовой шаблон кешируется
- Color gating через post-filter
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# === SETTINGS ===
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 1280

# Tracking
SEG_EVERY_N_FRAMES = 15  # Сегментация каждые N кадров
TEMPLATE_FRAMES = 30     # Кадров для фиксации шаблона
COLOR_HIST_BINS = 16     # Меньше bins = быстрее

# Torso ROI
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

# Display colors
COLORS_BGR = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 220),
    "yellow": (0, 220, 220),
    "blue":   (220, 0, 0),
    "purple": (180, 0, 180),
    "orange": (0, 165, 255),
    "unknown": (128, 128, 128),
}


class ColorTemplate:
    """Шаблон цвета жокея"""
    def __init__(self):
        self.samples: List[np.ndarray] = []
        self.template: Optional[np.ndarray] = None
        self.color_name: str = "unknown"
        self.locked: bool = False

    def add_sample(self, hist: np.ndarray):
        if self.locked or hist is None:
            return

        self.samples.append(hist)

        if len(self.samples) >= TEMPLATE_FRAMES:
            self.template = np.mean(self.samples, axis=0).astype(np.float32)
            self.color_name = self._classify()
            self.locked = True

    def _classify(self) -> str:
        if self.template is None:
            return "unknown"

        # Find dominant hue from histogram
        # Histogram is [H bins, S bins] flattened
        h_bins = COLOR_HIST_BINS
        hist_2d = self.template.reshape(h_bins, h_bins)
        hue_hist = np.sum(hist_2d, axis=1)
        dominant_idx = np.argmax(hue_hist)
        dominant_hue = dominant_idx * (180 // h_bins)

        if dominant_hue < 15 or dominant_hue > 165:
            return "red"
        elif 15 <= dominant_hue < 35:
            return "yellow"
        elif 35 <= dominant_hue < 45:
            return "orange"
        elif 45 <= dominant_hue < 85:
            return "green"
        elif 85 <= dominant_hue < 130:
            return "blue"
        elif 130 <= dominant_hue < 165:
            return "purple"

        return "unknown"

    def match(self, hist: np.ndarray) -> float:
        if self.template is None or hist is None:
            return 0.0
        return cv2.compareHist(self.template, hist.astype(np.float32), cv2.HISTCMP_CORREL)


def extract_color_hist(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Быстрое извлечение цветовой гистограммы"""
    if image is None or image.size < 500:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Только H и S каналы
    hist = cv2.calcHist(
        [hsv], [0, 1], mask,
        [COLOR_HIST_BINS, COLOR_HIST_BINS],
        [0, 180, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()


def get_torso_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple]:
    """Извлечь область торса"""
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1

    ty1 = y1 + int(h * TORSO_TOP)
    ty2 = y1 + int(h * TORSO_BOTTOM)

    # Clamp
    ty1 = max(0, ty1)
    ty2 = min(frame.shape[0], ty2)
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)

    if ty2 <= ty1 or x2 <= x1:
        return None, None

    return frame[ty1:ty2, x1:x2], (x1, ty1, x2, ty2)


class FastJockeyTracker:
    def __init__(self):
        print("Loading YOLO with BoT-SORT tracking...")
        self.model = YOLO("models/yolov8n.pt")

        # Optional: segmentation model (loaded lazily)
        self.seg_model = None

        # Track ID -> ColorTemplate
        self.color_templates: Dict[int, ColorTemplate] = defaultdict(ColorTemplate)

        # Track ID -> last bbox (for smoothing)
        self.last_bbox: Dict[int, Tuple] = {}

        self.frame_num = 0

    def _load_seg_model(self):
        if self.seg_model is None:
            print("Loading YOLO-seg (one time)...")
            self.seg_model = YOLO("yolov8n-seg.pt")

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

    def _get_seg_mask(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """Получить маску сегментации для bbox"""
        self._load_seg_model()

        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        results = self.seg_model.predict(
            crop, imgsz=256, conf=0.3, classes=[0],
            device="cuda:0", half=True, verbose=False
        )

        if results[0].masks is None or len(results[0].masks) == 0:
            return None

        # Get first mask
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        return (mask > 0.5).astype(np.uint8) * 255

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

        do_seg = (self.frame_num % SEG_EVERY_N_FRAMES == 0)

        for bbox, tid, conf in zip(boxes, track_ids, confs):
            # Smooth bbox
            bbox = self._smooth_bbox(tid, tuple(bbox))

            # Get torso region
            torso, torso_bbox = get_torso_region(frame, bbox)
            if torso is None:
                continue

            # Get color histogram
            mask = None
            if do_seg:
                mask = self._get_seg_mask(frame, bbox)
                if mask is not None:
                    # Crop mask to torso region
                    x1, y1, x2, y2 = bbox
                    tx1, ty1, tx2, ty2 = torso_bbox
                    mask = mask[ty1-y1:ty2-y1, tx1-x1:tx2-x1] if mask.shape[0] > 0 else None

            hist = extract_color_hist(torso, mask)

            # Update color template
            template = self.color_templates[tid]
            template.add_sample(hist)

            output.append({
                "track_id": tid,
                "bbox": bbox,
                "conf": conf,
                "color_name": template.color_name,
                "color_bgr": COLORS_BGR.get(template.color_name, (128, 128, 128)),
                "locked": template.locked,
                "samples": len(template.samples),
            })

        return output


def draw(frame: np.ndarray, tracks: List[Dict], frame_num: int) -> np.ndarray:
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
        status = f"[{t['samples']}]" if not locked else "OK"
        label = f"T{t['track_id']} {t['color_name']} {status}"
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2)

    # Info
    locked = sum(1 for t in tracks if t["locked"])
    cv2.putText(frame, f"Frame {frame_num} | Tracks: {len(tracks)} | Locked: {locked}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FAST JOCKEY TRACKING v6")
    print("BoT-SORT + Color Template + Lazy Segmentation")
    print("=" * 60)

    tracker = FastJockeyTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")
    print(f"Seg every: {SEG_EVERY_N_FRAMES} frames")
    print(f"Template lock: {TEMPLATE_FRAMES} samples")
    print()

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_botsort.mp4"
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
        frame_draw = draw(frame.copy(), tracks, frame_num)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.imshow("BoT-SORT Tracking", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if frame_num % 100 == 0:
            colors = [(t["track_id"], t["color_name"]) for t in tracks if t["locked"]]
            print(f"Frame {frame_num}/{total} | Locked: {colors}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL COLOR TEMPLATES:")
    for tid, tmpl in sorted(tracker.color_templates.items()):
        if tmpl.locked:
            print(f"  Track {tid}: {tmpl.color_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
