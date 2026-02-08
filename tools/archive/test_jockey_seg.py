"""
Jockey Tracking with Segmentation v5

Архитектура:
1. Kalman + IoU + Center + K=5 + smooth bbox — база трекинга
2. YOLO-seg для маски жокея (только внутри bbox)
3. CLAHE нормализация освещения
4. Цвет из маски торса (верх 40-50%)
5. Цвет = жёсткий gating (нет совпадения = нет матча)
6. Авто-шаблон: первые 2 сек = эталон цвета
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# === SETTINGS ===
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1280

# Tracking
MAX_MISSING = 30
SMOOTH_ALPHA = 0.5
MAX_DISTANCE = 400  # pixels
MIN_IOU = 0.1

# Color
TEMPLATE_FRAMES = 50  # ~2 sec @ 25fps для создания шаблона
COLOR_MATCH_THRESH = 0.5  # Минимум корреляции гистограмм
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

# 5 цветов для отображения
DISPLAY_COLORS = {
    "green":  (0, 200, 0),
    "red":    (0, 0, 220),
    "yellow": (0, 220, 220),
    "blue":   (220, 0, 0),
    "purple": (180, 0, 180),
    "unknown": (128, 128, 128),
}


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """Нормализация освещения через CLAHE на V-канале"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_color_histogram(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Получить цветовую гистограмму (HSV)
    mask: binary mask, цвет считается только по маске
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Гистограмма по H и S (игнорируем V - яркость)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Сравнить две гистограммы (0-1, больше = лучше)"""
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(hist1.reshape(-1).astype(np.float32),
                           hist2.reshape(-1).astype(np.float32),
                           cv2.HISTCMP_CORREL)


def classify_color_name(hist: np.ndarray) -> str:
    """Определить название цвета по гистограмме"""
    if hist is None:
        return "unknown"

    # Восстановить 2D гистограмму
    hist_2d = hist.reshape(30, 32)

    # Найти доминантный hue
    hue_hist = np.sum(hist_2d, axis=1)
    dominant_hue = np.argmax(hue_hist) * 6  # 30 bins -> 0-180

    # Классифицировать по hue
    if dominant_hue < 10 or dominant_hue > 170:
        return "red"
    elif 10 <= dominant_hue < 35:
        return "yellow"
    elif 35 <= dominant_hue < 85:
        return "green"
    elif 85 <= dominant_hue < 130:
        return "blue"
    elif 130 <= dominant_hue < 170:
        return "purple"

    return "unknown"


@dataclass
class JockeyTrack:
    track_id: int
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Kalman-like state
    velocity: Tuple[float, float] = (0.0, 0.0)

    # Stability
    frames_seen: int = 0
    frames_missing: int = 0

    # Color template
    color_template: Optional[np.ndarray] = None
    color_samples: List[np.ndarray] = field(default_factory=list)
    color_name: str = "unknown"
    template_locked: bool = False


class SegmentationTracker:
    def __init__(self):
        # Load YOLO-seg model
        print("Loading YOLO-seg model...")
        self.seg_model = YOLO("yolov8n-seg.pt")
        print("YOLO-seg loaded!")

        self.tracks: Dict[int, JockeyTrack] = {}
        self.next_id = 1

    def _get_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _distance(self, bbox1, bbox2):
        c1, c2 = self._get_center(bbox1), self._get_center(bbox2)
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        return inter / (area1 + area2 - inter)

    def _extract_torso_with_mask(self, frame: np.ndarray, bbox: Tuple, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечь торс с маской сегментации
        Returns: (torso_image, torso_mask)
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        # Torso region (top 40-50%)
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)

        # Clamp to frame
        ty1 = max(0, ty1)
        ty2 = min(frame.shape[0], ty2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)

        if ty2 <= ty1 or x2 <= x1:
            return None, None

        torso_img = frame[ty1:ty2, x1:x2].copy()

        # Apply mask if available
        torso_mask = None
        if mask is not None:
            torso_mask = mask[ty1:ty2, x1:x2]
            # Ensure mask is uint8
            if torso_mask.dtype != np.uint8:
                torso_mask = (torso_mask * 255).astype(np.uint8)

        return torso_img, torso_mask

    def _get_segmentation_mask(self, frame: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Получить маску сегментации для области bbox
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop region
        crop = frame[y1:y2, x1:x2]

        # Run segmentation on crop
        results = self.seg_model.predict(
            crop,
            imgsz=320,
            conf=0.3,
            classes=[PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        if len(results) == 0 or results[0].masks is None:
            return None

        masks = results[0].masks.data.cpu().numpy()

        if len(masks) == 0:
            return None

        # Combine all person masks
        combined_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        for m in masks:
            # Resize mask to crop size
            m_resized = cv2.resize(m, (x2-x1, y2-y1))
            combined_mask = np.maximum(combined_mask, (m_resized > 0.5).astype(np.uint8) * 255)

        # Create full-frame mask
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = combined_mask

        return full_mask

    def _smooth_bbox(self, track: JockeyTrack, new_bbox: Tuple) -> Tuple:
        old_bbox = track.bbox
        if old_bbox == (0, 0, 0, 0):
            return new_bbox

        # Update velocity
        old_c = self._get_center(old_bbox)
        new_c = self._get_center(new_bbox)
        track.velocity = (
            0.7 * track.velocity[0] + 0.3 * (new_c[0] - old_c[0]),
            0.7 * track.velocity[1] + 0.3 * (new_c[1] - old_c[1])
        )

        # EMA smoothing
        alpha = SMOOTH_ALPHA
        return (
            int(alpha * new_bbox[0] + (1-alpha) * old_bbox[0]),
            int(alpha * new_bbox[1] + (1-alpha) * old_bbox[1]),
            int(alpha * new_bbox[2] + (1-alpha) * old_bbox[2]),
            int(alpha * new_bbox[3] + (1-alpha) * old_bbox[3]),
        )

    def _match_detections(self, detections: List, frame: np.ndarray) -> List[Tuple[int, Tuple, np.ndarray]]:
        """
        Match detections to existing tracks using IoU + Center + Color gating
        Returns: list of (track_id, bbox, mask)
        """
        # Increase missing count for all tracks
        for track in self.tracks.values():
            track.frames_missing += 1

        matched = []
        used_tracks = set()
        used_dets = set()

        # Build detection info (bbox, mask, histogram)
        det_info = []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            mask = self._get_segmentation_mask(frame, bbox)

            # CLAHE normalization
            frame_norm = apply_clahe(frame)

            # Extract torso with mask
            torso_img, torso_mask = self._extract_torso_with_mask(frame_norm, bbox, mask)
            hist = get_color_histogram(torso_img, torso_mask) if torso_img is not None else None

            det_info.append({
                "idx": i,
                "bbox": bbox,
                "mask": mask,
                "hist": hist,
                "conf": det["conf"]
            })

        # Match by combined score: IoU + distance + color
        for det in det_info:
            if det["idx"] in used_dets:
                continue

            best_track = None
            best_score = -1

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue
                if track.frames_missing > MAX_MISSING:
                    continue

                # Distance check
                dist = self._distance(track.bbox, det["bbox"])
                if dist > MAX_DISTANCE:
                    continue

                # IoU
                iou = self._iou(track.bbox, det["bbox"])

                # Color matching (GATING)
                color_score = 1.0
                if track.template_locked and track.color_template is not None and det["hist"] is not None:
                    color_score = compare_histograms(track.color_template, det["hist"])
                    # Hard gating: if color doesn't match well, skip
                    if color_score < COLOR_MATCH_THRESH:
                        continue

                # Combined score
                score = iou * 0.4 + (1 - dist/MAX_DISTANCE) * 0.3 + color_score * 0.3

                if score > best_score:
                    best_score = score
                    best_track = tid

            if best_track is not None and best_score > 0.3:
                matched.append((best_track, det["bbox"], det["mask"], det["hist"]))
                used_tracks.add(best_track)
                used_dets.add(det["idx"])
            else:
                # New track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = JockeyTrack(track_id=new_id, bbox=det["bbox"])
                matched.append((new_id, det["bbox"], det["mask"], det["hist"]))
                used_dets.add(det["idx"])

        return matched

    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker
        detections: list of {"bbox": (x1,y1,x2,y2), "conf": float}
        """
        matches = self._match_detections(detections, frame)

        results = []

        for track_id, bbox, mask, hist in matches:
            track = self.tracks[track_id]

            # Update track
            track.bbox = self._smooth_bbox(track, bbox)
            track.frames_seen += 1
            track.frames_missing = 0

            # Update color template
            if hist is not None:
                if not track.template_locked:
                    track.color_samples.append(hist)

                    # Lock template after TEMPLATE_FRAMES
                    if len(track.color_samples) >= TEMPLATE_FRAMES:
                        # Average all samples
                        track.color_template = np.mean(track.color_samples, axis=0)
                        track.color_name = classify_color_name(track.color_template)
                        track.template_locked = True
                        print(f"[TEMPLATE] Track {track_id}: {track.color_name} locked after {TEMPLATE_FRAMES} frames")

            results.append({
                "track_id": track_id,
                "bbox": track.bbox,
                "color_name": track.color_name,
                "color_bgr": DISPLAY_COLORS.get(track.color_name, (128, 128, 128)),
                "frames_seen": track.frames_seen,
                "template_locked": track.template_locked,
                "mask": mask
            })

        # Remove old tracks
        to_remove = [tid for tid, t in self.tracks.items() if t.frames_missing > MAX_MISSING * 2]
        for tid in to_remove:
            del self.tracks[tid]

        return results


def draw(frame: np.ndarray, jockeys: List[Dict], frame_num: int) -> np.ndarray:
    for j in jockeys:
        x1, y1, x2, y2 = j["bbox"]
        bgr = j["color_bgr"]
        locked = j["template_locked"]

        # Draw bbox
        thickness = 4 if locked else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)

        # Draw torso line
        h = y2 - y1
        ty2 = y1 + int(h * TORSO_BOTTOM)
        cv2.line(frame, (x1, ty2), (x2, ty2), (255, 255, 255), 1)

        # Label
        status = "OK" if locked else "..."
        label = f"T{j['track_id']} {j['color_name']} [{status}]"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        # Draw mask overlay if available
        if j.get("mask") is not None and locked:
            mask_colored = np.zeros_like(frame)
            mask_colored[j["mask"] > 0] = bgr
            frame = cv2.addWeighted(frame, 1, mask_colored, 0.3, 0)

    # Info
    locked_count = sum(1 for j in jockeys if j["template_locked"])
    cv2.putText(frame, f"Frame {frame_num} | Tracks: {len(jockeys)} | Locked: {locked_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("JOCKEY TRACKING WITH SEGMENTATION v5")
    print("Kalman + IoU + Color Gating + YOLO-seg + CLAHE")
    print("=" * 60)

    # Detection model
    print("Loading detection model...")
    det_model = YOLO("models/yolov8n.pt")

    # Tracker with segmentation
    tracker = SegmentationTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")
    print(f"Template lock after: {TEMPLATE_FRAMES} frames (~{TEMPLATE_FRAMES/fps:.1f}s)")
    print()

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_seg.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect persons
        results = det_model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        detections = []
        for d in results[0].boxes.data.cpu().numpy():
            detections.append({
                "bbox": (int(d[0]), int(d[1]), int(d[2]), int(d[3])),
                "conf": float(d[4])
            })

        # Track
        jockeys = tracker.update(frame, detections)

        # Draw
        frame = draw(frame, jockeys, frame_num)

        if writer:
            writer.write(frame)

        disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
        cv2.imshow("Segmentation Tracking", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if frame_num % 100 == 0:
            colors = [(j["track_id"], j["color_name"], "L" if j["template_locked"] else ".") for j in jockeys]
            print(f"Frame {frame_num}/{total} | {colors}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
