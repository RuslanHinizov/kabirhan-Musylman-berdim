"""
Stable Jockey Color Tracking v3

Улучшения:
1. ROI торса - только верхние 35% bbox (без ковра лошади)
2. Color Gating - цвет не меняется без веской причины
3. K-кадров подтверждения - нужно K подряд кадров чтобы сменить ID
4. Smoothing bbox через EMA
5. Метрики стабильности

Цвета: Green, Red, Yellow, Blue, Purple
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from api.vision.dtypes import Detection

# === SETTINGS ===
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 1920

# Стабильность
K_FRAMES_CONFIRM = 8      # Кадров для подтверждения смены цвета (увеличено)
FORCE_SWITCH_FRAMES = 15  # Принудительная смена если видим другой цвет 15 кадров
COLOR_GATE_THRESHOLD = 0.20  # Минимум разницы для смены цвета (снижено)
BBOX_SMOOTHING = 0.4      # EMA коэффициент для bbox
MAX_MISSING_FRAMES = 30   # Кадров без детекции до сброса трека (увеличено)
MAX_JUMP_DISTANCE = 250   # Максимум пикселей за кадр (увеличено для 4K)

# ROI торса (% от bbox)
TORSO_TOP = 0.05    # Начало (от верха)
TORSO_BOTTOM = 0.40  # Конец (от верха) - НЕ включает ковёр лошади

# 5 цветов жокеев (HSV диапазоны)
JOCKEY_COLORS = {
    1: {"name": "Green",  "bgr": (0, 200, 0),   "hue_range": (30, 95)},
    2: {"name": "Red",    "bgr": (0, 0, 220),   "hue_range": [(0, 12), (165, 180)]},  # Wrap-around
    3: {"name": "Yellow", "bgr": (0, 220, 220), "hue_range": (12, 30)},
    4: {"name": "Blue",   "bgr": (220, 0, 0),   "hue_range": (95, 135)},
    5: {"name": "Purple", "bgr": (180, 0, 180), "hue_range": (135, 165)},
}


@dataclass
class JockeyTrack:
    """Трек одного жокея"""
    track_id: int
    color_id: int = 0                    # Текущий присвоенный цвет (1-5)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Color gating
    color_votes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    pending_color: int = 0               # Кандидат на смену цвета
    pending_count: int = 0               # Сколько кадров подряд видим pending_color

    # Стабильность
    frames_seen: int = 0
    frames_missing: int = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Velocity (для предсказания позиции)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) пикселей/кадр

    # Метрики
    color_changes: int = 0               # Сколько раз менялся цвет
    avg_color_score: float = 0.0
    jump_count: int = 0                  # Сколько раз пытался прыгнуть


def extract_torso_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Извлечь ROI торса (верхняя часть bbox, без ковра лошади)
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    if h < 40 or w < 20:
        return None

    # Только торс (верхние 35%)
    torso_y1 = y1 + int(h * TORSO_TOP)
    torso_y2 = y1 + int(h * TORSO_BOTTOM)
    torso_x1 = x1 + int(w * 0.15)
    torso_x2 = x2 - int(w * 0.15)

    # Границы кадра
    torso_y1 = max(0, torso_y1)
    torso_y2 = min(frame.shape[0], torso_y2)
    torso_x1 = max(0, torso_x1)
    torso_x2 = min(frame.shape[1], torso_x2)

    if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
        return None

    return frame[torso_y1:torso_y2, torso_x1:torso_x2]


def detect_color(torso_bgr: np.ndarray, debug=False) -> Tuple[int, float, Dict[int, float]]:
    """
    Определить цвет торса
    Returns: (color_id, confidence, all_scores)
    """
    if torso_bgr is None or torso_bgr.size == 0:
        return 0, 0.0, {}

    hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if total_pixels < 100:
        return 0, 0.0, {}

    # Debug: средний цвет
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])

    scores = {}

    for cid, color_info in JOCKEY_COLORS.items():
        hue_range = color_info["hue_range"]

        if isinstance(hue_range, list):
            # Red wrap-around
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for hr in hue_range:
                m = cv2.inRange(hsv, (hr[0], 25, 50), (hr[1], 255, 255))
                mask = cv2.bitwise_or(mask, m)
        else:
            mask = cv2.inRange(hsv, (hue_range[0], 25, 50), (hue_range[1], 255, 255))

        pixel_count = cv2.countNonZero(mask)
        scores[cid] = pixel_count / total_pixels

    if not scores:
        return 0, 0.0, {}

    best_cid = max(scores, key=scores.get)
    best_score = scores[best_cid]

    if debug:
        print(f"  [DEBUG] Hue={avg_hue:.0f} Sat={avg_sat:.0f} Val={avg_val:.0f} -> {JOCKEY_COLORS.get(best_cid, {}).get('name', '?')} ({best_score:.2f})")
        print(f"          Scores: G={scores.get(1,0):.2f} R={scores.get(2,0):.2f} Y={scores.get(3,0):.2f} B={scores.get(4,0):.2f} P={scores.get(5,0):.2f}")

    # Минимум 10% пикселей должны совпадать
    if best_score < 0.10:
        return 0, 0.0, scores

    return best_cid, best_score, scores


class StableColorTracker:
    """
    Стабильный трекинг жокеев по цвету

    - Color gating: цвет не меняется без K кадров подтверждения
    - ROI торса: игнорируем ковёр лошади
    - Smooth bbox: плавное движение
    """

    def __init__(self):
        self.tracks: Dict[int, JockeyTrack] = {}  # detection_id -> JockeyTrack
        self.color_to_track: Dict[int, int] = {}  # color_id -> track_id (обратный индекс)
        self.next_track_id = 1
        self.frame_num = 0

        # Метрики
        self.total_detections = 0
        self.successful_classifications = 0
        self.color_switches = 0
        self.gated_switches = 0  # Заблокированные смены цвета

    def _get_center(self, bbox: Tuple) -> Tuple[float, float]:
        """Получить центр bbox"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Расстояние между центрами bbox"""
        c1 = self._get_center(bbox1)
        c2 = self._get_center(bbox2)
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

    def _smooth_bbox(self, track: JockeyTrack, new_bbox: Tuple) -> Tuple[int, int, int, int]:
        """EMA сглаживание bbox - мягкое ограничение прыжков"""
        old_bbox = track.bbox

        if old_bbox == (0, 0, 0, 0):
            return new_bbox

        # Расстояние прыжка
        dist = self._distance(old_bbox, new_bbox)

        # Адаптивный alpha: чем дальше прыжок, тем меньше доверяем новому bbox
        if dist > MAX_JUMP_DISTANCE:
            # Далёкий прыжок - очень медленно двигаемся к нему
            alpha = 0.05  # Почти не двигаемся
            track.jump_count += 1
        elif dist > MAX_JUMP_DISTANCE * 0.5:
            # Средний прыжок - медленнее
            alpha = 0.15
        else:
            # Нормальное движение
            alpha = BBOX_SMOOTHING

        # Обновить velocity (только при нормальном движении)
        if dist < MAX_JUMP_DISTANCE:
            old_center = self._get_center(old_bbox)
            new_center = self._get_center(new_bbox)
            new_vx = new_center[0] - old_center[0]
            new_vy = new_center[1] - old_center[1]
            track.velocity = (
                0.8 * track.velocity[0] + 0.2 * new_vx,
                0.8 * track.velocity[1] + 0.2 * new_vy
            )

        # EMA сглаживание с адаптивным alpha
        return (
            int(alpha * new_bbox[0] + (1 - alpha) * old_bbox[0]),
            int(alpha * new_bbox[1] + (1 - alpha) * old_bbox[1]),
            int(alpha * new_bbox[2] + (1 - alpha) * old_bbox[2]),
            int(alpha * new_bbox[3] + (1 - alpha) * old_bbox[3]),
        )

    def _find_best_track(self, bbox: Tuple, color_id: int) -> Optional[int]:
        """Найти лучший существующий трек для детекции"""

        # 1. Сначала ищем по цвету, НО проверяем дистанцию
        if color_id > 0 and color_id in self.color_to_track:
            track_id = self.color_to_track[color_id]
            track = self.tracks.get(track_id)
            if track and track.bbox != (0, 0, 0, 0):
                dist = self._distance(track.bbox, bbox)
                # Для 4K - 600px это примерно 15% ширины кадра
                if dist < 600:
                    return track_id
                # Иначе - слишком далеко, ищем по IoU

        # 2. Ищем по близости bbox (IoU + дистанция)
        best_track_id = None
        best_score = 0

        for tid, track in self.tracks.items():
            if track.frames_missing > MAX_MISSING_FRAMES:
                continue
            if track.bbox == (0, 0, 0, 0):
                continue

            # Комбинированный скор: IoU + близость
            dist = self._distance(track.bbox, bbox)
            if dist > 800:  # Для 4K увеличено
                continue

            # IoU
            bx1, by1, bx2, by2 = bbox
            tx1, ty1, tx2, ty2 = track.bbox

            ix1 = max(bx1, tx1)
            iy1 = max(by1, ty1)
            ix2 = min(bx2, tx2)
            iy2 = min(by2, ty2)

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = (bx2 - bx1) * (by2 - by1)
                area2 = (tx2 - tx1) * (ty2 - ty1)
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0
            else:
                iou = 0

            # Бонус если цвет совпадает
            color_bonus = 0.4 if track.color_id == color_id else 0

            score = iou + color_bonus + (1 - dist/800) * 0.3

            if score > best_score:
                best_score = score
                best_track_id = tid

        return best_track_id if best_score > 0.15 else None

    def _try_assign_color(self, track: JockeyTrack, new_color: int, score: float) -> bool:
        """
        Попробовать присвоить цвет треку (с gating и K-подтверждением)
        Returns: True если цвет изменился
        """
        if new_color == 0:
            # Нет цвета - сбрасываем pending
            track.pending_count = 0
            return False

        # Если трек ещё без цвета - НЕ присваиваем сразу, ждём подтверждения
        if track.color_id == 0:
            if new_color == track.pending_color:
                track.pending_count += 1
            else:
                track.pending_color = new_color
                track.pending_count = 1

            # Нужно 3 кадра одинакового цвета для первого присвоения
            if track.pending_count >= 3:
                track.color_id = new_color
                track.color_votes[new_color] += 3
                track.pending_color = 0
                track.pending_count = 0
                self.color_to_track[new_color] = track.track_id
                print(f"[COLOR INIT] Track {track.track_id}: {JOCKEY_COLORS[new_color]['name']} (confirmed)")
                return True
            return False

        # Если тот же цвет - подтверждаем и сбрасываем pending
        if new_color == track.color_id:
            track.color_votes[new_color] += 1
            track.pending_color = 0
            track.pending_count = 0
            return False

        # Другой цвет - накапливаем pending
        if new_color == track.pending_color:
            track.pending_count += 1
        else:
            track.pending_color = new_color
            track.pending_count = 1

        # Проверяем условия смены цвета:
        # 1. FORCE_SWITCH_FRAMES - принудительная смена если долго видим другой цвет
        # 2. K_FRAMES_CONFIRM + color gate - обычная смена

        force_switch = track.pending_count >= FORCE_SWITCH_FRAMES
        normal_switch = (track.pending_count >= K_FRAMES_CONFIRM and
                        score >= track.avg_color_score + COLOR_GATE_THRESHOLD)

        if force_switch or normal_switch:
            old_color = track.color_id
            reason = "FORCE" if force_switch else "normal"

            # Убираем старый цвет из индекса
            if old_color in self.color_to_track and self.color_to_track[old_color] == track.track_id:
                del self.color_to_track[old_color]

            # Присваиваем новый
            track.color_id = new_color
            track.color_votes[new_color] += track.pending_count
            track.pending_color = 0
            track.pending_count = 0
            track.color_changes += 1
            track.avg_color_score = score  # Сбросить avg score

            self.color_to_track[new_color] = track.track_id
            self.color_switches += 1

            print(f"[COLOR {reason}] Track {track.track_id}: {JOCKEY_COLORS[old_color]['name']} -> {JOCKEY_COLORS[new_color]['name']} ({track.pending_count} frames)")
            return True

        # Не прошли условия - блокируем
        if track.pending_count == 1:
            self.gated_switches += 1

        return False

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Tuple]:
        """
        Обновить треки
        Returns: list of (jockey_id, bbox, color_name, bgr, confidence, metrics)
        """
        self.frame_num += 1
        self.total_detections += len(detections)

        # Увеличить frames_missing для всех треков
        for track in self.tracks.values():
            track.frames_missing += 1

        results = []
        seen_colors = set()

        for det in detections:
            bbox = det.bbox

            # 1. Извлечь ROI торса
            torso = extract_torso_roi(frame, bbox)
            if torso is None:
                continue

            # 2. Определить цвет
            color_id, color_score, all_scores = detect_color(torso)

            if color_id > 0:
                self.successful_classifications += 1

            # 3. Найти или создать трек
            track_id = self._find_best_track(bbox, color_id)

            if track_id is None:
                # Новый трек
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = JockeyTrack(track_id=track_id)

            track = self.tracks[track_id]

            # 4. Обновить трек
            track.bbox = self._smooth_bbox(track, bbox)
            track.frames_seen += 1
            track.frames_missing = 0
            track.confidence_history.append(det.confidence)
            track.avg_color_score = 0.8 * track.avg_color_score + 0.2 * color_score

            # 5. Попробовать присвоить цвет (с gating)
            self._try_assign_color(track, color_id, color_score)

            # 6. Собрать результат (только если есть цвет)
            if track.color_id > 0 and track.color_id not in seen_colors:
                seen_colors.add(track.color_id)

                color_info = JOCKEY_COLORS[track.color_id]
                metrics = {
                    "track_id": track.track_id,
                    "frames_seen": track.frames_seen,
                    "color_changes": track.color_changes,
                    "color_score": color_score,
                    "pending": f"{track.pending_color}:{track.pending_count}" if track.pending_count > 0 else "-"
                }

                results.append((
                    track.color_id,
                    track.bbox,
                    color_info["name"],
                    color_info["bgr"],
                    det.confidence,
                    metrics
                ))

        # Удалить старые треки
        to_remove = [tid for tid, t in self.tracks.items() if t.frames_missing > MAX_MISSING_FRAMES * 2]
        for tid in to_remove:
            track = self.tracks[tid]
            if track.color_id in self.color_to_track and self.color_to_track[track.color_id] == tid:
                del self.color_to_track[track.color_id]
            del self.tracks[tid]

        return results

    def get_metrics(self) -> Dict:
        """Получить метрики стабильности"""
        total_jumps = sum(t.jump_count for t in self.tracks.values())
        return {
            "total_detections": self.total_detections,
            "successful_classifications": self.successful_classifications,
            "classification_rate": self.successful_classifications / max(1, self.total_detections),
            "color_switches": self.color_switches,
            "gated_switches": self.gated_switches,
            "blocked_jumps": total_jumps,
            "active_tracks": len(self.tracks),
            "assigned_colors": list(self.color_to_track.keys()),
        }


def draw_frame(frame: np.ndarray, jockeys: List, tracker: StableColorTracker, frame_num: int) -> np.ndarray:
    """Отрисовать результат"""

    detected_ids = set()

    for jid, bbox, color_name, bgr, conf, metrics in jockeys:
        x1, y1, x2, y2 = bbox
        detected_ids.add(jid)

        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)

        # Торс ROI (показать область)
        h = y2 - y1
        torso_y1 = y1 + int(h * TORSO_TOP)
        torso_y2 = y1 + int(h * TORSO_BOTTOM)
        cv2.rectangle(frame, (x1, torso_y1), (x2, torso_y2), (255, 255, 255), 1)

        # Label
        label = f"J{jid} {color_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 15, y1), bgr, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Metrics под bbox
        info = f"T{metrics['track_id']} f:{metrics['frames_seen']} c:{metrics['color_changes']}"
        cv2.putText(frame, info, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    # === Panel ===
    panel_w = 320
    cv2.rectangle(frame, (5, 5), (panel_w, 280), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (panel_w, 280), (100, 100, 100), 2)

    cv2.putText(frame, f"Frame {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Detected: {len(jockeys)}/5", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Metrics
    m = tracker.get_metrics()
    cv2.putText(frame, f"Switches: {m['color_switches']} | Gated: {m['gated_switches']}",
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Legend
    y = 110
    for jid, color_info in JOCKEY_COLORS.items():
        cv2.rectangle(frame, (15, y), (45, y + 25), color_info["bgr"], -1)
        status = "OK" if jid in detected_ids else "--"
        status_color = (0, 255, 0) if jid in detected_ids else (100, 100, 100)
        cv2.putText(frame, f"J{jid} {color_info['name']}", (55, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status, (200, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y += 30

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video path")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("STABLE JOCKEY COLOR TRACKING v3")
    print(f"ROI Torso: {TORSO_TOP*100:.0f}%-{TORSO_BOTTOM*100:.0f}%")
    print(f"K-frames confirm: {K_FRAMES_CONFIRM}")
    print(f"Color gate threshold: {COLOR_GATE_THRESHOLD}")
    print(f"{'='*60}\n")

    if not Path(args.video).exists():
        print("Video not found!")
        return

    print("Loading YOLO...")
    model = YOLO("models/yolov8n.pt")
    tracker = StableColorTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames\n")

    scale = 0.5 if w > 1920 else 1.0
    dw, dh = int(w * scale), int(h * scale)

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_stable.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving: {out}")

    print("Press 'q' quit, SPACE pause, 'm' metrics\n")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect
        results = model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD,
                                classes=[PERSON_CLASS_ID], device="cuda:0", half=True, verbose=False)

        detections = [Detection(bbox=(int(d[0]), int(d[1]), int(d[2]), int(d[3])), confidence=d[4])
                      for d in results[0].boxes.data.cpu().numpy()]

        # Track
        jockeys = tracker.update(frame, detections)

        # Draw
        frame_drawn = draw_frame(frame, jockeys, tracker, frame_num)

        if writer:
            writer.write(frame_drawn)

        if not args.no_show:
            disp = cv2.resize(frame_drawn, (dw, dh)) if scale != 1.0 else frame_drawn
            cv2.imshow("Stable Tracking", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
            elif key == ord('m'):
                print(f"\n=== METRICS (frame {frame_num}) ===")
                for k, v in tracker.get_metrics().items():
                    print(f"  {k}: {v}")
                print()

        if frame_num % 100 == 0:
            jids = sorted([j[0] for j in jockeys])
            m = tracker.get_metrics()
            print(f"Frame {frame_num}/{total} | Jockeys: {jids} | Switches: {m['color_switches']} | Gated: {m['gated_switches']} | Jumps blocked: {m['blocked_jumps']}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final metrics
    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}")
    for k, v in tracker.get_metrics().items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
