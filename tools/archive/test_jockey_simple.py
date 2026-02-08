"""
Simple Jockey Tracking v4
ПРОСТОЙ ПОДХОД:
1. ByteTrack для позиционного трекинга
2. Голосование за цвет (накопление)
3. Без сложной логики - просто и стабильно
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from api.vision.bytetrack import ByteTracker
from api.vision.dtypes import Detection

# Settings
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1920

# 5 цветов (HSV)
COLORS = {
    1: {"name": "Green",  "bgr": (0, 200, 0),   "hue": (35, 90)},
    2: {"name": "Red",    "bgr": (0, 0, 220),   "hue": (0, 10, 170, 180)},  # wrap
    3: {"name": "Yellow", "bgr": (0, 220, 220), "hue": (15, 35)},
    4: {"name": "Blue",   "bgr": (220, 0, 0),   "hue": (95, 130)},
    5: {"name": "Purple", "bgr": (180, 0, 180), "hue": (130, 160)},
}


def get_torso_color(frame, bbox):
    """Получить цвет торса (верхние 40%)"""
    x1, y1, x2, y2 = bbox
    h = y2 - y1

    # Торс: верхние 40%
    ty1 = y1 + int(h * 0.05)
    ty2 = y1 + int(h * 0.40)
    tx1 = x1 + int((x2-x1) * 0.15)
    tx2 = x2 - int((x2-x1) * 0.15)

    ty1, ty2 = max(0, ty1), min(frame.shape[0], ty2)
    tx1, tx2 = max(0, tx1), min(frame.shape[1], tx2)

    if ty2 <= ty1 or tx2 <= tx1:
        return 0

    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size < 500:
        return 0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])

    if avg_sat < 30:  # Слишком бледный
        return 0

    # Определить цвет по hue
    for cid, info in COLORS.items():
        hue = info["hue"]
        if len(hue) == 4:  # Red wrap
            if avg_hue <= hue[1] or avg_hue >= hue[2]:
                return cid
        else:
            if hue[0] <= avg_hue <= hue[1]:
                return cid

    return 0


class SimpleTracker:
    def __init__(self):
        self.tracker = ByteTracker(
            track_thresh=0.4,
            track_buffer=50,
            match_thresh=0.7
        )
        # track_id -> {color_id: vote_count}
        self.color_votes = defaultdict(lambda: defaultdict(int))
        # track_id -> smoothed bbox
        self.smooth_bbox = {}

    def update(self, frame, detections):
        tracks = self.tracker.update(detections)

        results = []

        for track in tracks:
            tid = track.track_id
            bbox = track.bbox

            # Smooth bbox
            if tid in self.smooth_bbox:
                old = self.smooth_bbox[tid]
                bbox = (
                    int(0.6 * bbox[0] + 0.4 * old[0]),
                    int(0.6 * bbox[1] + 0.4 * old[1]),
                    int(0.6 * bbox[2] + 0.4 * old[2]),
                    int(0.6 * bbox[3] + 0.4 * old[3]),
                )
            self.smooth_bbox[tid] = bbox

            # Определить цвет
            color_id = get_torso_color(frame, bbox)
            if color_id > 0:
                self.color_votes[tid][color_id] += 1

            # Получить лучший цвет для этого трека
            votes = self.color_votes[tid]
            if votes:
                best_color = max(votes, key=votes.get)
                total_votes = sum(votes.values())
                confidence = votes[best_color] / total_votes if total_votes > 0 else 0

                results.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "color_id": best_color,
                    "color_name": COLORS[best_color]["name"],
                    "bgr": COLORS[best_color]["bgr"],
                    "votes": dict(votes),
                    "confidence": confidence
                })

        return results


def draw(frame, jockeys, frame_num):
    # Отрисовка
    for j in jockeys:
        x1, y1, x2, y2 = j["bbox"]
        bgr = j["bgr"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)

        # Торс ROI
        h = y2 - y1
        ty2 = y1 + int(h * 0.40)
        cv2.line(frame, (x1, ty2), (x2, ty2), (255, 255, 255), 1)

        # Label
        label = f"T{j['track_id']} {j['color_name']}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)

        # Votes
        votes_str = " ".join([f"{COLORS[c]['name'][0]}:{v}" for c, v in sorted(j["votes"].items())])
        cv2.putText(frame, votes_str, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Info
    cv2.putText(frame, f"Frame {frame_num} | Tracks: {len(jockeys)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=== SIMPLE JOCKEY TRACKING v4 ===")
    print("ByteTrack + Color Voting")
    print()

    model = YOLO("models/yolov8n.pt")
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_simple.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect
        res = model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD,
                           classes=[PERSON_CLASS_ID], device="cuda:0", half=True, verbose=False)

        dets = [Detection(bbox=(int(d[0]), int(d[1]), int(d[2]), int(d[3])), confidence=d[4])
                for d in res[0].boxes.data.cpu().numpy()]

        # Track
        jockeys = tracker.update(frame, dets)

        # Draw
        frame = draw(frame, jockeys, frame_num)

        if writer:
            writer.write(frame)

        disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
        cv2.imshow("Simple Tracking", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if frame_num % 100 == 0:
            colors = [j["color_name"] for j in jockeys]
            print(f"Frame {frame_num}/{total} | {colors}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
