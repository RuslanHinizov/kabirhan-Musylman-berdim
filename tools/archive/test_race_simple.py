"""
Simple Race Tracker

1. Всегда: лёгкая детекция лошадей (следим за X)
2. Триггер 30%: первая лошадь дошла → включаем классификацию
3. Классификация: находим все 5 цветов (K кадров стабильно)
4. Фиксация: color→ID, дальше только трекинг
5. Финиш: определяем места по X
"""

import cv2
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# === SETTINGS ===
DETECTION_CONF = 0.15
IMAGE_SIZE = 1280

# Zones
TRIGGER_ZONE = 0.0       # 0% - start immediately
FINISH_ZONE = 0.70       # 70% - finish line

# Classification
K_FRAMES_CONFIRM = 1     # 1 frame = instant confirm
MIN_COLOR_CONF = 0.4

# Torso
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

COLORS_BGR = {
    "blue": (255, 100, 0),
    "green": (0, 200, 0),
    "purple": (180, 0, 180),
    "red": (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}


class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class ColorClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("models/color_classifier.pt", map_location=self.device, weights_only=False)
        self.classes = ckpt['classes']
        self.model = SimpleColorCNN().to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_bgr):
        if img_bgr is None or img_bgr.size < 100:
            return None, 0.0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)
            conf, pred = probs.max(1)
        return self.classes[pred.item()], conf.item()


class RaceTracker:
    def __init__(self):
        print("Loading YOLO (light)...")
        self.yolo = YOLO("yolov8s.pt")
        self.classifier = None  # Load on demand

        self.frame_num = 0
        self.frame_width = 0

        # State
        self.state = "WAIT"  # WAIT → CLASSIFY → FIXED → FINISH
        self.classification_triggered = False

        # Color registry
        self.color_votes = defaultdict(lambda: defaultdict(int))
        self.color_registry = {}  # color → jockey_id
        self.all_colors_found = False

        # Tracking
        self.jockey_bbox = {}     # jockey_id → bbox
        self.jockey_x = {}        # jockey_id → x position

        # Finish
        self.finish_order = []
        self.finished = set()

    def _load_classifier(self):
        if self.classifier is None:
            print("  [TRIGGER] Loading color classifier...")
            self.classifier = ColorClassifier()

    def _extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        ty1 = max(0, y1 + int(h * TORSO_TOP))
        ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
        tx1 = max(0, x1 + int(w * 0.15))
        tx2 = min(frame.shape[1], x2 - int(w * 0.15))
        if ty2 - ty1 < 20 or tx2 - tx1 < 20:
            return None
        return frame[ty1:ty2, tx1:tx2]

    def update(self, frame):
        self.frame_num += 1
        self.frame_width = frame.shape[1]
        trigger_x = self.frame_width * TRIGGER_ZONE
        finish_x = self.frame_width * FINISH_ZONE

        # Always detect
        results = self.yolo.track(
            frame, imgsz=IMAGE_SIZE, conf=DETECTION_CONF, iou=0.3,
            classes=[0], tracker="botsort.yaml", persist=True,
            device="cuda:0", half=True, verbose=False
        )

        output = []
        if results[0].boxes is None or results[0].boxes.id is None:
            return output

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Start classification immediately on first detection
        if not self.classification_triggered and len(boxes) > 0:
            self.classification_triggered = True
            self.state = "CLASSIFY"
            self._load_classifier()
            print(f"  [START] Classification started (frame {self.frame_num})")

        # Process detections
        for bbox, tid in zip(boxes, track_ids):
            center_x = (bbox[0] + bbox[2]) / 2
            color = "unknown"
            jockey_id = None

            # === CLASSIFY state: find all colors ===
            if self.state == "CLASSIFY":
                torso = self._extract_torso(frame, bbox)
                color, conf = self.classifier.predict(torso)

                if color and conf >= MIN_COLOR_CONF:
                    self.color_votes[tid][color] += 1

                    # Check if color confirmed
                    votes = self.color_votes[tid]
                    best_color = max(votes, key=votes.get)
                    if votes[best_color] >= K_FRAMES_CONFIRM:
                        if best_color not in self.color_registry:
                            jid = len(self.color_registry) + 1
                            self.color_registry[best_color] = jid
                            print(f"  [FOUND] {best_color.upper()} → J{jid}")

                # Check if all 5 colors found
                if len(self.color_registry) >= 5 and not self.all_colors_found:
                    self.all_colors_found = True
                    self.state = "FIXED"
                    print(f"\n*** ALL 5 COLORS FOUND ***")
                    print(f"    Registry: {self.color_registry}\n")

            # === FIXED state: use registry ===
            if self.state in ["FIXED", "FINISH"]:
                # Quick color check
                torso = self._extract_torso(frame, bbox)
                color, conf = self.classifier.predict(torso) if self.classifier else (None, 0)

                if color in self.color_registry:
                    jockey_id = self.color_registry[color]
                    self.jockey_bbox[jockey_id] = tuple(map(int, bbox))
                    self.jockey_x[jockey_id] = center_x

            # === Check finish ===
            if center_x >= finish_x and jockey_id and jockey_id not in self.finished:
                self.finished.add(jockey_id)
                self.finish_order.append(jockey_id)
                pos = len(self.finish_order)
                color_name = [c for c, j in self.color_registry.items() if j == jockey_id][0]
                print(f"  [FINISH] {pos}. {color_name.upper()}")

                if len(self.finish_order) >= 5:
                    self.state = "DONE"

            output.append({
                'tid': tid,
                'jockey_id': jockey_id,
                'bbox': tuple(map(int, bbox)),
                'center_x': center_x,
                'color': color if color else "unknown",
            })

        return output

    def get_results(self):
        results = []
        for jid in self.finish_order:
            color = [c for c, j in self.color_registry.items() if j == jid][0]
            results.append(color)
        return results


def draw(frame, dets, tracker):
    h, w = frame.shape[:2]
    trigger_x = int(w * TRIGGER_ZONE)
    finish_x = int(w * FINISH_ZONE)

    # Lines
    cv2.line(frame, (trigger_x, 0), (trigger_x, h), (0, 255, 255), 2)
    cv2.line(frame, (finish_x, 0), (finish_x, h), (0, 255, 0), 3)
    cv2.putText(frame, "30%", (trigger_x+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, "FINISH", (finish_x+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Detections
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        color = d['color']
        jid = d['jockey_id']
        bgr = COLORS_BGR.get(color, (128,128,128))

        cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 3 if jid else 1)
        label = f"J{jid} {color}" if jid else color
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # Panel
    cv2.rectangle(frame, (5,5), (280, 180), (0,0,0), -1)
    cv2.putText(frame, f"Frame {tracker.frame_num}", (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"State: {tracker.state}", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"Colors: {len(tracker.color_registry)}/5", (15,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Results
    y = 100
    for i, jid in enumerate(tracker.finish_order):
        color = [c for c,j in tracker.color_registry.items() if j==jid][0]
        bgr = COLORS_BGR.get(color, (128,128,128))
        cv2.putText(frame, f"{i+1}. {color}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
        y += 25

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("SIMPLE RACE TRACKER")
    print("Trigger at 30% → Classify → Fix → Finish")
    print("=" * 50)

    tracker = RaceTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames\n")

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_race.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = tracker.update(frame)
        frame_draw = draw(frame.copy(), dets, tracker)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Race", cv2.WINDOW_NORMAL)
        cv2.imshow("Race", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if tracker.state == "DONE":
            cv2.waitKey(2000)
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Results
    print("\n" + "=" * 50)
    print("RESULTS:")
    for i, color in enumerate(tracker.get_results()):
        print(f"  {i+1}. {color.upper()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
