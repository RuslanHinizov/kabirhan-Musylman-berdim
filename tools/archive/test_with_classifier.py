"""
Jockey tracking with trained color classifier
YOLO detection + BoT-SORT tracking + CNN color classification
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

# Settings
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1280
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45
TORSO_LEFT = 0.15
TORSO_RIGHT = 0.15

# Color voting
MIN_VOTES_TO_LOCK = 10
LOCK_CONFIDENCE = 0.7

# Display colors
COLORS_BGR = {
    "blue":   (255, 100, 0),
    "green":  (0, 200, 0),
    "purple": (180, 0, 180),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}


class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ColorClassifier:
    def __init__(self, model_path="models/color_classifier.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        img_size = checkpoint.get('img_size', 64)

        # Load model
        self.model = SimpleColorCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"Color classifier loaded: {self.classes}")

    def predict(self, image_bgr):
        """Predict color from BGR image"""
        if image_bgr is None or image_bgr.size < 100:
            return None, 0.0

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Transform
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)

        return self.classes[pred.item()], conf.item()


class JockeyTracker:
    def __init__(self):
        print("Loading YOLO...")
        self.yolo = YOLO("models/yolov8n.pt")

        print("Loading color classifier...")
        self.classifier = ColorClassifier()

        # Track state
        self.track_votes = defaultdict(lambda: defaultdict(int))
        self.track_locked = {}  # track_id -> color
        self.track_bbox = {}    # track_id -> smoothed bbox

    def extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        tx1 = x1 + int(w * TORSO_LEFT)
        tx2 = x2 - int(w * TORSO_RIGHT)

        ty1 = max(0, ty1)
        ty2 = min(frame.shape[0], ty2)
        tx1 = max(0, tx1)
        tx2 = min(frame.shape[1], tx2)

        if ty2 - ty1 < 20 or tx2 - tx1 < 20:
            return None

        return frame[ty1:ty2, tx1:tx2]

    def smooth_bbox(self, track_id, bbox, alpha=0.6):
        if track_id not in self.track_bbox:
            self.track_bbox[track_id] = bbox
            return bbox

        old = self.track_bbox[track_id]
        smoothed = tuple(int(alpha * b + (1 - alpha) * o) for b, o in zip(bbox, old))
        self.track_bbox[track_id] = smoothed
        return smoothed

    def update(self, frame):
        # Detect & track
        results = self.yolo.track(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
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

        for bbox, tid in zip(boxes, track_ids):
            bbox = self.smooth_bbox(tid, tuple(bbox))

            # Get color
            if tid in self.track_locked:
                color = self.track_locked[tid]
                conf = 1.0
                locked = True
            else:
                # Extract torso and classify
                torso = self.extract_torso(frame, bbox)
                color, conf = self.classifier.predict(torso)

                if color and conf > 0.5:
                    self.track_votes[tid][color] += 1

                # Check if can lock
                votes = self.track_votes[tid]
                total = sum(votes.values())
                if total >= MIN_VOTES_TO_LOCK:
                    best_color = max(votes, key=votes.get)
                    best_ratio = votes[best_color] / total
                    if best_ratio >= LOCK_CONFIDENCE:
                        self.track_locked[tid] = best_color
                        color = best_color
                        print(f"  [LOCK] Track {tid} -> {best_color}")

                locked = tid in self.track_locked

            output.append({
                "track_id": tid,
                "bbox": bbox,
                "color": color if color else "unknown",
                "conf": conf,
                "locked": locked,
                "votes": dict(self.track_votes[tid]),
            })

        return output


def draw(frame, tracks, frame_num, tracker):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = t["color"]
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        locked = t["locked"]

        # Box
        thick = 4 if locked else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thick)

        # Torso line
        h = y2 - y1
        ty2 = y1 + int(h * TORSO_BOTTOM)
        cv2.line(frame, (x1, ty2), (x2, ty2), (255, 255, 255), 1)

        # Label
        status = "OK" if locked else f"{t['conf']:.0%}"
        label = f"T{t['track_id']} {color} {status}"
        cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    # Info
    locked_count = len(tracker.track_locked)
    cv2.rectangle(frame, (5, 5), (320, 180), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Tracks: {len(tracks)} | Locked: {locked_count}/5",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Locked colors
    y = 80
    for color in ["green", "red", "yellow", "blue", "purple"]:
        is_locked = color in tracker.track_locked.values()
        bgr = COLORS_BGR[color]
        status_color = (0, 255, 0) if is_locked else (100, 100, 100)

        cv2.rectangle(frame, (15, y), (35, y+15), bgr, -1)
        tid = [k for k, v in tracker.track_locked.items() if v == color]
        tid_str = f"T{tid[0]}" if tid else "---"
        cv2.putText(frame, f"{color}: {tid_str}",
                    (45, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += 20

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("JOCKEY TRACKING WITH CNN CLASSIFIER")
    print("=" * 60)

    tracker = JockeyTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_classifier.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving to: {out}")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        tracks = tracker.update(frame)
        frame_draw = draw(frame.copy(), tracks, frame_num, tracker)

        if writer:
            writer.write(frame_draw)

        try:
            disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
            cv2.imshow("Jockey Tracking", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        except:
            pass

        if frame_num % 100 == 0:
            locked = list(tracker.track_locked.items())
            print(f"Frame {frame_num}/{total} | Locked: {locked}")

    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    for tid, color in sorted(tracker.track_locked.items()):
        print(f"  Track {tid}: {color}")
    print("=" * 60)


if __name__ == "__main__":
    main()
