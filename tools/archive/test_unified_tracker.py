"""
Unified Jockey Tracker
1 color = 1 jockey (no duplicates)

Logic:
- Detect person -> classify color
- If color already has owner -> assign to that owner
- If color is new -> create new jockey
- Result: exactly 5 jockeys (green, red, yellow, blue, purple)
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
CONFIDENCE_THRESHOLD = 0.15  # Lower to detect from far
IMAGE_SIZE = 1280
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45
TORSO_LEFT = 0.15
TORSO_RIGHT = 0.15

# Classification
MIN_CONFIDENCE = 0.5  # Min confidence to assign color

# Display colors
COLORS_BGR = {
    "blue":   (255, 100, 0),
    "green":  (0, 200, 0),
    "purple": (180, 0, 180),
    "red":    (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}

# Jockey IDs by color
JOCKEY_IDS = {
    "green": 1,
    "red": 2,
    "yellow": 3,
    "blue": 4,
    "purple": 5,
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
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = checkpoint['classes']
        img_size = checkpoint.get('img_size', 64)

        self.model = SimpleColorCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_bgr):
        if image_bgr is None or image_bgr.size < 100:
            return None, 0.0

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)

        return self.classes[pred.item()], conf.item()


class UnifiedJockeyTracker:
    """
    Color = Identity
    5 colors = 5 jockeys, no more
    """

    def __init__(self):
        print("Loading YOLO (yolov8s - better for groups)...")
        self.yolo = YOLO("yolov8s.pt")  # larger model, better separation

        print("Loading color classifier...")
        self.classifier = ColorClassifier()

        # Jockey state: color -> {bbox, last_seen, smoothed_bbox}
        self.jockeys = {}

        self.frame_num = 0

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

    def smooth_bbox(self, color, bbox, alpha=0.8):
        """Smooth bbox for specific jockey (by color)"""
        if color not in self.jockeys or self.jockeys[color].get('bbox') is None:
            return bbox

        old = self.jockeys[color]['bbox']
        # High alpha = follow new detection quickly (less lag)
        smoothed = tuple(int(alpha * b + (1 - alpha) * o) for b, o in zip(bbox, old))
        return smoothed

    def update(self, frame):
        self.frame_num += 1

        # Track persons with BoT-SORT (faster, better temporal consistency)
        results = self.yolo.track(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.3,  # Lower NMS threshold = more detections of close objects
            classes=[0],
            tracker="botsort.yaml",
            persist=True,
            device="cuda:0",
            half=True,
            verbose=False
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return self._get_output()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # For each detection, classify color and assign to jockey
        detections_by_color = defaultdict(list)  # color -> [(bbox, conf, color_conf)]

        for bbox, det_conf in zip(boxes, confs):
            torso = self.extract_torso(frame, bbox)
            color, color_conf = self.classifier.predict(torso)

            if color and color_conf >= MIN_CONFIDENCE:
                detections_by_color[color].append({
                    'bbox': tuple(bbox),
                    'det_conf': det_conf,
                    'color_conf': color_conf,
                })

        # Update jockeys - one detection per color (best confidence)
        for color, dets in detections_by_color.items():
            # Pick best detection for this color
            best = max(dets, key=lambda d: d['color_conf'] * d['det_conf'])
            bbox = best['bbox']

            # Smooth bbox
            bbox = self.smooth_bbox(color, bbox)

            # Update jockey
            self.jockeys[color] = {
                'bbox': bbox,
                'last_seen': self.frame_num,
                'det_conf': best['det_conf'],
                'color_conf': best['color_conf'],
            }

        return self._get_output()

    def _get_output(self):
        """Get current jockeys (only recently seen)"""
        output = []
        for color, data in self.jockeys.items():
            # Skip if not seen recently (more than 60 frames = 2.4 sec)
            if self.frame_num - data['last_seen'] > 60:
                continue

            jockey_id = JOCKEY_IDS.get(color, 0)
            output.append({
                'jockey_id': jockey_id,
                'color': color,
                'bbox': data['bbox'],
                'det_conf': data['det_conf'],
                'color_conf': data['color_conf'],
                'last_seen': data['last_seen'],
            })

        return output


def draw(frame, jockeys, frame_num):
    for j in jockeys:
        x1, y1, x2, y2 = map(int, j['bbox'])
        color = j['color']
        jid = j['jockey_id']
        bgr = COLORS_BGR.get(color, (128, 128, 128))

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)

        # Torso line
        h = y2 - y1
        ty2 = y1 + int(h * TORSO_BOTTOM)
        cv2.line(frame, (x1, ty2), (x2, ty2), (255, 255, 255), 1)

        # Label
        label = f"J{jid} {color}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 3)

    # Info panel
    cv2.rectangle(frame, (5, 5), (250, 170), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame {frame_num}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Jockeys: {len(jockeys)}/5", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Jockey list
    y = 80
    for color in ["green", "red", "yellow", "blue", "purple"]:
        jid = JOCKEY_IDS[color]
        bgr = COLORS_BGR[color]
        found = any(j['color'] == color for j in jockeys)
        status = "OK" if found else "---"
        status_color = (0, 255, 0) if found else (100, 100, 100)

        cv2.rectangle(frame, (15, y), (35, y+15), bgr, -1)
        cv2.putText(frame, f"J{jid} {color}: {status}", (45, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y += 18

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("UNIFIED JOCKEY TRACKER")
    print("1 Color = 1 Jockey (No Duplicates)")
    print("=" * 60)

    tracker = UnifiedJockeyTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_unified.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving to: {out}")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        jockeys = tracker.update(frame)
        frame_draw = draw(frame.copy(), jockeys, frame_num)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Unified Tracker", cv2.WINDOW_NORMAL)
        cv2.imshow("Unified Tracker", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if frame_num % 100 == 0:
            colors = [j['color'] for j in jockeys]
            print(f"Frame {frame_num}/{total} | Jockeys: {colors}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Final jockeys: {list(tracker.jockeys.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
