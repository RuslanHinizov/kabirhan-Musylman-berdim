"""
Finish Line Detector
- Detect when horse reaches 75% of screen width
- Determine finish order by color
- Result: 1st green, 2nd red, etc.
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
CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 1280
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45
TORSO_LEFT = 0.15
TORSO_RIGHT = 0.15

# Finish line zone (75% from left)
FINISH_ZONE_START = 0.75

# Color classifier confidence
MIN_COLOR_CONFIDENCE = 0.5

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


class FinishDetector:
    def __init__(self):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8s.pt")

        print("Loading color classifier...")
        self.classifier = ColorClassifier()

        # Finish results
        self.finish_order = []  # List of colors in finish order
        self.finished_colors = set()  # Colors that already finished

        # Detection state
        self.detection_started = False
        self.frame_width = 0
        self.finish_line_x = 0

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

    def update(self, frame):
        self.frame_num += 1
        self.frame_width = frame.shape[1]
        self.finish_line_x = int(self.frame_width * FINISH_ZONE_START)

        # Track persons
        results = self.yolo.track(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.3,
            classes=[0],
            tracker="botsort.yaml",
            persist=True,
            device="cuda:0",
            half=True,
            verbose=False
        )

        detections = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        # Process each detection
        for bbox, det_conf in zip(boxes, confs):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2

            # Classify color
            torso = self.extract_torso(frame, bbox)
            color, color_conf = self.classifier.predict(torso)

            if color is None or color_conf < MIN_COLOR_CONFIDENCE:
                color = "unknown"

            detections.append({
                'bbox': tuple(map(int, bbox)),
                'center_x': center_x,
                'color': color,
                'color_conf': color_conf,
                'in_finish_zone': center_x >= self.finish_line_x,
            })

            # Check if crossed finish line
            if center_x >= self.finish_line_x and color != "unknown":
                if color not in self.finished_colors:
                    self.finished_colors.add(color)
                    self.finish_order.append(color)
                    position = len(self.finish_order)
                    print(f"  [FINISH] {position}. {color.upper()}")

        return detections

    def get_results(self):
        return self.finish_order.copy()

    def is_race_complete(self):
        return len(self.finish_order) >= 5


def draw(frame, detections, detector):
    h, w = frame.shape[:2]
    finish_x = detector.finish_line_x

    # Draw finish zone
    cv2.line(frame, (finish_x, 0), (finish_x, h), (0, 255, 0), 3)
    cv2.putText(frame, "FINISH 75%", (finish_x + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        in_zone = det['in_finish_zone']

        # Box
        thick = 4 if in_zone else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thick)

        # Label
        label = f"{color}"
        if in_zone:
            label += " [FINISH]"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    # Draw results panel
    cv2.rectangle(frame, (5, 5), (300, 50 + len(detector.finish_order) * 35), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame {detector.frame_num} | FINISH ORDER:", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y = 70
    for i, color in enumerate(detector.finish_order):
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        text = f"{i+1}. {color.upper()}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
        y += 35

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("FINISH LINE DETECTOR")
    print("Detects finish order when jockeys reach 75% of screen")
    print("=" * 60)

    detector = FinishDetector()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")
    print(f"Finish line at: {int(w * FINISH_ZONE_START)}px (75%)")
    print()

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_finish.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving to: {out}")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        detections = detector.update(frame)
        frame_draw = draw(frame.copy(), detections, detector)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Finish Detector", cv2.WINDOW_NORMAL)
        cv2.imshow("Finish Detector", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        # Stop if race complete
        if detector.is_race_complete():
            print("\n*** RACE COMPLETE! ***")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    for i, color in enumerate(detector.finish_order):
        print(f"  {i+1}. {color.upper()}")

    if len(detector.finish_order) < 5:
        print(f"\n  (Only {len(detector.finish_order)}/5 detected)")
    print("=" * 60)


if __name__ == "__main__":
    main()
