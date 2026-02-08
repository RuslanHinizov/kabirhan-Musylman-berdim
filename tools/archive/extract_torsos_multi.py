"""
Extract torsos from multiple videos
Classify by color and save to folders
"""

import cv2
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Settings
DETECTION_CONF = 0.15
MIN_COLOR_CONF = 0.3  # Lower threshold to catch more
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45
SAVE_EVERY_N = 1  # Save every frame

OUTPUT_DIR = "data/torso_crops_new"


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


def extract_torso(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    ty1 = max(0, y1 + int(h * TORSO_TOP))
    ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
    tx1 = max(0, x1 + int(w * 0.15))
    tx2 = min(frame.shape[1], x2 - int(w * 0.15))
    if ty2 - ty1 < 30 or tx2 - tx1 < 30:
        return None
    return frame[ty1:ty2, tx1:tx2]


def process_video(video_path, yolo, classifier, output_dir):
    video_name = Path(video_path).stem
    print(f"\n{'='*50}")
    print(f"Processing: {video_name}")
    print(f"{'='*50}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    saved_count = {color: 0 for color in classifier.classes}
    saved_count["unknown"] = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect (use simple detection, not tracking - more reliable)
        results = yolo(
            frame, imgsz=1280, conf=DETECTION_CONF, iou=0.3,
            classes=[0], device="cuda:0", half=True, verbose=False
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Save every N frames globally
        if frame_num % SAVE_EVERY_N != 0:
            continue

        for idx, bbox in enumerate(boxes):

            # Extract torso
            torso = extract_torso(frame, bbox)
            if torso is None:
                continue

            # Classify color
            color, conf = classifier.predict(torso)
            if color is None or conf < MIN_COLOR_CONF:
                color = "unknown"

            # Save to color folder
            color_dir = output_dir / color
            color_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{video_name}_f{frame_num:05d}_b{idx}.jpg"
            filepath = color_dir / filename
            cv2.imwrite(str(filepath), torso)
            saved_count[color] = saved_count.get(color, 0) + 1

        if frame_num % 50 == 0:
            print(f"  Frame {frame_num}/{total}")

    cap.release()

    print(f"\n  Saved crops:")
    for color, count in saved_count.items():
        if count > 0:
            print(f"    {color}: {count}")

    return saved_count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="Video files")
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("TORSO EXTRACTOR (Multi-Video)")
    print(f"Output: {output_dir}")
    print("=" * 50)

    print("\nLoading YOLO...")
    yolo = YOLO("yolov8s.pt")

    print("Loading classifier...")
    classifier = ColorClassifier()

    total_saved = {color: 0 for color in classifier.classes}
    total_saved["unknown"] = 0

    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping")
            continue

        saved = process_video(video_path, yolo, classifier, output_dir)
        for color, count in saved.items():
            total_saved[color] = total_saved.get(color, 0) + count

    # Summary
    print("\n" + "=" * 50)
    print("TOTAL SAVED:")
    for color, count in total_saved.items():
        if count > 0:
            print(f"  {color}: {count}")
    print(f"\nOutput: {output_dir}")
    print("Review and clean folders, then retrain!")
    print("=" * 50)


if __name__ == "__main__":
    main()
