"""
Race Tracker with 3 Zones: START → MIDDLE → FINISH

ZONES:
- START (0-30%): Initialize color→ID, confirm K frames
- MIDDLE (30-70%): Track only, no ID changes
- FINISH (70-100%): Determine finish order

Result: 1st green, 2nd red, etc.
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
CONFIDENCE_THRESHOLD = 0.15
IMAGE_SIZE = 1280
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

# Zone boundaries (% of frame width)
START_ZONE_END = 0.30      # 0-30%
MIDDLE_ZONE_END = 0.70     # 30-70%
# FINISH zone is 70-100%

# Confirmation frames
K_FRAMES_INIT = 5          # Frames to confirm color in START
K_FRAMES_FINISH = 3        # Frames to confirm finish position

# Color classifier
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


class RaceZoneTracker:
    def __init__(self):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8s.pt")

        print("Loading color classifier...")
        self.classifier = ColorClassifier()

        # Frame info
        self.frame_num = 0
        self.frame_width = 0

        # === START ZONE: Color Registry ===
        self.color_registry = {}           # color -> jockey_id (1-5)
        self.color_votes = defaultdict(lambda: defaultdict(int))  # track_id -> {color: count}
        self.init_complete = False
        self.next_jockey_id = 1

        # === MIDDLE ZONE: Tracking ===
        self.jockey_bboxes = {}            # jockey_id -> smoothed bbox
        self.jockey_last_seen = {}         # jockey_id -> frame_num

        # === FINISH ZONE: Results ===
        self.finish_order = []             # List of jockey_ids in finish order
        self.finish_x_history = defaultdict(list)  # jockey_id -> [x positions]
        self.finished_jockeys = set()

        # State
        self.state = "INIT"  # INIT, TRACKING, FINISHING, DONE

    def get_zone(self, x):
        """Determine zone by x position"""
        if x < self.frame_width * START_ZONE_END:
            return "START"
        elif x < self.frame_width * MIDDLE_ZONE_END:
            return "MIDDLE"
        else:
            return "FINISH"

    def extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        tx1 = x1 + int(w * 0.15)
        tx2 = x2 - int(w * 0.15)
        ty1, ty2 = max(0, ty1), min(frame.shape[0], ty2)
        tx1, tx2 = max(0, tx1), min(frame.shape[1], tx2)
        if ty2 - ty1 < 20 or tx2 - tx1 < 20:
            return None
        return frame[ty1:ty2, tx1:tx2]

    def smooth_bbox(self, jockey_id, bbox, alpha=0.7):
        if jockey_id not in self.jockey_bboxes:
            self.jockey_bboxes[jockey_id] = bbox
            return bbox
        old = self.jockey_bboxes[jockey_id]
        smoothed = tuple(int(alpha * b + (1-alpha) * o) for b, o in zip(bbox, old))
        self.jockey_bboxes[jockey_id] = smoothed
        return smoothed

    def update(self, frame):
        self.frame_num += 1
        self.frame_width = frame.shape[1]

        # Track with YOLO
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

        output = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return output

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        if track_ids is None:
            return output
        track_ids = track_ids.cpu().numpy().astype(int)

        # Process detections
        for bbox, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            zone = self.get_zone(center_x)

            # Classify color
            torso = self.extract_torso(frame, bbox)
            color, color_conf = self.classifier.predict(torso)
            if color is None or color_conf < MIN_COLOR_CONFIDENCE:
                color = "unknown"

            jockey_id = None

            # === START ZONE: Initialize Registry ===
            if zone == "START" and not self.init_complete:
                if color != "unknown":
                    self.color_votes[tid][color] += 1

                    # Check if can register this color
                    votes = self.color_votes[tid]
                    total = sum(votes.values())
                    if total >= K_FRAMES_INIT:
                        best_color = max(votes, key=votes.get)
                        if best_color not in self.color_registry:
                            self.color_registry[best_color] = self.next_jockey_id
                            print(f"  [INIT] {best_color.upper()} -> Jockey #{self.next_jockey_id}")
                            self.next_jockey_id += 1

                # Check if init complete
                if len(self.color_registry) >= 5:
                    self.init_complete = True
                    self.state = "TRACKING"
                    print(f"\n*** INIT COMPLETE: {list(self.color_registry.keys())} ***\n")

            # Get jockey_id from registry
            if color in self.color_registry:
                jockey_id = self.color_registry[color]
                bbox_smooth = self.smooth_bbox(jockey_id, tuple(map(int, bbox)))
                self.jockey_last_seen[jockey_id] = self.frame_num
            else:
                bbox_smooth = tuple(map(int, bbox))

            # === FINISH ZONE: Record positions ===
            if zone == "FINISH" and jockey_id is not None:
                self.finish_x_history[jockey_id].append(center_x)

                # Check if crossed finish (reached 90% of width)
                if center_x >= self.frame_width * 0.90:
                    if jockey_id not in self.finished_jockeys:
                        # Confirm with K frames
                        recent_x = self.finish_x_history[jockey_id][-K_FRAMES_FINISH:]
                        if len(recent_x) >= K_FRAMES_FINISH and all(x >= self.frame_width * 0.85 for x in recent_x):
                            self.finished_jockeys.add(jockey_id)
                            self.finish_order.append(jockey_id)
                            position = len(self.finish_order)
                            color_name = self._get_color_by_id(jockey_id)
                            print(f"  [FINISH] {position}. {color_name.upper()} (Jockey #{jockey_id})")

                            if len(self.finish_order) >= 5:
                                self.state = "DONE"

            output.append({
                'track_id': tid,
                'jockey_id': jockey_id,
                'bbox': bbox_smooth,
                'center_x': center_x,
                'color': color,
                'color_conf': color_conf,
                'zone': zone,
            })

        return output

    def _get_color_by_id(self, jockey_id):
        for color, jid in self.color_registry.items():
            if jid == jockey_id:
                return color
        return "unknown"

    def get_results(self):
        """Get finish results as list of (position, color)"""
        results = []
        for i, jid in enumerate(self.finish_order):
            color = self._get_color_by_id(jid)
            results.append((i + 1, color))
        return results


def draw(frame, detections, tracker):
    h, w = frame.shape[:2]

    # Draw zone lines
    start_x = int(w * START_ZONE_END)
    finish_x = int(w * MIDDLE_ZONE_END)

    cv2.line(frame, (start_x, 0), (start_x, h), (255, 255, 0), 2)
    cv2.line(frame, (finish_x, 0), (finish_x, h), (0, 255, 0), 3)

    # Zone labels
    cv2.putText(frame, "START", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "MIDDLE", (start_x + 10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "FINISH", (finish_x + 10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        jid = det['jockey_id']
        zone = det['zone']
        bgr = COLORS_BGR.get(color, (128, 128, 128))

        # Box
        thick = 4 if zone == "FINISH" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thick)

        # Label
        if jid:
            label = f"J{jid} {color}"
        else:
            label = f"? {color}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    # Info panel
    panel_h = 220
    cv2.rectangle(frame, (5, 5), (320, panel_h), (0, 0, 0), -1)

    cv2.putText(frame, f"Frame {tracker.frame_num} | State: {tracker.state}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Registry
    cv2.putText(frame, "Registry:", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y = 75
    for color, jid in tracker.color_registry.items():
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        cv2.rectangle(frame, (15, y-10), (30, y+5), bgr, -1)
        cv2.putText(frame, f"J{jid}: {color}", (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    # Finish results
    if tracker.finish_order:
        cv2.putText(frame, "FINISH:", (15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 30
        for i, jid in enumerate(tracker.finish_order):
            color = tracker._get_color_by_id(jid)
            bgr = COLORS_BGR.get(color, (128, 128, 128))
            cv2.putText(frame, f"{i+1}. {color}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            y += 22

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RACE ZONE TRACKER")
    print("START (0-30%) → MIDDLE (30-70%) → FINISH (70-100%)")
    print("=" * 60)

    tracker = RaceZoneTracker()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")
    print(f"Zones: START <{int(w*0.3)}px | MIDDLE | FINISH >{int(w*0.7)}px")
    print()

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_zones.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving to: {out}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = tracker.update(frame)
        frame_draw = draw(frame.copy(), detections, tracker)

        if writer:
            writer.write(frame_draw)

        disp = cv2.resize(frame_draw, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Race Zones", cv2.WINDOW_NORMAL)
        cv2.imshow("Race Zones", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

        if tracker.state == "DONE":
            print("\n*** RACE COMPLETE ***")
            cv2.waitKey(2000)
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    results = tracker.get_results()
    for pos, color in results:
        print(f"  {pos}. {color.upper()}")
    if len(results) < 5:
        print(f"\n  (Only {len(results)}/5 finished)")
    print("=" * 60)


if __name__ == "__main__":
    main()
