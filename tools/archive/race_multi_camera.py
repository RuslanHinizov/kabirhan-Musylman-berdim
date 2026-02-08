"""
Multi-Camera Race Tracker
- 4 cameras covering 0-400m track
- Tracks distance and time for each jockey
- Determines finish order by finish time
"""

import cv2
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# === SETTINGS ===
DETECTION_CONF = 0.15
MIN_COLOR_CONF = 0.5
TORSO_TOP = 0.05
TORSO_BOTTOM = 0.45

# Camera distance mapping (meters)
CAMERA_SEGMENTS = {
    'cam1': (0, 100),      # 0-100m
    'cam2': (100, 200),    # 100-200m
    'cam3': (200, 300),    # 200-300m
    'cam4': (300, 400),    # 300-400m (finish)
}

FINISH_DISTANCE = 400  # meters

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


class MultiCameraRaceTracker:
    def __init__(self):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8s.pt")

        print("Loading classifier...")
        self.classifier = ColorClassifier()
        self.colors = self.classifier.classes

        # Jockey state: color -> {distance, time, camera, finished}
        self.jockeys = {color: {
            'max_distance': 0,
            'finish_time': None,
            'last_camera': None,
            'last_time': 0,
            'finished': False,
            'bbox': None,
        } for color in self.colors}

        # Timing
        self.total_time = 0
        self.current_fps = 25
        self.current_camera = None
        self.current_time = 0

        # Results
        self.finish_order = []

    def extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        ty1 = max(0, y1 + int(h * TORSO_TOP))
        ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
        tx1 = max(0, x1 + int(w * 0.15))
        tx2 = min(frame.shape[1], x2 - int(w * 0.15))

        if ty2 - ty1 < 20 or tx2 - tx1 < 20:
            return None

        return frame[ty1:ty2, tx1:tx2]

    def x_to_distance(self, x_norm, camera_name):
        """Convert normalized X (0-1) to distance in meters"""
        start_m, end_m = CAMERA_SEGMENTS[camera_name]
        return start_m + x_norm * (end_m - start_m)

    def process_frame(self, frame, camera_name, current_time, frame_width):
        """Process a single frame, return detections"""
        self.current_camera = camera_name
        self.current_time = current_time

        detections = []

        # Detect
        results = self.yolo(
            frame, imgsz=1280, conf=DETECTION_CONF, iou=0.15,
            classes=[0], device="cuda:0", half=True, verbose=False
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Process each detection
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            x_norm = center_x / frame_width

            # Classify color
            torso = self.extract_torso(frame, bbox)
            color, conf = self.classifier.predict(torso)

            if color is None or conf < MIN_COLOR_CONF:
                continue

            # Calculate distance
            distance = self.x_to_distance(x_norm, camera_name)

            # Update jockey state
            jockey = self.jockeys[color]

            if distance > jockey['max_distance']:
                jockey['max_distance'] = distance
                jockey['last_camera'] = camera_name
                jockey['last_time'] = current_time
                jockey['bbox'] = tuple(map(int, bbox))

                # Check finish
                if distance >= FINISH_DISTANCE * 0.95 and not jockey['finished']:
                    jockey['finished'] = True
                    jockey['finish_time'] = current_time
                    self.finish_order.append(color)
                    print(f"  [{current_time:.1f}s] {color.upper()} FINISHED! (#{len(self.finish_order)})")

            detections.append({
                'color': color,
                'bbox': tuple(map(int, bbox)),
                'distance': distance,
                'conf': conf,
            })

        return detections

    def get_standings(self):
        """Get current standings sorted by distance (desc), then by time (asc)"""
        standings = [(c, j) for c, j in self.jockeys.items() if j['max_distance'] > 0]
        # Sort: higher distance first (rounded to avoid float issues), then lower time first
        standings.sort(key=lambda x: (-round(x[1]['max_distance']), x[1]['last_time']))
        return standings

    def get_results(self):
        """Get final race results"""
        results = []

        # First add finished jockeys in order
        for color in self.finish_order:
            j = self.jockeys[color]
            results.append({
                'color': color,
                'distance': j['max_distance'],
                'time': j['finish_time'],
                'finished': True,
            })

        # Then add non-finished jockeys by distance
        unfinished = [(c, j) for c, j in self.jockeys.items()
                      if not j['finished'] and j['max_distance'] > 0]
        unfinished.sort(key=lambda x: -x[1]['max_distance'])

        for color, j in unfinished:
            results.append({
                'color': color,
                'distance': j['max_distance'],
                'time': j['last_time'],
                'finished': False,
            })

        return results


def draw_frame(frame, detections, tracker, camera_name):
    """Draw detections and info panel on frame"""
    h, w = frame.shape[:2]

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        dist = det['distance']

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
        label = f"{color} {dist:.0f}m"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    # Info panel (left side)
    cv2.rectangle(frame, (5, 5), (350, 280), (0, 0, 0), -1)

    # Header
    seg = CAMERA_SEGMENTS[camera_name]
    cv2.putText(frame, f"{camera_name.upper()}: {seg[0]}-{seg[1]}m",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Time: {tracker.current_time:.1f}s",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Standings
    cv2.putText(frame, "STANDINGS:", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y = 130
    for i, (color, j) in enumerate(tracker.get_standings()):
        bgr = COLORS_BGR.get(color, (128, 128, 128))
        status = "FIN" if j['finished'] else f"{j['max_distance']:.0f}m"
        text = f"{i+1}. {color}: {status}"
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
        y += 28

    # Finish order (right side)
    if tracker.finish_order:
        cv2.rectangle(frame, (w-250, 5), (w-5, 50 + len(tracker.finish_order)*30), (0, 0, 0), -1)
        cv2.putText(frame, "FINISH ORDER:", (w-240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y = 65
        for i, color in enumerate(tracker.finish_order):
            bgr = COLORS_BGR.get(color, (128, 128, 128))
            t = tracker.jockeys[color]['finish_time']
            cv2.putText(frame, f"{i+1}. {color} ({t:.1f}s)", (w-235, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            y += 28

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs=4, help="4 video files (cam1, cam2, cam3, cam4)")
    parser.add_argument("--output", default="results/multi_camera")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-display", action="store_true", help="Run without display")
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-CAMERA RACE TRACKER")
    print("Track: 0-400m | 4 cameras")
    print("Press Q to quit, SPACE to pause")
    print("=" * 60)

    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracker
    tracker = MultiCameraRaceTracker()

    # Process cameras in order
    camera_names = ['cam1', 'cam2', 'cam3', 'cam4']
    current_time = 0
    writer = None

    for video_path, cam_name in zip(args.videos, camera_names):
        if not Path(video_path).exists():
            print(f"WARNING: {video_path} not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {cam_name}: {CAMERA_SEGMENTS[cam_name][0]}-{CAMERA_SEGMENTS[cam_name][1]}m")
        print(f"{'='*50}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        if args.save and writer is None:
            out_path = str(output_dir / "race_full.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (frame_width, frame_height))
            print(f"Saving to: {out_path}")

        scale = 0.5 if frame_width > 1920 else 1.0
        frame_num = 0
        start_time = current_time

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            current_time = start_time + frame_num / fps

            # Process frame
            detections = tracker.process_frame(frame, cam_name, current_time, frame_width)

            # Draw
            frame_draw = draw_frame(frame.copy(), detections, tracker, cam_name)

            if writer:
                writer.write(frame_draw)

            # Display
            if not args.no_display:
                disp = cv2.resize(frame_draw, (int(frame_width*scale), int(frame_height*scale)))
                cv2.imshow("Race Tracker", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    if writer:
                        writer.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    cv2.waitKey(0)

            # Progress
            if frame_num % 100 == 0:
                print(f"  Frame {frame_num}/{total_frames} | Time: {current_time:.1f}s")

        cap.release()
        current_time = start_time + total_frames / fps

        # Save results for this camera
        cam_result_file = output_dir / f"{cam_name}_result.txt"
        with open(cam_result_file, "w") as f:
            f.write(f"{cam_name.upper()} RESULTS ({CAMERA_SEGMENTS[cam_name][0]}-{CAMERA_SEGMENTS[cam_name][1]}m)\n")
            f.write("=" * 40 + "\n\n")
            standings = tracker.get_standings()
            for i, (color, j) in enumerate(standings):
                status = f"{j['finish_time']:.1f}s FINISHED" if j['finished'] else f"{j['last_time']:.1f}s"
                line = f"{i+1}. {color.upper()}: {j['max_distance']:.0f}m | {status}"
                f.write(line + "\n")
        print(f"\n  [SAVED] {cam_result_file}")

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RACE RESULTS")
    print("=" * 60)

    results = tracker.get_results()

    result_lines = []
    for i, r in enumerate(results):
        status = f"{r['time']:.1f}s" if r['finished'] else f"DNF @ {r['distance']:.0f}m"
        line = f"{i+1}. {r['color'].upper()}: {r['distance']:.0f}m | {status}"
        print(f"  {line}")
        result_lines.append(line)

    # Save results
    result_file = output_dir / "race_result.txt"
    with open(result_file, "w") as f:
        f.write("MULTI-CAMERA RACE RESULTS\n")
        f.write("=" * 40 + "\n\n")
        for line in result_lines:
            f.write(line + "\n")

    print(f"\n  [SAVED] {result_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
