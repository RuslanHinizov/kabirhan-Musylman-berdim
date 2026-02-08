"""
Extract torso crops from video for manual review
Saves to folders by track_id
"""

import cv2
import sys
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Settings
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1280

# Torso region - EXPERIMENT with these!
TORSO_TOP = 0.0       # Start from top of bbox
TORSO_BOTTOM = 0.50   # 50% of height
TORSO_LEFT = 0.15     # Skip 15% from sides
TORSO_RIGHT = 0.15

# Save every N frames (to avoid too many similar images)
SAVE_EVERY_N = 5

# Min crop size
MIN_CROP_SIZE = 30


def extract_torso(frame, bbox):
    """Extract torso region from bbox"""
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    # Calculate torso region
    ty1 = y1 + int(h * TORSO_TOP)
    ty2 = y1 + int(h * TORSO_BOTTOM)
    tx1 = x1 + int(w * TORSO_LEFT)
    tx2 = x2 - int(w * TORSO_RIGHT)

    # Clamp to frame
    ty1 = max(0, ty1)
    ty2 = min(frame.shape[0], ty2)
    tx1 = max(0, tx1)
    tx2 = min(frame.shape[1], tx2)

    if ty2 - ty1 < MIN_CROP_SIZE or tx2 - tx1 < MIN_CROP_SIZE:
        return None

    return frame[ty1:ty2, tx1:tx2].copy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", default="data/torso_crops")
    parser.add_argument("--top", type=float, default=0.0, help="Torso top %")
    parser.add_argument("--bottom", type=float, default=0.50, help="Torso bottom %")
    args = parser.parse_args()

    global TORSO_TOP, TORSO_BOTTOM
    TORSO_TOP = args.top
    TORSO_BOTTOM = args.bottom

    print("=" * 60)
    print("TORSO EXTRACTOR")
    print(f"Torso region: {TORSO_TOP*100:.0f}% - {TORSO_BOTTOM*100:.0f}%")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print("Loading YOLO...")
    model = YOLO("models/yolov8n.pt")

    # Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {total} frames @ {fps:.0f}fps")
    print()

    # Track counts
    track_counts = defaultdict(int)
    saved_total = 0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Track with BoT-SORT
        results = model.track(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],  # person
            tracker="botsort.yaml",
            persist=True,
            device="cuda:0",
            half=True,
            verbose=False
        )

        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for bbox, tid in zip(boxes, track_ids):
            track_counts[tid] += 1

            # Save every N frames per track
            if track_counts[tid] % SAVE_EVERY_N != 0:
                continue

            # Extract torso
            torso = extract_torso(frame, bbox)
            if torso is None:
                continue

            # Create track folder
            track_dir = os.path.join(args.output, f"track_{tid:03d}")
            os.makedirs(track_dir, exist_ok=True)

            # Save
            filename = f"frame_{frame_num:05d}.jpg"
            filepath = os.path.join(track_dir, filename)
            cv2.imwrite(filepath, torso)
            saved_total += 1

        if frame_num % 200 == 0:
            print(f"Frame {frame_num}/{total} | Tracks: {len(track_counts)} | Saved: {saved_total}")

    cap.release()

    # Summary
    print()
    print("=" * 60)
    print("DONE!")
    print(f"Total tracks: {len(track_counts)}")
    print(f"Total crops saved: {saved_total}")
    print()
    print("Tracks by frame count:")
    for tid, count in sorted(track_counts.items(), key=lambda x: -x[1])[:10]:
        folder = os.path.join(args.output, f"track_{tid:03d}")
        n_files = len(os.listdir(folder)) if os.path.exists(folder) else 0
        print(f"  Track {tid}: {count} frames, {n_files} crops saved")
    print()
    print(f"Output folder: {args.output}")
    print("Review folders and delete bad crops!")
    print("=" * 60)


if __name__ == "__main__":
    main()
