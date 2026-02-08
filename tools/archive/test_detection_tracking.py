"""
Test Detection + Tracking
Проверка детекции YOLO + ByteTrack трекинг

Usage:
    python tools/test_detection_tracking.py data/videos/exp10_cam1.mp4
    python tools/test_detection_tracking.py data/videos/exp10_cam1.mp4 --save
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from api.vision.bytetrack import ByteTracker
from api.vision.dtypes import Detection

# Class IDs
HORSE_CLASS_ID = 17
PERSON_CLASS_ID = 0

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960

# Colors for tracks (by ID)
TRACK_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 255, 0),  # Lime
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Sky blue
]


def get_color(track_id: int) -> tuple:
    """Get color for track ID"""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_tracks(frame, horse_tracks, person_tracks, frame_num):
    """Draw tracked objects on frame"""

    # Draw horses (thicker box)
    for track in horse_tracks:
        x1, y1, x2, y2 = track.bbox
        tid = track.track_id
        conf = track.confidence
        color = get_color(tid)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw label
        label = f"H{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw center point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)

    # Draw persons (thinner box)
    for track in person_tracks:
        x1, y1, x2, y2 = track.bbox
        tid = track.track_id
        conf = track.confidence
        color = get_color(tid + 100)  # Offset color

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"P{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
        cv2.putText(frame, label, (x1 + 5, y2 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw info
    info = f"Frame {frame_num} | Horses: {len(horse_tracks)} | Persons: {len(person_tracks)}"
    cv2.putText(frame, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    return frame


def test_detection_tracking(video_path: str, show: bool = True, save: bool = False):
    """Run detection + tracking test"""

    print(f"\n{'='*60}")
    print(f"DETECTION + TRACKING TEST")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Tracker: ByteTrack (Kalman Filter)")
    print(f"{'='*60}\n")

    # Check video exists
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("models/yolov8n.pt")
    print("Model loaded!")

    # Create trackers (separate for horses and persons)
    horse_tracker = ByteTracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        low_thresh=0.1,
        confirm_frames=3
    )
    person_tracker = ByteTracker(
        track_thresh=0.4,
        track_buffer=20,
        match_thresh=0.7,
        low_thresh=0.1,
        confirm_frames=2
    )
    print("ByteTrack trackers created!\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames\n")

    # Resize for display if 4K
    display_scale = 0.5 if width > 1920 else 1.0
    display_w = int(width * display_scale)
    display_h = int(height * display_scale)

    # Video writer
    writer = None
    if save:
        output_path = str(Path(video_path).stem) + "_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving to: {output_path}\n")

    # Stats
    unique_horses = set()
    unique_persons = set()
    frame_num = 0

    print("Processing... (Press 'q' to quit, SPACE to pause)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run YOLO detection
        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[HORSE_CLASS_ID, PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Parse detections
        detections = results[0].boxes.data.cpu().numpy()

        horse_dets = []
        person_dets = []

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if int(cls_id) == HORSE_CLASS_ID:
                horse_dets.append(Detection(bbox=bbox, confidence=conf))
            elif int(cls_id) == PERSON_CLASS_ID:
                person_dets.append(Detection(bbox=bbox, confidence=conf))

        # Update trackers
        horse_tracks = horse_tracker.update(horse_dets)
        person_tracks = person_tracker.update(person_dets)

        # Track unique IDs
        for t in horse_tracks:
            unique_horses.add(t.track_id)
        for t in person_tracks:
            unique_persons.add(t.track_id)

        # Draw
        frame_drawn = draw_tracks(frame, horse_tracks, person_tracks, frame_num)

        # Save
        if writer:
            writer.write(frame_drawn)

        # Show
        if show:
            # Resize for display
            if display_scale != 1.0:
                display_frame = cv2.resize(frame_drawn, (display_w, display_h))
            else:
                display_frame = frame_drawn

            cv2.imshow("Detection + Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Paused at frame {frame_num}. Press any key...")
                cv2.waitKey(0)

        # Progress
        if frame_num % 50 == 0:
            print(f"Frame {frame_num}/{total_frames} | "
                  f"Horse tracks: {len(horse_tracks)} (unique: {len(unique_horses)}) | "
                  f"Person tracks: {len(person_tracks)} (unique: {len(unique_persons)})")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # Summary
    print(f"\n{'='*60}")
    print(f"TRACKING SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames: {frame_num}")
    print(f"Unique horse tracks: {len(unique_horses)}")
    print(f"Unique person tracks: {len(unique_persons)}")
    print(f"Horse IDs: {sorted(unique_horses)}")
    print(f"Person IDs: {sorted(unique_persons)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test Detection + Tracking")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-show", action="store_true", help="Don't show window")

    args = parser.parse_args()

    test_detection_tracking(
        video_path=args.video,
        show=not args.no_show,
        save=args.save
    )


if __name__ == "__main__":
    main()
