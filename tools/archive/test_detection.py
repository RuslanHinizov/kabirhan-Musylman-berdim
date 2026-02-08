"""
Test Detection Only
Проверка детекции YOLO на одном видео

Usage:
    python tools/test_detection.py data/videos/exp10_cam1.mp4
    python tools/test_detection.py data/videos/exp10_cam1.mp4 --show
    python tools/test_detection.py data/videos/exp10_cam1.mp4 --save
"""

import cv2
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Class IDs
HORSE_CLASS_ID = 17
PERSON_CLASS_ID = 0

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960

# Colors for drawing
COLORS = {
    'horse': (0, 255, 0),    # Green
    'person': (255, 0, 0),   # Blue
}


def draw_detections(frame, detections, frame_num):
    """Draw bounding boxes on frame"""
    horses = 0
    persons = 0

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cls_id = int(det[5])

        if cls_id == HORSE_CLASS_ID:
            color = COLORS['horse']
            label = f"Horse {conf:.2f}"
            horses += 1
        elif cls_id == PERSON_CLASS_ID:
            color = COLORS['person']
            label = f"Person {conf:.2f}"
            persons += 1
        else:
            continue

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)

        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw frame info
    info = f"Frame {frame_num} | Horses: {horses} | Persons: {persons}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame, horses, persons


def test_detection(video_path: str, show: bool = False, save: bool = False,
                   skip_frames: int = 0, max_frames: int = 0):
    """Run detection test on video"""

    print(f"\n{'='*60}")
    print(f"DETECTION TEST")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Model: yolov8n.pt")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"Classes: Horse ({HORSE_CLASS_ID}), Person ({PERSON_CLASS_ID})")
    print(f"{'='*60}\n")

    # Check video exists
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("models/yolov8n.pt")
    print("Model loaded!\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s\n")

    # Setup video writer if saving
    writer = None
    if save:
        output_path = str(Path(video_path).stem) + "_detections.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving to: {output_path}\n")

    # Skip frames if requested
    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        print(f"Skipping to frame {skip_frames}\n")

    # Statistics
    total_horses = 0
    total_persons = 0
    frames_with_horses = 0
    frames_with_persons = 0
    frame_num = skip_frames

    print("Processing... (Press 'q' to quit, 'p' to pause)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Max frames limit
        if max_frames > 0 and (frame_num - skip_frames) >= max_frames:
            break

        # Run detection
        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[HORSE_CLASS_ID, PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Get detections
        detections = results[0].boxes.data.cpu().numpy()

        # Draw on frame
        frame_drawn, horses, persons = draw_detections(frame, detections, frame_num)

        # Update stats
        if horses > 0:
            frames_with_horses += 1
            total_horses += horses
        if persons > 0:
            frames_with_persons += 1
            total_persons += persons

        # Print progress every 30 frames
        if frame_num % 30 == 0:
            print(f"Frame {frame_num}/{total_frames} | "
                  f"Horses: {horses} | Persons: {persons} | "
                  f"Detections: {len(detections)}")

        # Save frame
        if writer:
            writer.write(frame_drawn)

        # Show frame
        if show:
            cv2.imshow("Detection Test", frame_drawn)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(0)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # Print summary
    processed_frames = frame_num - skip_frames
    print(f"\n{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed frames: {processed_frames}")
    print(f"Frames with horses: {frames_with_horses} ({100*frames_with_horses/processed_frames:.1f}%)")
    print(f"Frames with persons: {frames_with_persons} ({100*frames_with_persons/processed_frames:.1f}%)")
    print(f"Total horse detections: {total_horses}")
    print(f"Total person detections: {total_persons}")
    print(f"Avg horses/frame: {total_horses/processed_frames:.2f}")
    print(f"Avg persons/frame: {total_persons/processed_frames:.2f}")
    print(f"{'='*60}\n")

    if save:
        print(f"Output saved to: {output_path}")


def main():
    global CONFIDENCE_THRESHOLD

    parser = argparse.ArgumentParser(description="Test YOLO Detection")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--skip", type=int, default=0, help="Skip N frames from start")
    parser.add_argument("--max", type=int, default=0, help="Process max N frames (0=all)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")

    args = parser.parse_args()
    CONFIDENCE_THRESHOLD = args.conf

    test_detection(
        video_path=args.video,
        show=args.show,
        save=args.save,
        skip_frames=args.skip,
        max_frames=args.max
    )


if __name__ == "__main__":
    main()
