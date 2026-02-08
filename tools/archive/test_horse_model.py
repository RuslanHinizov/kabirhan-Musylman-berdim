"""
Test trained Horse Racing model
Classes: body, helmet, horse, label, number
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Class names and colors
CLASS_NAMES = ['body', 'helmet', 'horse', 'label', 'number']
CLASS_COLORS = {
    0: (0, 255, 0),     # body - green
    1: (255, 0, 255),   # helmet - magenta
    2: (255, 165, 0),   # horse - orange
    3: (255, 255, 0),   # label - cyan
    4: (0, 0, 255),     # number - red
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default="runs/horse_racing/yolov8n_horse/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("HORSE RACING MODEL TEST")
    print(f"Model: {args.model}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 60)

    # Load model
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames")

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(Path(args.video).stem) + "_horse_det.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"Saving to: {out}")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect
        results = model.predict(
            frame,
            imgsz=1280,
            conf=args.conf,
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Draw detections
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)

            color = CLASS_COLORS.get(cls, (255, 255, 255))
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info
        cv2.putText(frame, f"Frame {frame_num}/{total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Legend
        y = 60
        for cls, name in enumerate(CLASS_NAMES):
            color = CLASS_COLORS[cls]
            cv2.putText(frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25

        if writer:
            writer.write(frame)

        try:
            disp = cv2.resize(frame, (int(w*scale), int(h*scale)))
            cv2.imshow("Horse Racing Detection", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        except cv2.error:
            pass  # No display available

        if frame_num % 100 == 0:
            boxes = results[0].boxes
            counts = {}
            if boxes is not None and len(boxes) > 0:
                for cls in boxes.cls.cpu().numpy():
                    name = CLASS_NAMES[int(cls)]
                    counts[name] = counts.get(name, 0) + 1
            print(f"Frame {frame_num}/{total} | Detections: {counts}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\nDone!")


if __name__ == "__main__":
    main()
