"""
Benchmark detection pipeline speed.

Measures YOLO inference time, color classification time, and total cycle time.
Can compare PyTorch (.pt) vs TensorRT (.engine) models.

Usage:
    python tools/benchmark_detection.py --video video/exp10_cam1.mp4
    python tools/benchmark_detection.py --video video/exp10_cam1.mp4 --frames 100
    python tools/benchmark_detection.py --video video/exp10_cam1.mp4 --compare yolov8s.pt yolov8s.engine
"""

import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def benchmark_model(model_path: str, frames: list[np.ndarray], imgsz: int = 1280):
    """Run benchmark on a single model and return timing stats."""
    import torch

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    _dev = 0 if torch.cuda.is_available() else "cpu"
    _half = torch.cuda.is_available()

    # Warmup (3 frames)
    for f in frames[:3]:
        model(f, imgsz=imgsz, conf=0.35, iou=0.3, classes=[0],
              device=_dev, half=_half, verbose=False)

    # Benchmark
    times = []
    det_counts = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        results = model(frame, imgsz=imgsz, conf=0.35, iou=0.3, classes=[0],
                        device=_dev, half=_half, verbose=False)
        dt = (time.perf_counter() - t0) * 1000  # ms
        times.append(dt)

        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        det_counts.append(n_det)

        if (i + 1) % 20 == 0:
            avg = sum(times[-20:]) / 20
            print(f"  Frame {i+1}/{len(frames)}: avg {avg:.1f}ms, detections: {n_det}")

    times_arr = np.array(times)
    print(f"\nResults ({model_path}):")
    print(f"  Frames:    {len(frames)}")
    print(f"  Mean:      {times_arr.mean():.1f} ms")
    print(f"  Median:    {np.median(times_arr):.1f} ms")
    print(f"  Std:       {times_arr.std():.1f} ms")
    print(f"  Min:       {times_arr.min():.1f} ms")
    print(f"  Max:       {times_arr.max():.1f} ms")
    print(f"  P95:       {np.percentile(times_arr, 95):.1f} ms")
    print(f"  P99:       {np.percentile(times_arr, 99):.1f} ms")
    print(f"  FPS:       {1000.0 / times_arr.mean():.1f}")
    print(f"  Avg dets:  {np.mean(det_counts):.1f}")

    return {
        "model": model_path,
        "mean_ms": round(times_arr.mean(), 1),
        "median_ms": round(float(np.median(times_arr)), 1),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 1),
        "fps": round(1000.0 / times_arr.mean(), 1),
    }


def load_frames(video_path: str, max_frames: int = 50) -> list[np.ndarray]:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"ERROR: No frames read from {video_path}")
        sys.exit(1)

    print(f"Loaded {len(frames)} frames from {video_path} "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection pipeline")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file for benchmarking")
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of frames to benchmark (default: 50)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO input image size (default: 1280)")
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="Model paths to compare (e.g., yolov8s.pt yolov8s.engine)")
    args = parser.parse_args()

    frames = load_frames(args.video, args.frames)

    if args.compare:
        models = args.compare
    else:
        try:
            from api.config import MODEL_PATH_YOLO
            models = [MODEL_PATH_YOLO]
        except ImportError:
            models = ["yolov8s.pt"]

    results = []
    for model_path in models:
        if not Path(model_path).exists():
            print(f"WARNING: Model not found, skipping: {model_path}")
            continue
        stats = benchmark_model(model_path, frames, args.imgsz)
        results.append(stats)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['model']:30s}  mean={r['mean_ms']:6.1f}ms  "
                  f"p95={r['p95_ms']:6.1f}ms  fps={r['fps']:5.1f}")

        speedup = results[0]["mean_ms"] / results[1]["mean_ms"]
        print(f"\n  Speedup: {speedup:.1f}x "
              f"({'faster' if speedup > 1 else 'slower'} with {results[1]['model']})")


if __name__ == "__main__":
    main()
