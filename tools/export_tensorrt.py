"""
Export YOLOv8s to TensorRT engine format for 3-5x faster inference.

The exported .engine file is GPU-specific — it only works on the same
GPU model and TensorRT version it was built on.

Usage:
    python tools/export_tensorrt.py
    python tools/export_tensorrt.py --model yolov8s.pt --imgsz 1280
    python tools/export_tensorrt.py --model yolov8s.pt --imgsz 640   # smaller, faster

After export, update .env:
    MODEL_PATH_YOLO=yolov8s.engine
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO .pt model (default: from config or yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Input image size (default: 1280)")
    parser.add_argument("--half", action="store_true", default=True,
                        help="Use FP16 (default: True)")
    parser.add_argument("--workspace", type=int, default=4,
                        help="TensorRT workspace size in GB (default: 4)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (default: 1)")
    args = parser.parse_args()

    # Determine model path
    model_path = args.model
    if model_path is None:
        try:
            from api.config import MODEL_PATH_YOLO
            if MODEL_PATH_YOLO.endswith(".pt"):
                model_path = MODEL_PATH_YOLO
            else:
                model_path = "yolov8s.pt"
        except ImportError:
            model_path = "yolov8s.pt"

    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. TensorRT export requires a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_path}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.half}")
    print(f"Workspace: {args.workspace} GB")
    print(f"Batch size: {args.batch}")
    print()

    model = YOLO(model_path)

    print("Starting TensorRT export (this may take several minutes)...")
    t0 = time.time()

    engine_path = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=0,
        workspace=args.workspace,
        batch=args.batch,
    )

    elapsed = time.time() - t0
    print(f"\nExport complete in {elapsed:.1f}s")
    print(f"Engine file: {engine_path}")
    print(f"\nTo use this engine, update your .env file:")
    print(f"  MODEL_PATH_YOLO={engine_path}")


if __name__ == "__main__":
    main()
