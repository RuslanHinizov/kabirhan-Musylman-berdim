"""
Train YOLOv8 on Horse Racing dataset
Classes: body, helmet, horse, label, number
"""

import torch
from multiprocessing import freeze_support


def main():
    from ultralytics import YOLO

    print("=" * 60)
    print("YOLO HORSE RACING TRAINING")
    print("=" * 60)

    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load base model
    print("\nLoading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # Train
    print("\nStarting training...")
    results = model.train(
        data="data/Horse Racing level 2.v5i.yolo26/data_fixed.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,  # Windows fix
        patience=10,
        save=True,
        project="runs/horse_racing",
        name="yolov8n_horse",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best model: runs/horse_racing/yolov8n_horse/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    main()
