#!/bin/bash
set -e

# Export TensorRT engine on first run (if .env specifies .engine but file doesn't exist)
if [ -f ".env" ]; then
    ENGINE_PATH=$(grep MODEL_PATH_YOLO .env | cut -d= -f2 | tr -d ' ')
    if [[ "$ENGINE_PATH" == *.engine ]] && [ ! -f "$ENGINE_PATH" ]; then
        echo "TensorRT engine not found, exporting from yolov8s.pt..."
        python tools/export_tensorrt.py --model yolov8s.pt
        echo "Export complete."
    fi
fi

# Start the server
exec python -m api.server "$@"
