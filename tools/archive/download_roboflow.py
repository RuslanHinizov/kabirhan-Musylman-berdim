"""
Download Horse Racing dataset from Roboflow and train YOLOv8
"""

from roboflow import Roboflow
import os

# You need to get API key from https://app.roboflow.com/settings/api
# Click on your workspace -> Settings -> API Key (Private API Key)

API_KEY = os.environ.get("ROBOFLOW_API_KEY", None)

if API_KEY is None:
    print("=" * 60)
    print("ROBOFLOW API KEY REQUIRED")
    print("=" * 60)
    print()
    print("1. Go to: https://app.roboflow.com/")
    print("2. Create account or login")
    print("3. Go to Settings -> API Key")
    print("4. Copy your Private API Key")
    print()
    print("Then run:")
    print('  set ROBOFLOW_API_KEY=your_key_here')
    print('  python tools/download_roboflow.py')
    print()
    print("Or edit this file and paste your key directly")
    print("=" * 60)

    # Allow manual input
    key = input("\nPaste API key here (or press Enter to exit): ").strip()
    if not key:
        exit(1)
    API_KEY = key

print(f"Using API key: {API_KEY[:8]}...")

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)

# Download the horse racing dataset
print("\nDownloading Horse Racing Level 2 dataset...")
project = rf.workspace("new-workspace-vyhrr").project("horse-racing-level-2")

# Get dataset info
print(f"Project: {project.name}")
print(f"Type: {project.type}")

# Download in YOLOv8 format
dataset = project.version(1).download("yolov8", location="data/horse_racing_dataset")

print("\nDataset downloaded to: data/horse_racing_dataset")
print("Ready for training!")
