"""
Camera Simulation for Scale Testing

Simulates N RTSP cameras by serving local video files over the built-in
FastAPI server's video mode, or by creating dummy camera entries that
replay video files in loop.

Usage:
    # Start server with simulated 25 cameras from 3 video files
    python tools/simulate_cameras.py --count 25 --videos video/exp10_cam1.mp4 video/exp10_cam2.mp4 video/exp10_cam3.mp4

    # Start server with 10 cameras, auto-start race
    python tools/simulate_cameras.py --count 10 --videos video/exp10_cam1.mp4 --auto-start
"""

import sys
import argparse
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def start_simulated_cameras(server_url: str, count: int, video_files: list[str]):
    """Configure N simulated cameras on the running server."""
    print(f"Configuring {count} simulated cameras on {server_url}...")

    for i in range(1, count + 1):
        cam_id = f"analytics-{i}"
        # Round-robin through provided video files
        video = video_files[(i - 1) % len(video_files)]
        video_path = str(Path(video).resolve())

        # Use the camera update API to set "URL" (video path used in local mode)
        try:
            resp = requests.put(
                f"{server_url}/api/cameras/{cam_id}",
                json={"rtspUrl": video_path},
                timeout=5,
            )
            if resp.status_code == 200:
                print(f"  {cam_id}: configured with {video}")
            else:
                print(f"  {cam_id}: FAILED ({resp.status_code})")
        except Exception as e:
            print(f"  {cam_id}: ERROR - {e}")

    print(f"\n{count} cameras configured. Starting all...")

    try:
        resp = requests.post(f"{server_url}/api/cameras/start-all", timeout=10)
        if resp.status_code == 200:
            print("All cameras started.")
        else:
            print(f"Start-all failed: {resp.status_code}")
    except Exception as e:
        print(f"Start-all error: {e}")


def monitor_metrics(server_url: str, duration_sec: int = 60, interval: float = 5.0):
    """Monitor server metrics during scale test."""
    print(f"\nMonitoring metrics for {duration_sec}s (interval: {interval}s)...")
    print(f"{'Time':>6s}  {'Det FPS':>8s}  {'Cycle ms':>9s}  {'Cams':>5s}  {'GPU MB':>7s}  {'WebRTC':>7s}")
    print("-" * 50)

    start = time.time()
    while time.time() - start < duration_sec:
        try:
            resp = requests.get(f"{server_url}/api/system/metrics", timeout=5)
            if resp.status_code == 200:
                m = resp.json()
                det = m.get("detection", {})
                gpu = m.get("gpu", {})
                bc = m.get("broadcast", {})
                cams = m.get("cameras", {})

                elapsed = time.time() - start
                print(f"{elapsed:6.0f}s  "
                      f"{det.get('fps', 0):8.1f}  "
                      f"{det.get('last_cycle_ms', 0):9.1f}  "
                      f"{cams.get('analytics_running', 0):5d}  "
                      f"{gpu.get('memory_allocated_mb', 0):7.0f}  "
                      f"{bc.get('webrtc_connections', 0):7d}")
        except Exception:
            pass

        time.sleep(interval)

    print("\nMonitoring complete.")


def main():
    parser = argparse.ArgumentParser(description="Simulate cameras for scale testing")
    parser.add_argument("--count", type=int, default=25,
                        help="Number of cameras to simulate (default: 25)")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Video file(s) for simulation")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--auto-start", action="store_true",
                        help="Auto-start race after configuring cameras")
    parser.add_argument("--monitor", type=int, default=0,
                        help="Monitor duration in seconds (0=skip)")
    args = parser.parse_args()

    # Verify video files exist
    for v in args.videos:
        if not Path(v).exists():
            print(f"ERROR: Video file not found: {v}")
            sys.exit(1)

    # Wait for server
    print(f"Checking server at {args.server}...")
    for attempt in range(10):
        try:
            resp = requests.get(f"{args.server}/api/system/health", timeout=3)
            if resp.status_code == 200:
                print("Server is running.")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server not reachable after 20s")
        sys.exit(1)

    start_simulated_cameras(args.server, args.count, args.videos)

    if args.monitor > 0:
        monitor_metrics(args.server, args.monitor)


if __name__ == "__main__":
    main()
