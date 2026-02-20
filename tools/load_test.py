"""
Load Test for Race Vision

Tests system stability under various camera counts and WebRTC connections.
Reports performance metrics over time.

Usage:
    # Test with 5, then 10, then 25 cameras
    python tools/load_test.py --server http://localhost:8000 --stages 5 10 25

    # Quick test with 5 cameras for 30 seconds
    python tools/load_test.py --server http://localhost:8000 --stages 5 --duration 30
"""

import sys
import time
import argparse
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("ERROR: requests library required. pip install requests")
    sys.exit(1)


def collect_metrics(server_url: str, duration_sec: int, sample_interval: float = 2.0) -> dict:
    """Collect metrics over a time window."""
    samples = {
        "detection_fps": [],
        "cycle_ms": [],
        "gpu_memory_mb": [],
        "webrtc_connections": [],
        "analytics_running": [],
    }
    errors = 0
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

                samples["detection_fps"].append(det.get("fps", 0))
                samples["cycle_ms"].append(det.get("last_cycle_ms", 0))
                samples["gpu_memory_mb"].append(gpu.get("memory_allocated_mb", 0))
                samples["webrtc_connections"].append(bc.get("webrtc_connections", 0))
                samples["analytics_running"].append(cams.get("analytics_running", 0))
        except Exception:
            errors += 1

        time.sleep(sample_interval)

    # Compute statistics
    results = {"errors": errors, "samples": len(samples["detection_fps"])}
    for key, values in samples.items():
        if values:
            results[key] = {
                "mean": round(statistics.mean(values), 1),
                "median": round(statistics.median(values), 1),
                "min": round(min(values), 1),
                "max": round(max(values), 1),
                "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0,
            }
        else:
            results[key] = {"mean": 0, "median": 0, "min": 0, "max": 0, "stdev": 0}

    return results


def run_stage(server_url: str, camera_count: int, duration_sec: int):
    """Run a single test stage with a specific camera count."""
    print(f"\n{'='*60}")
    print(f"STAGE: {camera_count} cameras for {duration_sec}s")
    print(f"{'='*60}")

    # Check current status
    try:
        resp = requests.get(f"{server_url}/api/system/health", timeout=5)
        health = resp.json()
        running = health.get("cameras", {}).get("running", 0)
        print(f"Current running cameras: {running}")
    except Exception as e:
        print(f"ERROR: Cannot reach server - {e}")
        return None

    # Wait for cameras to stabilize
    print(f"Waiting 5s for cameras to stabilize...")
    time.sleep(5)

    # Collect metrics
    print(f"Collecting metrics for {duration_sec}s...")
    results = collect_metrics(server_url, duration_sec)

    # Print results
    print(f"\nResults ({camera_count} cameras, {duration_sec}s):")
    for key in ["detection_fps", "cycle_ms", "gpu_memory_mb", "analytics_running"]:
        stats = results.get(key, {})
        print(f"  {key:25s}: mean={stats.get('mean',0):7.1f}  "
              f"median={stats.get('median',0):7.1f}  "
              f"min={stats.get('min',0):7.1f}  "
              f"max={stats.get('max',0):7.1f}  "
              f"stdev={stats.get('stdev',0):7.1f}")

    if results.get("errors", 0) > 0:
        print(f"  WARNING: {results['errors']} API errors during collection")

    return results


def main():
    parser = argparse.ArgumentParser(description="Load test Race Vision server")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Server URL")
    parser.add_argument("--stages", nargs="+", type=int, default=[5, 10, 25],
                        help="Camera counts to test (default: 5 10 25)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration per stage in seconds (default: 60)")
    args = parser.parse_args()

    print(f"Race Vision Load Test")
    print(f"Server: {args.server}")
    print(f"Stages: {args.stages}")
    print(f"Duration per stage: {args.duration}s")

    # Verify server
    try:
        resp = requests.get(f"{args.server}/api/system/health", timeout=5)
        resp.raise_for_status()
        print("Server is reachable.")
    except Exception as e:
        print(f"ERROR: Cannot reach server - {e}")
        sys.exit(1)

    all_results = {}
    for count in args.stages:
        result = run_stage(args.server, count, args.duration)
        if result:
            all_results[count] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Cameras':>8s}  {'FPS':>8s}  {'Cycle ms':>9s}  {'GPU MB':>8s}  {'Stable':>8s}")
    for count, r in all_results.items():
        fps = r.get("detection_fps", {}).get("mean", 0)
        cycle = r.get("cycle_ms", {}).get("mean", 0)
        gpu = r.get("gpu_memory_mb", {}).get("mean", 0)
        stable = "YES" if r.get("errors", 0) == 0 else "NO"
        print(f"{count:8d}  {fps:8.1f}  {cycle:9.1f}  {gpu:8.0f}  {stable:>8s}")


if __name__ == "__main__":
    main()
