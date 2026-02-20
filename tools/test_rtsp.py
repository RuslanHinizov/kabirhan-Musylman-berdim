"""
RTSP Stream Test
Quick connectivity + frame grab test for a single RTSP camera.

Usage:
    python tools/test_rtsp.py
    python tools/test_rtsp.py --url "rtsp://admin:pass@ip:554/stream"
    python tools/test_rtsp.py --save-video 10       # record 10 seconds
    python tools/test_rtsp.py --no-display           # headless mode
    python tools/test_rtsp.py --gpu                  # GPU decode (h264_cuvid)
"""

import cv2
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Default RTSP URL — loaded from config (.env) if available
try:
    from api.config import DEFAULT_RTSP_URL as DEFAULT_URL
except ImportError:
    DEFAULT_URL = "rtsp://admin:admin@192.168.1.100:554/stream"

# Find ffmpeg: try imageio_ffmpeg first, then system PATH, then bundled path
def _find_ffmpeg() -> str:
    # 1. Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    # 2. Try system PATH (where ffmpeg)
    import shutil
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    # 3. Fallback to bundled path (may not exist)
    return str(Path(__file__).resolve().parent.parent /
               ".venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")

FFMPEG = _find_ffmpeg()

OUTPUT_DIR = "results/rtsp_test"


def detect_codec(url: str) -> str:
    """Probe stream to detect video codec (h264 or hevc)"""
    cmd = [
        FFMPEG, "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        output = result.stderr
        for line in output.split('\n'):
            if 'Video:' in line:
                lower = line.lower()
                if 'hevc' in lower or 'h265' in lower or 'h.265' in lower:
                    return "hevc"
                elif 'h264' in lower or 'h.264' in lower or 'avc' in lower:
                    return "h264"
    except Exception:
        pass
    return "h264"  # fallback


def test_ffmpeg_probe(url: str, gpu: bool = False, codec: str = "h264"):
    """Probe stream info using ffmpeg"""
    print(f"\n{'='*60}")
    print("RTSP STREAM PROBE (ffmpeg)")
    print(f"{'='*60}")
    print(f"URL: {url.replace(url.split('@')[0].split('//')[1], '***:***')}")

    cmd = [
        FFMPEG,
        "-rtsp_transport", "tcp",
    ]
    if gpu:
        cuvid = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"
        cmd += ["-hwaccel", "cuda", "-c:v", cuvid]
        print(f"  GPU decoder: {cuvid}")

    cmd += [
        "-i", url,
        "-frames:v", "1",
        "-f", "null", "-"
    ]

    print(f"\nProbing stream...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        output = result.stderr  # ffmpeg writes info to stderr

        # Parse stream info
        for line in output.split('\n'):
            line = line.strip()
            if 'Stream #' in line:
                print(f"  {line}")
            elif 'Duration' in line:
                print(f"  {line}")
            elif 'Input #' in line:
                print(f"  {line}")

        if result.returncode == 0:
            print("\n  Probe: OK")
            return True
        else:
            # Show last few error lines
            err_lines = [l for l in output.split('\n') if l.strip()]
            for l in err_lines[-3:]:
                print(f"  ERROR: {l.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print("  ERROR: Connection timeout (15s)")
        return False
    except FileNotFoundError:
        print(f"  ERROR: ffmpeg not found at {FFMPEG}")
        return False


class FFmpegReader:
    """Read frames from RTSP via ffmpeg subprocess (handles HEVC properly)"""

    def __init__(self, url: str, width: int, height: int, gpu: bool = False,
                 codec: str = "h264", cam_type: str = "analytics"):
        self.url = url
        self.width = width
        self.height = height
        self.process = None

        cmd = [FFMPEG, "-rtsp_transport", "tcp"]

        if cam_type == "ptz":
            # PTZ: minimize buffering for lowest possible latency
            cmd += [
                "-fflags", "nobuffer+genpts+discardcorrupt",
                "-flags", "low_delay",
                "-analyzeduration", "500000",
                "-probesize", "1048576",
                "-max_delay", "0",
                "-reorder_queue_size", "0",
                "-vsync", "0",
                "-err_detect", "ignore_err",
            ]
        else:
            # Analytics: prefer stable decode over ultra-low latency
            cmd += [
                "-fflags", "+genpts+discardcorrupt",
                "-flags", "low_delay",
                "-vsync", "0",
            ]
        if gpu:
            cuvid = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"
            cmd += ["-hwaccel", "cuda", "-c:v", cuvid]

        cmd += [
            "-i", url,
            # Downscale before raw pipe output to keep multi-camera mode stable.
            "-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-"
        ]
        self.cmd = cmd
        self.frame_size = width * height * 3
        self._is_ptz = (cam_type == "ptz")

    def start(self):
        # PTZ: minimal pipe buffer to avoid stale frames accumulating
        # Analytics: larger buffer for stable decode
        buf = self.frame_size if self._is_ptz else self.frame_size * 2
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=buf,
        )

    def read(self):
        if self.process is None:
            return False, None
        raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    def release(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def parse_resolution(url: str) -> tuple:
    """Get stream resolution from direct frame decode, then probe, then OpenCV fallback."""
    import re

    # Method 0: decode one JPEG frame from ffmpeg image2pipe (most reliable)
    cmd = [
        FFMPEG, "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-an",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.stdout:
            arr = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                if w > 0 and h > 0:
                    print(f"  [parse_resolution] frame decode detected: {w}x{h}")
                    return w, h
    except subprocess.TimeoutExpired:
        print(f"  [parse_resolution] frame decode timed out (20s)")
    except Exception as e:
        print(f"  [parse_resolution] frame decode error: {e}")

    # Method 1: ffmpeg probe
    cmd = [
        FFMPEG, "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1", "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        for line in result.stderr.split('\n'):
            if 'Video:' in line:
                match = re.search(r'(\d{3,5})x(\d{3,5})', line)
                if match:
                    return int(match.group(1)), int(match.group(2))
        # Log why it failed
        err_lines = [l.strip() for l in result.stderr.split('\n') if l.strip()]
        if err_lines:
            print(f"  [parse_resolution] ffmpeg probe output (last 3 lines):")
            for l in err_lines[-3:]:
                print(f"    {l}")
    except subprocess.TimeoutExpired:
        print(f"  [parse_resolution] ffmpeg probe timed out (20s)")
    except FileNotFoundError:
        print(f"  [parse_resolution] ffmpeg not found at {FFMPEG}")
    except Exception as e:
        print(f"  [parse_resolution] ffmpeg probe error: {e}")

    # Method 2: OpenCV fallback (more compatible with various RTSP servers)
    print(f"  [parse_resolution] Trying OpenCV fallback...")
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        # Set TCP transport for OpenCV
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)

        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                print(f"  [parse_resolution] OpenCV detected: {w}x{h}")
                cap.release()
                return w, h

            # Try reading a frame to get resolution
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [parse_resolution] OpenCV frame read: {w}x{h}")
                cap.release()
                return w, h

        cap.release()
    except Exception as e:
        print(f"  [parse_resolution] OpenCV fallback error: {e}")

    # Method 3: Try common resolutions as last resort
    # Return 0,0 so caller knows it failed
    print(f"  [parse_resolution] All methods failed for URL")
    return 0, 0


def test_ffmpeg_reader(url: str, gpu: bool = False, codec: str = "h264") -> dict:
    """Test connection by reading frames through ffmpeg pipe"""
    print(f"\n{'='*60}")
    print("RTSP FRAME READ (ffmpeg pipe)")
    print(f"{'='*60}")

    info = {}

    # Get resolution first
    print("  Getting resolution...", end=" ", flush=True)
    w, h = parse_resolution(url)
    if w == 0:
        print("FAILED (could not detect resolution)")
        return info
    print(f"{w}x{h}")

    # Start reader
    mode = f"GPU ({('hevc_cuvid' if codec == 'hevc' else 'h264_cuvid')})" if gpu else "CPU"
    print(f"  Decode: {mode}")
    print(f"  Connecting...", end=" ", flush=True)

    reader = FFmpegReader(url, w, h, gpu=gpu, codec=codec)
    reader.start()

    t0 = time.time()
    ret, frame = reader.read()
    connect_time = time.time() - t0

    if not ret:
        print(f"FAILED ({connect_time:.1f}s)")
        reader.release()
        return info

    print(f"OK ({connect_time:.1f}s)")

    fps_reported = 30  # from probe
    info = {"reader": reader, "frame": frame, "width": w, "height": h, "fps": fps_reported}

    # Read 60 frames to measure throughput
    print(f"\n  Reading 60 frames...", end=" ", flush=True)
    t0 = time.time()
    good = 0
    for _ in range(60):
        ret, f = reader.read()
        if ret and f is not None:
            good += 1
            info["frame"] = f
    elapsed = time.time() - t0
    actual_fps = good / elapsed if elapsed > 0 else 0

    print(f"{good}/60 OK, {actual_fps:.1f} fps, {elapsed/max(good,1)*1000:.0f}ms/frame")

    return info


def save_screenshot(frame, output_dir: str):
    """Save a single frame as screenshot"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(Path(output_dir) / f"rtsp_test_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  Screenshot saved: {path}")
    return path


def record_clip(url: str, seconds: int, output_dir: str, gpu: bool = False, codec: str = "h264"):
    """Record a short clip using ffmpeg"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = str(Path(output_dir) / f"rtsp_clip_{ts}.mp4")

    print(f"\n{'='*60}")
    print(f"RECORDING {seconds}s CLIP")
    print(f"{'='*60}")
    print(f"Output: {output_file}")

    cmd = [
        FFMPEG, "-y",
        "-rtsp_transport", "tcp",
    ]

    if gpu:
        # GPU decode + GPU encode
        cuvid = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"
        cmd += [
            "-hwaccel", "cuda",
            "-c:v", cuvid,
            "-i", url,
            "-t", str(seconds),
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
        ]
        print(f"Decode: {cuvid}, Encode: h264_nvenc")
    else:
        # Copy stream (no re-encode, fastest)
        cmd += [
            "-i", url,
            "-t", str(seconds),
            "-c:v", "copy",
        ]

    cmd += [
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_file
    ]

    print(f"Mode: {'GPU (cuvid→nvenc)' if gpu else 'copy (no re-encode)'}")
    print(f"Recording...", end=" ", flush=True)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 30)
        elapsed = time.time() - t0

        if result.returncode == 0:
            size_mb = Path(output_file).stat().st_size / 1024 / 1024
            print(f"OK ({elapsed:.1f}s, {size_mb:.1f} MB)")
            print(f"  File: {output_file}")
        else:
            print(f"FAILED (code {result.returncode})")
            err_lines = result.stderr.strip().split('\n')
            for l in err_lines[-3:]:
                print(f"  {l.strip()}")
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({seconds + 30}s)")


def live_preview(info: dict, output_dir: str, duration: int = 0):
    """Show live preview window using FFmpegReader"""
    reader = info["reader"]

    print(f"\n{'='*60}")
    print("LIVE PREVIEW (press 'q'/ESC to quit, 's' for screenshot)")
    print(f"{'='*60}")

    cv2.namedWindow("RTSP Test", cv2.WINDOW_NORMAL)
    w, h = info["width"], info["height"]
    if w > 1920:
        cv2.resizeWindow("RTSP Test", w // 2, h // 2)
    else:
        cv2.resizeWindow("RTSP Test", w, h)

    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            print("Stream ended or lost connection")
            break

        frame_count += 1
        elapsed = time.time() - t0
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        # OSD
        cv2.putText(frame, f"FPS: {actual_fps:.1f} | Frame: {frame_count}",
                     (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("RTSP Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_screenshot(frame, output_dir)

        if duration > 0 and elapsed >= duration:
            break

    cv2.destroyAllWindows()
    reader.release()
    print(f"  Frames shown: {frame_count}, avg FPS: {actual_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="RTSP Stream Test")
    parser.add_argument("--url", default=DEFAULT_URL, help="RTSP URL")
    parser.add_argument("--no-display", action="store_true", help="No preview window")
    parser.add_argument("--save-video", type=int, default=0,
                        metavar="SEC", help="Record N seconds clip")
    parser.add_argument("--gpu", action="store_true", help="Use GPU decode/encode (h264_cuvid/nvenc)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  RTSP STREAM TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # Step 0: detect codec
    print("\nDetecting stream codec...", end=" ", flush=True)
    codec = detect_codec(args.url)
    print(f"{codec.upper()}")

    # Step 1: ffmpeg probe
    probe_ok = test_ffmpeg_probe(args.url, gpu=args.gpu, codec=codec)

    if not probe_ok:
        print("\n  RESULT: Stream unreachable")
        return 1

    # Step 2: ffmpeg pipe reader test
    info = test_ffmpeg_reader(args.url, gpu=args.gpu, codec=codec)

    if not info:
        print("\n  RESULT: ffmpeg probe OK but frame read failed")
        return 1

    # Save screenshot
    save_screenshot(info["frame"], args.output)

    # Step 3: Record clip if requested (uses separate ffmpeg process)
    if args.save_video > 0:
        record_clip(args.url, args.save_video, args.output, gpu=args.gpu, codec=codec)

    # Step 4: Live preview
    if not args.no_display:
        live_preview(info, args.output, duration=0)
    else:
        info["reader"].release()

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
