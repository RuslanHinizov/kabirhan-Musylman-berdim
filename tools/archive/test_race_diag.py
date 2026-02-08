"""
Race Tracker - DIAGNOSTIC VERSION

Full metrics for every detection:
- All softmax probabilities (not just top-1)
- Torso crop saved to disk
- HSV dominant hue/saturation analysis
- Detection confidence from YOLO
- Tracking ID stability
- Per-frame CSV log
- Voting system for stable classification

Usage:
  python tools/test_race_diag.py data/videos/exp10_cam2.mp4
  python tools/test_race_diag.py data/videos/exp10_cam2.mp4 --no-display
  python tools/test_race_diag.py data/videos/exp10_cam2.mp4 --save-crops
"""

import cv2
import sys
import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

# Output folder
SCRIPT_NAME = "race_diag"
RESULTS_DIR = "results"

from ultralytics import YOLO

# ============================================================
# PARAMETERS — tune these
# ============================================================
DETECTION_CONF = 0.35       # YOLO confidence (was 0.50 — too strict)
DETECTION_IOU = 0.3         # NMS IoU threshold
IMGSZ = 1280               # YOLO input resolution

# Torso extraction region (% of person bbox)
TORSO_TOP = 0.05            # Start from 5% (skip head top)
TORSO_BOTTOM = 0.45         # End at 45% (was 0.65 — too low, captures legs)
TORSO_LEFT = 0.15           # Inset 15% from left
TORSO_RIGHT = 0.15          # Inset 15% from right

# Classifier thresholds
MIN_COLOR_CONF = 0.50       # Accept predictions above this (was 0.75)
LOG_ALL_CONF = 0.20         # Log predictions above this for diagnostics

# Voting system
MIN_VOTES_TO_DECIDE = 5     # Minimum frames before deciding color for a track
VOTE_LOCK_RATIO = 0.60      # 60% of votes must agree to lock color

# Min crop size
MIN_CROP_PIXELS = 400       # Skip crops smaller than this (was 100 — too small)

COLORS_BGR = {
    "blue": (255, 100, 0),
    "green": (0, 200, 0),
    "purple": (180, 0, 180),
    "red": (0, 0, 255),
    "yellow": (0, 230, 230),
    "unknown": (128, 128, 128),
}


class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ColorClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("models/color_classifier.pt", map_location=self.device, weights_only=False)
        self.classes = ckpt['classes']
        self.model = SimpleColorCNN().to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_full(self, img_bgr):
        """Return ALL class probabilities, not just top-1."""
        if img_bgr is None or img_bgr.size < MIN_CROP_PIXELS:
            return None, 0.0, {}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.softmax(logits, dim=1)[0]

        # Build full probability dict
        prob_dict = {}
        for i, cls_name in enumerate(self.classes):
            prob_dict[cls_name] = round(probs[i].item(), 4)

        best_idx = probs.argmax().item()
        best_cls = self.classes[best_idx]
        best_conf = probs[best_idx].item()

        return best_cls, best_conf, prob_dict


def analyze_hsv(crop_bgr):
    """Analyze HSV statistics of a torso crop."""
    if crop_bgr is None or crop_bgr.size < 100:
        return {}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Filter out very dark or desaturated pixels
    mask = (s > 30) & (v > 40)
    if mask.sum() < 20:
        return {
            'dominant_hue': -1, 'mean_sat': float(s.mean()),
            'mean_val': float(v.mean()), 'colored_pct': 0.0,
            'hsv_color_guess': 'unknown'
        }

    h_filtered = h[mask]
    s_filtered = s[mask]
    v_filtered = v[mask]

    # Hue histogram (180 bins for OpenCV hue range)
    hist = cv2.calcHist([h_filtered.reshape(-1, 1)], [0], None, [18], [0, 180])
    dominant_bin = hist.argmax()
    dominant_hue = int(dominant_bin * 10 + 5)

    # Guess color from hue
    if dominant_hue < 10 or dominant_hue > 170:
        hsv_guess = 'red'
    elif 10 <= dominant_hue < 25:
        hsv_guess = 'yellow/orange'
    elif 25 <= dominant_hue < 40:
        hsv_guess = 'yellow'
    elif 40 <= dominant_hue < 85:
        hsv_guess = 'green'
    elif 85 <= dominant_hue < 130:
        hsv_guess = 'blue'
    elif 130 <= dominant_hue <= 170:
        hsv_guess = 'purple'
    else:
        hsv_guess = 'unknown'

    return {
        'dominant_hue': dominant_hue,
        'mean_sat': round(float(s_filtered.mean()), 1),
        'mean_val': round(float(v_filtered.mean()), 1),
        'colored_pct': round(float(mask.sum()) / mask.size * 100, 1),
        'hsv_color_guess': hsv_guess,
    }


class DiagnosticTracker:
    def __init__(self, output_dir, save_crops=False):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8s.pt")

        print("Loading classifier...")
        self.classifier = ColorClassifier()
        print(f"  Classes: {self.classifier.classes}")

        self.output_dir = Path(output_dir)
        self.save_crops = save_crops
        self.frame_num = 0

        # Voting system per track ID
        self.track_votes = defaultdict(lambda: defaultdict(int))  # {track_id: {color: count}}
        self.track_locked = {}  # {track_id: color}
        self.track_total_votes = defaultdict(int)

        # Results
        self.best_result = []
        self.final_result = []
        self.result_saved = False

        # Diagnostic log
        self.csv_path = self.output_dir / "detections_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame', 'track_id', 'det_conf',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'bbox_w', 'bbox_h', 'center_x',
            'crop_w', 'crop_h', 'crop_pixels',
            'pred_color', 'pred_conf',
            'p_blue', 'p_green', 'p_purple', 'p_red', 'p_yellow',
            'hsv_hue', 'hsv_sat', 'hsv_val', 'colored_pct', 'hsv_guess',
            'cnn_vs_hsv_agree',
            'voted_color', 'vote_count', 'vote_ratio', 'is_locked',
        ])

        # Stats
        self.total_detections = 0
        self.total_classified = 0
        self.low_conf_count = 0
        self.tiny_crop_count = 0
        self.cnn_hsv_agree_count = 0
        self.cnn_hsv_total = 0

        # Crop output dir
        if self.save_crops:
            self.crops_dir = self.output_dir / "crops"
            self.crops_dir.mkdir(exist_ok=True)

    def extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        ty1 = max(0, y1 + int(h * TORSO_TOP))
        ty2 = min(frame.shape[0], y1 + int(h * TORSO_BOTTOM))
        tx1 = max(0, x1 + int(w * TORSO_LEFT))
        tx2 = min(frame.shape[1], x2 - int(w * TORSO_RIGHT))
        if ty2 - ty1 < 10 or tx2 - tx1 < 10:
            return None
        return frame[ty1:ty2, tx1:tx2]

    def update(self, frame):
        self.frame_num += 1

        results = self.yolo.track(
            frame, imgsz=IMGSZ, conf=DETECTION_CONF, iou=DETECTION_IOU,
            classes=[0], tracker="botsort.yaml", persist=True,
            device="cuda:0", half=True, verbose=False
        )

        jockeys = []
        diag_entries = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return jockeys, diag_entries

        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(bboxes))

        for i, (bbox, det_conf, tid) in enumerate(zip(bboxes, confs, track_ids)):
            self.total_detections += 1
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            bw, bh = int(x2 - x1), int(y2 - y1)

            # Extract torso
            torso = self.extract_torso(frame, bbox)

            if torso is None:
                self.tiny_crop_count += 1
                continue

            crop_h, crop_w = torso.shape[:2]
            crop_pixels = crop_h * crop_w

            if crop_pixels < MIN_CROP_PIXELS:
                self.tiny_crop_count += 1
                continue

            # Classify with FULL probabilities
            pred_color, pred_conf, prob_dict = self.classifier.predict_full(torso)

            # HSV analysis
            hsv_info = analyze_hsv(torso)

            # CNN vs HSV agreement
            hsv_guess = hsv_info.get('hsv_color_guess', 'unknown')
            # Normalize HSV guess for comparison
            hsv_norm = hsv_guess.split('/')[0]  # "yellow/orange" -> "yellow"
            agree = pred_color == hsv_norm if pred_color and hsv_norm != 'unknown' else None
            if agree is not None:
                self.cnn_hsv_total += 1
                if agree:
                    self.cnn_hsv_agree_count += 1

            # Voting
            if pred_color and pred_conf >= LOG_ALL_CONF:
                self.track_votes[tid][pred_color] += 1
                self.track_total_votes[tid] += 1

            # Get voted color
            voted_color = None
            vote_count = 0
            vote_ratio = 0.0
            is_locked = tid in self.track_locked

            if is_locked:
                voted_color = self.track_locked[tid]
            elif self.track_total_votes[tid] >= MIN_VOTES_TO_DECIDE:
                # Find best voted color
                votes = self.track_votes[tid]
                best_vote_color = max(votes, key=votes.get)
                best_vote_count = votes[best_vote_color]
                total = self.track_total_votes[tid]
                ratio = best_vote_count / total

                if ratio >= VOTE_LOCK_RATIO:
                    self.track_locked[tid] = best_vote_color
                    voted_color = best_vote_color
                    is_locked = True
                    print(f"  [LOCK] Track {tid} -> {best_vote_color} "
                          f"({best_vote_count}/{total} = {ratio:.0%})")
                else:
                    voted_color = best_vote_color

                vote_count = best_vote_count
                vote_ratio = ratio

            # Save crop
            if self.save_crops and pred_conf >= LOG_ALL_CONF:
                crop_name = f"f{self.frame_num:04d}_t{tid}_{pred_color}_{pred_conf:.2f}.jpg"
                cv2.imwrite(str(self.crops_dir / crop_name), torso)

            # CSV row
            self.csv_writer.writerow([
                self.frame_num, tid, round(det_conf, 3),
                int(x1), int(y1), int(x2), int(y2),
                bw, bh, round(center_x, 1),
                crop_w, crop_h, crop_pixels,
                pred_color or '', round(pred_conf, 4),
                prob_dict.get('blue', 0), prob_dict.get('green', 0),
                prob_dict.get('purple', 0), prob_dict.get('red', 0),
                prob_dict.get('yellow', 0),
                hsv_info.get('dominant_hue', -1),
                hsv_info.get('mean_sat', 0),
                hsv_info.get('mean_val', 0),
                hsv_info.get('colored_pct', 0),
                hsv_guess,
                agree,
                voted_color or '', vote_count, round(vote_ratio, 3), is_locked,
            ])

            # Build entry for display
            effective_color = voted_color if is_locked else (pred_color if pred_conf >= MIN_COLOR_CONF else None)

            entry = {
                'bbox': tuple(map(int, bbox)),
                'center_x': center_x,
                'track_id': tid,
                'det_conf': det_conf,
                'pred_color': pred_color,
                'pred_conf': pred_conf,
                'prob_dict': prob_dict,
                'hsv_info': hsv_info,
                'voted_color': voted_color,
                'is_locked': is_locked,
                'effective_color': effective_color,
            }
            diag_entries.append(entry)

            if effective_color:
                self.total_classified += 1
                jockeys.append({
                    'bbox': tuple(map(int, bbox)),
                    'center_x': center_x,
                    'color': effective_color,
                    'conf': pred_conf,
                    'track_id': tid,
                })
            elif pred_conf < MIN_COLOR_CONF and pred_color:
                self.low_conf_count += 1

        # Update best result
        colors = set(j['color'] for j in jockeys)
        if len(colors) > len(self.best_result):
            jockeys_sorted = sorted(jockeys, key=lambda j: -j['center_x'])
            self.best_result = []
            seen = set()
            for j in jockeys_sorted:
                if j['color'] not in seen:
                    seen.add(j['color'])
                    self.best_result.append(j['color'])
            print(f"  [Frame {self.frame_num}] Found {len(colors)} colors: {self.best_result}")

        if len(colors) >= 5 and not self.result_saved:
            self.final_result = self.best_result.copy()
            self.result_saved = True
            print(f"\n*** ALL 5 COLORS FOUND at frame {self.frame_num} ***")

        return jockeys, diag_entries

    def print_summary(self):
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print(f"\n[PARAMETERS]")
        print(f"  DETECTION_CONF:    {DETECTION_CONF}")
        print(f"  MIN_COLOR_CONF:    {MIN_COLOR_CONF}")
        print(f"  TORSO region:      top={TORSO_TOP} bottom={TORSO_BOTTOM} L={TORSO_LEFT} R={TORSO_RIGHT}")
        print(f"  IMGSZ:             {IMGSZ}")
        print(f"  MIN_CROP_PIXELS:   {MIN_CROP_PIXELS}")
        print(f"  VOTE_LOCK_RATIO:   {VOTE_LOCK_RATIO}")
        print(f"  MIN_VOTES:         {MIN_VOTES_TO_DECIDE}")

        print(f"\n[DETECTION STATS]")
        print(f"  Total frames:      {self.frame_num}")
        print(f"  Total detections:  {self.total_detections}")
        print(f"  Classified:        {self.total_classified}")
        print(f"  Low conf rejected: {self.low_conf_count}")
        print(f"  Tiny crop skipped: {self.tiny_crop_count}")

        if self.cnn_hsv_total > 0:
            pct = self.cnn_hsv_agree_count / self.cnn_hsv_total * 100
            print(f"\n[CNN vs HSV AGREEMENT]")
            print(f"  Agree:  {self.cnn_hsv_agree_count}/{self.cnn_hsv_total} ({pct:.1f}%)")
            print(f"  -> Low agreement means CNN or HSV is confused")

        print(f"\n[TRACKING / VOTING]")
        print(f"  Tracks seen:       {len(self.track_votes)}")
        print(f"  Tracks locked:     {len(self.track_locked)}")
        for tid in sorted(self.track_locked):
            votes = self.track_votes[tid]
            total = self.track_total_votes[tid]
            print(f"    Track {tid}: {self.track_locked[tid]} "
                  f"(votes: {dict(votes)}, total={total})")

        # Show unlocked tracks
        unlocked = set(self.track_votes.keys()) - set(self.track_locked.keys())
        if unlocked:
            print(f"\n  UNLOCKED tracks (unstable classification):")
            for tid in sorted(unlocked):
                votes = dict(self.track_votes[tid])
                total = self.track_total_votes[tid]
                if total > 0:
                    best = max(votes, key=votes.get)
                    ratio = votes[best] / total
                    print(f"    Track {tid}: best={best} ({votes[best]}/{total}={ratio:.0%}) "
                          f"all_votes={votes}")

        # Confusion patterns: which colors get confused with each other
        print(f"\n[PER-TRACK VOTE DISTRIBUTION]")
        for tid in sorted(self.track_votes):
            votes = dict(self.track_votes[tid])
            total = self.track_total_votes[tid]
            if total >= 3:
                sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
                bars = " | ".join(f"{c}:{n}({n/total:.0%})" for c, n in sorted_votes)
                locked_str = " [LOCKED]" if tid in self.track_locked else ""
                print(f"    Track {tid:3d}: [{bars}]{locked_str}")

        print(f"\n[FILES]")
        print(f"  CSV log: {self.csv_path}")
        if self.save_crops:
            n_crops = len(list(self.crops_dir.glob("*.jpg")))
            print(f"  Crops:   {self.crops_dir} ({n_crops} files)")

    def close(self):
        self.csv_file.close()


def draw_diag(frame, jockeys, diag_entries, tracker):
    """Draw with full diagnostic info on each detection."""
    for entry in diag_entries:
        x1, y1, x2, y2 = entry['bbox']
        tid = entry['track_id']
        pred = entry['pred_color'] or '?'
        conf = entry['pred_conf']
        eff = entry['effective_color']
        locked = entry['is_locked']

        if eff:
            bgr = COLORS_BGR.get(eff, (128, 128, 128))
            thickness = 4 if locked else 2
        else:
            bgr = (80, 80, 80)
            thickness = 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)

        # Label: track_id | pred_color conf | voted
        label1 = f"T{tid} det:{entry['det_conf']:.2f}"
        label2 = f"CNN:{pred} {conf:.2f}"
        hsv_g = entry['hsv_info'].get('hsv_color_guess', '?')
        label3 = f"HSV:{hsv_g} h={entry['hsv_info'].get('dominant_hue', '?')}"

        # Show top-2 probabilities
        probs = entry['prob_dict']
        if probs:
            sorted_p = sorted(probs.items(), key=lambda x: -x[1])[:3]
            label4 = " ".join(f"{c[0]}:{p:.2f}" for c, p in sorted_p)
        else:
            label4 = ""

        vote_str = ""
        if entry['voted_color']:
            vote_str = f"VOTE:{entry['voted_color']}"
            if locked:
                vote_str += " LOCKED"

        y_text = y1 - 5
        for lbl in [vote_str, label4, label3, label2, label1]:
            if lbl:
                cv2.putText(frame, lbl, (x1, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
                y_text -= 18

        # Draw torso region
        h, w = y2 - y1, x2 - x1
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        tx1 = x1 + int(w * TORSO_LEFT)
        tx2 = x2 - int(w * TORSO_RIGHT)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)

    # Info panel
    cv2.rectangle(frame, (5, 5), (350, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {tracker.frame_num}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(diag_entries)}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Classified: {len(jockeys)}", (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Locked tracks: {len(tracker.track_locked)}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    n_colors = len(set(j['color'] for j in jockeys))
    cv2.putText(frame, f"Unique colors: {n_colors}/5", (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    status = "COMPLETE" if tracker.result_saved else "searching..."
    st_c = (0, 255, 0) if tracker.result_saved else (128, 128, 128)
    cv2.putText(frame, status, (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, st_c, 2)

    # Result panel
    result = tracker.final_result if tracker.final_result else tracker.best_result
    if result:
        y = 190
        for i, color in enumerate(result):
            bgr = COLORS_BGR.get(color, (128, 128, 128))
            cv2.putText(frame, f"{i+1}. {color}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            y += 30

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--save-crops", action="store_true", help="Save every torso crop to disk")
    parser.add_argument("--no-display", action="store_true", help="No GUI window")
    args = parser.parse_args()

    print("=" * 60)
    print("RACE TRACKER — DIAGNOSTIC MODE")
    print("Full metrics for every detection")
    print("=" * 60)

    video_name = Path(args.video).stem
    output_dir = Path(RESULTS_DIR) / SCRIPT_NAME / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    tracker = DiagnosticTracker(output_dir, save_crops=args.save_crops)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames\n")

    # Log parameters
    params = {
        'DETECTION_CONF': DETECTION_CONF,
        'DETECTION_IOU': DETECTION_IOU,
        'IMGSZ': IMGSZ,
        'MIN_COLOR_CONF': MIN_COLOR_CONF,
        'LOG_ALL_CONF': LOG_ALL_CONF,
        'TORSO_TOP': TORSO_TOP,
        'TORSO_BOTTOM': TORSO_BOTTOM,
        'TORSO_LEFT': TORSO_LEFT,
        'TORSO_RIGHT': TORSO_RIGHT,
        'MIN_CROP_PIXELS': MIN_CROP_PIXELS,
        'MIN_VOTES_TO_DECIDE': MIN_VOTES_TO_DECIDE,
        'VOTE_LOCK_RATIO': VOTE_LOCK_RATIO,
        'video': args.video,
        'resolution': f'{w}x{h}',
        'fps': fps,
        'total_frames': total,
    }
    with open(output_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(output_dir / f"{video_name}_diag.mp4")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        jockeys, diag_entries = tracker.update(frame)
        frame_draw = draw_diag(frame.copy(), jockeys, diag_entries, tracker)

        if writer:
            writer.write(frame_draw)

        if not args.no_display:
            disp = cv2.resize(frame_draw, (int(w * scale), int(h * scale)))
            cv2.namedWindow("Race DIAG", cv2.WINDOW_NORMAL)
            cv2.imshow("Race DIAG", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    tracker.print_summary()

    # Final result
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    result = tracker.final_result if tracker.final_result else tracker.best_result
    if result:
        for i, color in enumerate(result):
            print(f"  {i+1}. {color.upper()}")
        if len(result) < 5:
            print(f"\n  (Only {len(result)}/5 colors found)")

        txt_path = output_dir / "result.txt"
        with open(txt_path, "w") as f:
            for i, color in enumerate(result):
                f.write(f"{i+1}. {color.upper()}\n")
        print(f"\n  [SAVED] {txt_path}")
    else:
        print("  No jockeys detected")
    print("=" * 60)

    tracker.close()


if __name__ == "__main__":
    main()
