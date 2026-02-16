"""
Race Tracker - Count Based v2

Position-based voting (no tracking dependency):
1. Detect jockeys in each frame
2. Sort by X position → position 1st, 2nd, 3rd...
3. Classify color of each
4. Vote per POSITION across many frames
5. Final result = majority color per position

Diagnostics:
- Full softmax probabilities in CSV
- Per-frame and per-position vote log
- Torso crop saving (--save-crops)
- HSV cross-check
"""

import cv2
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPT_NAME = "race_count"
RESULTS_DIR = "results"

from ultralytics import YOLO

# ============================================================
# PARAMETERS
# ============================================================
# Detection
DETECTION_CONF = 0.35
DETECTION_IOU = 0.3
IMGSZ = 1280

# Torso extraction (% of person bbox)
TORSO_TOP = 0.10            # Skip more of the head/helmet
TORSO_BOTTOM = 0.40         # Stop before horse blanket area
TORSO_LEFT = 0.20           # Tighter sides (less background/horse)
TORSO_RIGHT = 0.20

# Bbox filters (reject horse heads / blankets / wide detections)
MIN_ASPECT_RATIO = 1.2      # h/w — jockeys are tall+narrow (~2.0-3.5), horse blankets < 1.0
MIN_BBOX_HEIGHT = 100        # Skip very small detections (pixels)
EDGE_MARGIN = 10             # Reject bboxes touching frame edge (horse heads entering frame)

# Classifier
MIN_COLOR_CONF = 0.60       # Accept top-1 prediction above this
MIN_REASSIGN_CONF = 0.20    # Accept reassigned (2nd/3rd choice) above this
MIN_CROP_PIXELS = 400       # Skip tiny crops
MAX_CROP_PIXELS = 15000     # Skip huge crops (horse+rider, not just jockey torso)

# Position voting — weighted by frame completeness
MIN_JOCKEYS_FOR_VOTE = 3    # Only vote on frames with >= N unique colors
MIN_VOTES_PER_POS = 5       # Minimum weighted votes before deciding a position
REQUIRED_COLORS = 5         # Expected number of jockeys

# Weight: more jockeys visible = more reliable ordering
# Frame with 5/5 colors: weight 5 (most reliable — true order)
# Frame with 4/5 colors: weight 2
# Frame with 3/5 colors: weight 1 (least reliable — gaps in order)
VOTE_WEIGHTS = {5: 5, 4: 2, 3: 1}

# Uniqueness: in each frame, resolve duplicate colors by confidence
ENFORCE_UNIQUE_PER_FRAME = True

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
        """Return prediction + ALL class probabilities."""
        if img_bgr is None or img_bgr.size < MIN_CROP_PIXELS:
            return None, 0.0, {}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(t), dim=1)[0]
        prob_dict = {self.classes[i]: round(probs[i].item(), 4) for i in range(len(self.classes))}
        best_idx = probs.argmax().item()
        return self.classes[best_idx], probs[best_idx].item(), prob_dict


def analyze_hsv(crop_bgr):
    """Quick HSV analysis for cross-checking CNN."""
    if crop_bgr is None or crop_bgr.size < 100:
        return 'unknown', -1
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (s > 30) & (v > 40)
    if mask.sum() < 20:
        return 'unknown', -1
    h_filtered = h[mask]
    hist = cv2.calcHist([h_filtered.reshape(-1, 1)], [0], None, [18], [0, 180])
    dominant_hue = int(hist.argmax() * 10 + 5)
    if dominant_hue < 10 or dominant_hue > 170:
        return 'red', dominant_hue
    elif 20 <= dominant_hue < 40:
        return 'yellow', dominant_hue
    elif 40 <= dominant_hue < 85:
        return 'green', dominant_hue
    elif 85 <= dominant_hue < 130:
        return 'blue', dominant_hue
    elif 130 <= dominant_hue <= 170:
        return 'purple', dominant_hue
    return 'unknown', dominant_hue


class RaceTracker:
    def __init__(self, output_dir, save_crops=False):
        print("Loading YOLO...")
        self.yolo = YOLO("yolov8s.pt")
        print("Loading classifier...")
        self.classifier = ColorClassifier()
        print(f"  Classes: {self.classifier.classes}")

        self.output_dir = Path(output_dir)
        self.save_crops = save_crops
        self.frame_num = 0

        # Position-based voting: position_votes[pos_index] = Counter({color: count})
        self.position_votes = defaultdict(Counter)
        self.vote_frames = 0  # How many frames contributed to voting

        self.best_result = []
        self.final_result = []
        self.result_saved = False
        self.save_frame = None

        # Diagnostics
        self.csv_path = self.output_dir / "detections_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'frame', 'det_idx', 'det_conf',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'center_x', 'position',
            'crop_w', 'crop_h',
            'pred_color', 'pred_conf',
            'p_blue', 'p_green', 'p_purple', 'p_red', 'p_yellow',
            'hsv_guess', 'hsv_hue', 'cnn_hsv_agree',
            'used_in_vote',
        ])

        self.total_detections = 0
        self.total_voted = 0

        if self.save_crops:
            self.crops_dir = self.output_dir / "crops"
            # Clear old crops
            if self.crops_dir.exists():
                for f in self.crops_dir.glob("*.jpg"):
                    f.unlink()
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

        results = self.yolo(
            frame, imgsz=IMGSZ, conf=DETECTION_CONF, iou=DETECTION_IOU,
            classes=[0], device="cuda:0", half=True, verbose=False
        )

        detections = []

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return [], []

        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # Classify all detections
        for i, (bbox, det_conf) in enumerate(zip(bboxes, confs)):
            self.total_detections += 1
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1

            # Filter: bbox touching frame edge (horse heads entering frame)
            fh, fw = frame.shape[:2]
            if x1 <= EDGE_MARGIN or x2 >= fw - EDGE_MARGIN:
                continue

            # Filter: aspect ratio (reject wide horse blanket detections)
            if bh < MIN_BBOX_HEIGHT:
                continue
            aspect = bh / max(bw, 1)
            if aspect < MIN_ASPECT_RATIO:
                continue

            center_x = (x1 + x2) / 2

            torso = self.extract_torso(frame, bbox)
            if torso is None:
                continue

            crop_h, crop_w = torso.shape[:2]
            crop_pixels = crop_h * crop_w
            if crop_pixels < MIN_CROP_PIXELS or crop_pixels > MAX_CROP_PIXELS:
                continue

            pred_color, pred_conf, prob_dict = self.classifier.predict_full(torso)
            hsv_guess, hsv_hue = analyze_hsv(torso)

            # Keep all detections above MIN_REASSIGN_CONF — _enforce_unique
            # may reassign them to a different (unused) color
            if pred_color and pred_conf >= MIN_REASSIGN_CONF:
                detections.append({
                    'bbox': tuple(map(int, bbox)),
                    'center_x': center_x,
                    'det_conf': det_conf,
                    'color': pred_color,
                    'conf': pred_conf,
                    'prob_dict': prob_dict,
                    'hsv_guess': hsv_guess,
                    'hsv_hue': hsv_hue,
                    'crop_size': (crop_w, crop_h),
                    'torso': torso if self.save_crops else None,
                })

        # Sort by X position (rightmost = 1st place)
        detections.sort(key=lambda d: -d['center_x'])

        # Enforce unique colors per frame (keep highest confidence for each color)
        if ENFORCE_UNIQUE_PER_FRAME:
            detections = self._enforce_unique(detections)

        # Assign positions and vote (weighted by completeness)
        jockeys = []
        n_unique = len(detections)
        used_in_vote = n_unique >= MIN_JOCKEYS_FOR_VOTE
        vote_weight = VOTE_WEIGHTS.get(n_unique, 1) if used_in_vote else 0

        if used_in_vote:
            self.vote_frames += 1

        for pos, det in enumerate(detections):
            # Vote for this position with weight
            if used_in_vote:
                self.position_votes[pos][det['color']] += vote_weight
                self.total_voted += vote_weight

            # Save crop (only for frames used in voting)
            if self.save_crops and used_in_vote and det.get('torso') is not None:
                crop_name = f"f{self.frame_num:04d}_p{pos}_{det['color']}_{det['conf']:.2f}.jpg"
                cv2.imwrite(str(self.crops_dir / crop_name), det['torso'])

            # CSV
            agree = det['color'] == det['hsv_guess']
            self.csv_writer.writerow([
                self.frame_num, pos, round(det['det_conf'], 3),
                *det['bbox'], round(det['center_x'], 1), pos,
                det['crop_size'][0], det['crop_size'][1],
                det['color'], round(det['conf'], 4),
                det['prob_dict'].get('blue', 0), det['prob_dict'].get('green', 0),
                det['prob_dict'].get('purple', 0), det['prob_dict'].get('red', 0),
                det['prob_dict'].get('yellow', 0),
                det['hsv_guess'], det['hsv_hue'], agree,
                used_in_vote,
            ])

            jockeys.append({
                'bbox': det['bbox'],
                'center_x': det['center_x'],
                'color': det['color'],
                'conf': det['conf'],
                'position': pos,
            })

        # Compute current best result from votes
        current_result = self._compute_result()
        if len(current_result) > len(self.best_result):
            self.best_result = current_result
            print(f"  [Frame {self.frame_num}] {n_unique} colors (w={vote_weight}) "
                  f"result: {current_result}")

        if len(current_result) >= REQUIRED_COLORS and not self.result_saved:
            self.final_result = current_result
            self.result_saved = True
            print(f"\n*** ALL {REQUIRED_COLORS} COLORS FOUND at frame {self.frame_num} "
                  f"(after {self.vote_frames} voted frames) ***")

        return jockeys, detections

    def _enforce_unique(self, detections):
        """Assign unique colors using full softmax probabilities.

        Instead of dropping duplicates, reassign them to their next-best
        unused color. This prevents losing a 'green' jockey that got
        classified as 'blue' when a stronger 'blue' exists.
        """
        if len(detections) <= 1:
            return detections

        # Sort by top-1 confidence descending — strongest gets first pick
        dets = sorted(detections, key=lambda d: -d['conf'])

        used_colors = set()
        assigned = []

        for det in dets:
            # Rank all colors by probability for this detection
            sorted_colors = sorted(det['prob_dict'].items(), key=lambda x: -x[1])

            found = False
            for color, prob in sorted_colors:
                if color not in used_colors and prob >= MIN_REASSIGN_CONF:
                    used_colors.add(color)
                    # Create new entry with possibly reassigned color
                    new_det = dict(det)
                    new_det['color'] = color
                    new_det['conf'] = prob
                    assigned.append(new_det)
                    found = True
                    break
            if not found:
                # Log the dropped detection so it's not silent
                orig_color = det.get('color', '?')
                orig_conf = det.get('conf', 0)
                print(f"  [WARN] Dropped detection: orig={orig_color} conf={orig_conf:.2f}, "
                      f"no unused color with prob >= {MIN_REASSIGN_CONF} "
                      f"(used: {used_colors})")

        # Sort back by center_x descending (rightmost = 1st place)
        return sorted(assigned, key=lambda d: -d['center_x'])

    def _compute_result(self):
        """Compute result from position votes with uniqueness constraint.

        Three passes:
        1. Assign positions that have >= MIN_VOTES_PER_POS for an unused color
        2. For remaining positions, accept any votes >= 2 for an unused color
        3. Fallback: if N-1 of N colors placed, put the missing color in the
           best remaining position (by total votes for that color)
        """
        if not self.position_votes:
            return []

        all_colors = set(self.classifier.classes)
        max_pos = max(self.position_votes.keys()) + 1

        # Pass 1: strict threshold
        result = [None] * max_pos
        used_colors = set()

        for pos in range(max_pos):
            votes = self.position_votes.get(pos, Counter())
            for color, count in votes.most_common():
                if count < MIN_VOTES_PER_POS:
                    break
                if color not in used_colors:
                    used_colors.add(color)
                    result[pos] = color
                    break

        # Pass 2: fill gaps with lower threshold (>= 2 votes)
        for pos in range(max_pos):
            if result[pos] is not None:
                continue
            votes = self.position_votes.get(pos, Counter())
            for color, count in votes.most_common():
                if count < 2:
                    break
                if color not in used_colors:
                    used_colors.add(color)
                    result[pos] = color
                    break

        # Pass 3: fallback — if missing 1 color, place it in best open position
        missing = all_colors - used_colors
        open_positions = [p for p in range(max_pos) if result[p] is None]

        if len(missing) == 1 and open_positions:
            missing_color = missing.pop()
            # Find the open position with most votes for this color
            best_pos = max(open_positions,
                           key=lambda p: self.position_votes.get(p, Counter()).get(missing_color, 0))
            result[best_pos] = missing_color
            used_colors.add(missing_color)
        elif missing and open_positions:
            # Multiple missing — place each in its best open position
            for color in sorted(missing):
                if not open_positions:
                    break
                # Sum all votes for this color across all open positions
                best_pos = max(open_positions,
                               key=lambda p: self.position_votes.get(p, Counter()).get(color, 0))
                votes_here = self.position_votes.get(best_pos, Counter()).get(color, 0)
                if votes_here > 0:
                    result[best_pos] = color
                    used_colors.add(color)
                    open_positions.remove(best_pos)

        return [c for c in result if c is not None]

    def print_summary(self):
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print(f"\n[PARAMETERS]")
        print(f"  DETECTION_CONF:       {DETECTION_CONF}")
        print(f"  MIN_COLOR_CONF:       {MIN_COLOR_CONF}")
        print(f"  MIN_ASPECT_RATIO:     {MIN_ASPECT_RATIO} (h/w)")
        print(f"  MIN_BBOX_HEIGHT:      {MIN_BBOX_HEIGHT}px")
        print(f"  TORSO region:         top={TORSO_TOP} bot={TORSO_BOTTOM} L={TORSO_LEFT} R={TORSO_RIGHT}")
        print(f"  IMGSZ:                {IMGSZ}")
        print(f"  MIN_JOCKEYS_FOR_VOTE: {MIN_JOCKEYS_FOR_VOTE}")
        print(f"  MIN_VOTES_PER_POS:    {MIN_VOTES_PER_POS}")
        print(f"  VOTE_WEIGHTS:         {VOTE_WEIGHTS}")
        print(f"  ENFORCE_UNIQUE:       {ENFORCE_UNIQUE_PER_FRAME}")

        print(f"\n[STATS]")
        print(f"  Total frames:         {self.frame_num}")
        print(f"  Total detections:     {self.total_detections}")
        print(f"  Voted frames:         {self.vote_frames}")
        print(f"  Total votes cast:     {self.total_voted}")

        print(f"\n[POSITION VOTE TABLE]")
        print(f"  {'Pos':<5} {'1st':>8} {'2nd':>8} {'3rd':>8} {'Winner':>10} {'Conf':>6}")
        print(f"  {'-'*50}")

        for pos in sorted(self.position_votes.keys()):
            votes = self.position_votes[pos]
            total = sum(votes.values())
            top3 = votes.most_common(3)

            cols = []
            for color, count in top3:
                pct = count / total * 100 if total > 0 else 0
                cols.append(f"{color[0].upper()}:{count}({pct:.0f}%)")
            while len(cols) < 3:
                cols.append("")

            winner = top3[0][0] if top3 else "?"
            win_pct = top3[0][1] / total * 100 if top3 and total > 0 else 0

            print(f"  {pos+1:<5} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} "
                  f"{winner:>10} {win_pct:>5.0f}%")

        # Global color appearance count (across all positions)
        global_counts = Counter()
        for pos in self.position_votes:
            for color, count in self.position_votes[pos].items():
                global_counts[color] += count
        print(f"\n[GLOBAL COLOR COUNTS] (total votes across all positions)")
        for color, count in global_counts.most_common():
            print(f"    {color:>8}: {count} votes")
        all_colors = set(self.classifier.classes)
        missing = all_colors - set(global_counts.keys())
        if missing:
            print(f"    NEVER SEEN: {missing}")

        print(f"\n[FILES]")
        print(f"  CSV log: {self.csv_path}")
        if self.save_crops:
            n = len(list(self.crops_dir.glob("*.jpg")))
            print(f"  Crops:   {self.crops_dir} ({n} files)")

    def close(self):
        self.csv_file.close()


def draw(frame, jockeys, tracker):
    for j in jockeys:
        x1, y1, x2, y2 = j['bbox']
        color = j['color']
        bgr = COLORS_BGR.get(color, (128, 128, 128))

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
        label = f"P{j['position']+1} {color} {j['conf']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        # Draw torso region
        h, w = y2 - y1, x2 - x1
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        tx1 = x1 + int(w * TORSO_LEFT)
        tx2 = x2 - int(w * TORSO_RIGHT)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)

    # Info panel
    cv2.rectangle(frame, (5, 5), (300, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {tracker.frame_num}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(jockeys)}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Voted frames: {tracker.vote_frames}", (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    n_colors = len(set(j['color'] for j in jockeys))
    cv2.putText(frame, f"Colors this frame: {n_colors}/{REQUIRED_COLORS}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Result panel
    result = tracker.final_result if tracker.final_result else tracker.best_result
    if result:
        y = 160
        for i, color in enumerate(result):
            bgr = COLORS_BGR.get(color, (128, 128, 128))
            # Show vote count
            votes = tracker.position_votes.get(i, Counter())
            top = votes.most_common(1)
            v_str = f" ({top[0][1]}v)" if top else ""
            cv2.putText(frame, f"{i+1}. {color}{v_str}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            y += 30

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--save-crops", action="store_true", help="Save torso crops")
    parser.add_argument("--no-display", action="store_true", help="No GUI window")
    args = parser.parse_args()

    print("=" * 60)
    print("RACE TRACKER (Position-Based Voting)")
    print(f"{REQUIRED_COLORS} jockeys -> sort by X -> vote per position")
    print("=" * 60)

    video_name = Path(args.video).stem
    output_dir = Path(RESULTS_DIR) / SCRIPT_NAME / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    tracker = RaceTracker(output_dir, save_crops=args.save_crops)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.0f}fps, {total} frames\n")

    params = {
        'DETECTION_CONF': DETECTION_CONF, 'DETECTION_IOU': DETECTION_IOU,
        'IMGSZ': IMGSZ, 'MIN_COLOR_CONF': MIN_COLOR_CONF,
        'TORSO_TOP': TORSO_TOP, 'TORSO_BOTTOM': TORSO_BOTTOM,
        'TORSO_LEFT': TORSO_LEFT, 'TORSO_RIGHT': TORSO_RIGHT,
        'MIN_CROP_PIXELS': MIN_CROP_PIXELS, 'MIN_REASSIGN_CONF': MIN_REASSIGN_CONF,
        'MIN_JOCKEYS_FOR_VOTE': MIN_JOCKEYS_FOR_VOTE,
        'MIN_VOTES_PER_POS': MIN_VOTES_PER_POS,
        'ENFORCE_UNIQUE_PER_FRAME': ENFORCE_UNIQUE_PER_FRAME,
        'video': args.video, 'resolution': f'{w}x{h}', 'fps': fps,
    }
    with open(output_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)

    scale = 0.5 if w > 1920 else 1.0

    writer = None
    if args.save:
        out = str(output_dir / f"{video_name}_result.mp4")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        jockeys, _ = tracker.update(frame)
        frame_draw = draw(frame.copy(), jockeys, tracker)

        if tracker.result_saved and tracker.save_frame is None:
            tracker.save_frame = frame_draw.copy()
            png_path = output_dir / f"frame_{tracker.frame_num}.png"
            cv2.imwrite(str(png_path), frame_draw)

        if writer:
            writer.write(frame_draw)

        if not args.no_display:
            disp = cv2.resize(frame_draw, (int(w * scale), int(h * scale)))
            cv2.namedWindow("Race", cv2.WINDOW_NORMAL)
            cv2.imshow("Race", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    tracker.print_summary()

    # Final
    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    result = tracker.final_result if tracker.final_result else tracker.best_result
    if result:
        for i, color in enumerate(result):
            print(f"  {i+1}. {color.upper()}")
        if len(result) < REQUIRED_COLORS:
            print(f"\n  (Only {len(result)}/{REQUIRED_COLORS} colors found)")

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
