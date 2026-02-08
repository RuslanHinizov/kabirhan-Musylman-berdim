"""
Full Pipeline Test: Detection + Tracking + ReID
5 лошадей + 5 жокеев с стабильными ID и цветами

Usage:
    python tools/test_full_pipeline.py data/videos/exp10_cam1.mp4
    python tools/test_full_pipeline.py data/videos/exp10_cam1.mp4 --save
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from api.vision.bytetrack import ByteTracker
from api.vision.dtypes import Detection
from api.vision.jockey_horse_reid import JockeyHorseReID

# Class IDs
HORSE_CLASS_ID = 17
PERSON_CLASS_ID = 0

# Detection settings - IMPROVED for distant objects
CONFIDENCE_THRESHOLD = 0.15  # Lower for far objects
IMAGE_SIZE = 1920  # Higher for 4K video
IOU_THRESHOLD = 0.3

# Expected count
EXPECTED_HORSES = 5
EXPECTED_JOCKEYS = 5

# Distinct colors for 5 horses/jockeys (BGR format)
PAIR_COLORS = [
    ((0, 0, 255), "Red"),       # Horse 1 - Red
    ((0, 255, 0), "Green"),     # Horse 2 - Green
    ((255, 0, 0), "Blue"),      # Horse 3 - Blue
    ((0, 255, 255), "Yellow"),  # Horse 4 - Yellow
    ((255, 0, 255), "Magenta"), # Horse 5 - Magenta
]


class StableTracker:
    """
    Tracker with ReID for stable IDs
    Maintains consistent IDs using appearance features
    """

    def __init__(self, reid_model: JockeyHorseReID, max_horses: int = 5):
        self.reid = reid_model
        self.max_horses = max_horses

        # ByteTrack for frame-to-frame tracking
        self.horse_tracker = ByteTracker(
            track_thresh=0.3,
            track_buffer=60,  # Longer buffer for occlusions
            match_thresh=0.7,
            low_thresh=0.1,
            confirm_frames=2
        )
        self.person_tracker = ByteTracker(
            track_thresh=0.25,
            track_buffer=40,
            match_thresh=0.6,
            low_thresh=0.1,
            confirm_frames=2
        )

        # ReID gallery: stable_id -> feature vector
        self.horse_gallery = {}  # stable_id -> (feature, bbox_history)
        self.person_gallery = {}

        # Mapping: track_id -> stable_id
        self.horse_id_map = {}
        self.person_id_map = {}

        # Next stable ID
        self.next_horse_id = 1
        self.next_person_id = 1

        # Pairing: horse_stable_id -> person_stable_id
        self.pairs = {}

    def _extract_features(self, frame, tracks, is_horse=True):
        """Extract ReID features for tracks"""
        features = {}
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            # Ensure valid bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Get feature from ReID model (same method for both)
            feat = self.reid.get_embedding(crop)

            if feat is not None:
                features[track.track_id] = feat

        return features

    def _match_to_gallery(self, features, gallery, id_map, is_horse=True):
        """Match features to gallery and assign stable IDs"""
        threshold = 0.75  # Higher threshold = more strict matching = more unique IDs

        for track_id, feat in features.items():
            # Already mapped?
            if track_id in id_map:
                stable_id = id_map[track_id]
                # Update gallery feature (moving average)
                if stable_id in gallery:
                    old_feat = gallery[stable_id]
                    gallery[stable_id] = 0.8 * old_feat + 0.2 * feat
                continue

            # Find best match in gallery
            best_match = None
            best_sim = threshold

            for stable_id, gallery_feat in gallery.items():
                # Cosine similarity (features are normalized)
                sim = float(np.dot(feat, gallery_feat))
                if sim > best_sim:
                    best_sim = sim
                    best_match = stable_id

            if best_match is not None:
                # Matched to existing
                id_map[track_id] = best_match
                # Update gallery
                gallery[best_match] = 0.7 * gallery[best_match] + 0.3 * feat
            else:
                # New identity
                if is_horse:
                    if self.next_horse_id <= self.max_horses:
                        new_id = self.next_horse_id
                        self.next_horse_id += 1
                        id_map[track_id] = new_id
                        gallery[new_id] = feat
                else:
                    if self.next_person_id <= self.max_horses:
                        new_id = self.next_person_id
                        self.next_person_id += 1
                        id_map[track_id] = new_id
                        gallery[new_id] = feat

    def update(self, frame, horse_dets, person_dets):
        """
        Update tracker with new detections
        Returns: (horse_results, person_results)
        Each result is list of (stable_id, bbox, confidence)
        """
        # ByteTrack update
        horse_tracks = self.horse_tracker.update(horse_dets)
        person_tracks = self.person_tracker.update(person_dets)

        # Extract features
        horse_feats = self._extract_features(frame, horse_tracks, is_horse=True)
        person_feats = self._extract_features(frame, person_tracks, is_horse=False)

        # Match to gallery
        self._match_to_gallery(horse_feats, self.horse_gallery, self.horse_id_map, is_horse=True)
        self._match_to_gallery(person_feats, self.person_gallery, self.person_id_map, is_horse=False)

        # Build results with stable IDs (unique per frame)
        horse_results = []
        seen_horse_ids = set()
        for track in horse_tracks:
            stable_id = self.horse_id_map.get(track.track_id)
            if stable_id is not None and stable_id <= self.max_horses:
                if stable_id not in seen_horse_ids:
                    horse_results.append((stable_id, track.bbox, track.confidence))
                    seen_horse_ids.add(stable_id)

        person_results = []
        seen_person_ids = set()
        for track in person_tracks:
            stable_id = self.person_id_map.get(track.track_id)
            if stable_id is not None and stable_id <= self.max_horses:
                if stable_id not in seen_person_ids:
                    person_results.append((stable_id, track.bbox, track.confidence))
                    seen_person_ids.add(stable_id)

        # Pair jockeys with horses by proximity
        self._update_pairs(horse_results, person_results)

        return horse_results, person_results

    def _update_pairs(self, horses, persons):
        """Pair jockeys with nearest horses"""
        for h_id, h_bbox, _ in horses:
            h_cx = (h_bbox[0] + h_bbox[2]) / 2
            h_cy = (h_bbox[1] + h_bbox[3]) / 2

            best_person = None
            best_dist = float('inf')

            for p_id, p_bbox, _ in persons:
                p_cx = (p_bbox[0] + p_bbox[2]) / 2
                p_cy = (p_bbox[1] + p_bbox[3]) / 2

                dist = ((h_cx - p_cx)**2 + (h_cy - p_cy)**2)**0.5

                # Person should be above horse (jockey)
                if p_cy < h_cy and dist < best_dist:
                    best_dist = dist
                    best_person = p_id

            if best_person is not None and best_dist < 300:
                self.pairs[h_id] = best_person


def draw_results(frame, horses, persons, pairs, frame_num):
    """Draw results with consistent colors"""

    # Draw horses
    for stable_id, bbox, conf in horses:
        if stable_id > len(PAIR_COLORS):
            continue

        x1, y1, x2, y2 = bbox
        color, color_name = PAIR_COLORS[stable_id - 1]

        # Thick box for horse
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        # Label
        label = f"H{stable_id} ({color_name}) {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw persons with same color as paired horse
    for stable_id, bbox, conf in persons:
        # Find paired horse color
        paired_horse = None
        for h_id, p_id in pairs.items():
            if p_id == stable_id:
                paired_horse = h_id
                break

        if paired_horse and paired_horse <= len(PAIR_COLORS):
            color, color_name = PAIR_COLORS[paired_horse - 1]
        elif stable_id <= len(PAIR_COLORS):
            color, color_name = PAIR_COLORS[stable_id - 1]
        else:
            continue

        x1, y1, x2, y2 = bbox

        # Thinner box for jockey
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label at bottom
        label = f"J{stable_id} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
        cv2.putText(frame, label, (x1 + 5, y2 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Info panel
    info = f"Frame {frame_num} | Horses: {len(horses)}/{EXPECTED_HORSES} | Jockeys: {len(persons)}/{EXPECTED_JOCKEYS}"
    cv2.rectangle(frame, (5, 5), (700, 50), (0, 0, 0), -1)
    cv2.putText(frame, info, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    return frame


def test_full_pipeline(video_path: str, show: bool = True, save: bool = False):
    """Run full pipeline test"""

    print(f"\n{'='*60}")
    print(f"FULL PIPELINE TEST")
    print(f"Detection + ByteTrack + ReID")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Expected: {EXPECTED_HORSES} horses, {EXPECTED_JOCKEYS} jockeys")
    print(f"Image size: {IMAGE_SIZE} (for distant objects)")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}\n")

    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    # Load models
    print("Loading YOLO model...")
    model = YOLO("models/yolov8n.pt")

    print("Loading ReID model (OSNet)...")
    reid = JockeyHorseReID(
        similarity_threshold=0.55,
        jockey_weight=0.7,
        horse_weight=0.3,
        device="cuda"
    )

    # Create stable tracker
    tracker = StableTracker(reid, max_horses=EXPECTED_HORSES)
    print("Models loaded!\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Display scale for 4K
    display_scale = 0.5 if width > 1920 else 1.0
    display_w = int(width * display_scale)
    display_h = int(height * display_scale)

    # Video writer
    writer = None
    if save:
        output_path = str(Path(video_path).stem) + "_full_pipeline.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving to: {output_path}")

    print("\nProcessing... (Press 'q' to quit, SPACE to pause)\n")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # YOLO detection with higher resolution
        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[HORSE_CLASS_ID, PERSON_CLASS_ID],
            device="cuda:0",
            half=True,
            verbose=False
        )

        # Parse detections
        detections = results[0].boxes.data.cpu().numpy()

        horse_dets = []
        person_dets = []

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if int(cls_id) == HORSE_CLASS_ID:
                horse_dets.append(Detection(bbox=bbox, confidence=conf))
            elif int(cls_id) == PERSON_CLASS_ID:
                person_dets.append(Detection(bbox=bbox, confidence=conf))

        # Update tracker with ReID
        horses, persons = tracker.update(frame, horse_dets, person_dets)

        # Draw results
        frame_drawn = draw_results(frame, horses, persons, tracker.pairs, frame_num)

        # Save
        if writer:
            writer.write(frame_drawn)

        # Show
        if show:
            if display_scale != 1.0:
                display_frame = cv2.resize(frame_drawn, (display_w, display_h))
            else:
                display_frame = frame_drawn

            cv2.imshow("Full Pipeline Test", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Paused at frame {frame_num}")
                cv2.waitKey(0)

        # Progress
        if frame_num % 50 == 0:
            h_ids = sorted([h[0] for h in horses])
            p_ids = sorted([p[0] for p in persons])
            print(f"Frame {frame_num}/{total_frames} | "
                  f"Horses: {h_ids} | Jockeys: {p_ids}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {frame_num} frames")
    print(f"Horse gallery: {list(tracker.horse_gallery.keys())}")
    print(f"Person gallery: {list(tracker.person_gallery.keys())}")
    print(f"Pairs (Horse->Jockey): {tracker.pairs}")
    print(f"{'='*60}\n")

    if save:
        print(f"Saved to: {output_path}")


def main():
    global IMAGE_SIZE, CONFIDENCE_THRESHOLD

    parser = argparse.ArgumentParser(description="Full Pipeline Test")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--no-show", action="store_true", help="Don't show window")
    parser.add_argument("--imgsz", type=int, default=1920, help="YOLO image size")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence")

    args = parser.parse_args()
    IMAGE_SIZE = args.imgsz
    CONFIDENCE_THRESHOLD = args.conf

    test_full_pipeline(
        video_path=args.video,
        show=not args.no_show,
        save=args.save
    )


if __name__ == "__main__":
    main()
