import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from utils.features import landmarks_to_feature_vector

# ====== CONFIG ======
DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "dataset_pose.csv")

CATEGORIES = {
    "Fall": 1,
    "No_Fall": 0,
}

FRAME_STEP = 3   # use every 3rd frame to reduce size (you can change)
MIN_CONF = 0.5   # minimum detection confidence for pose


def process_video(video_path, label, pose):
    """Extract pose feature vectors from a single video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    samples = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # skip frames to reduce dataset size
        if frame_idx % FRAME_STEP != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            continue

        lm_list = results.pose_landmarks.landmark
        feat = landmarks_to_feature_vector(lm_list)
        samples.append(feat)

    cap.release()
    print(f"    -> {len(samples)} frames used from {os.path.basename(video_path)}")
    return samples


def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=MIN_CONF,
        min_tracking_confidence=0.5
    )

    all_features = []
    all_labels = []

    for class_name, label in CATEGORIES.items():
        raw_dir = os.path.join(DATA_DIR, class_name, "Raw_Video")
        if not os.path.isdir(raw_dir):
            print(f"[WARN] Missing folder: {raw_dir}")
            continue

        print(f"[INFO] Processing class '{class_name}' videos in: {raw_dir}")

        for fname in os.listdir(raw_dir):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            fpath = os.path.join(raw_dir, fname)
            print(f"[VIDEO] {fpath}")
            samples = process_video(fpath, label, pose)

            if samples:
                all_features.extend(samples)
                all_labels.extend([label] * len(samples))

    if not all_features:
        print("[ERROR] No samples extracted. Check your videos and paths.")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"[INFO] Final feature matrix shape: {X.shape}, labels: {y.shape}")

    # build DataFrame
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
