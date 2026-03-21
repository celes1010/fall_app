import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ===== PATHS =====
DATA_PATH = os.path.join("data", "dataset_pose.csv")
MODEL_PATH = os.path.join("models", "fall_pose_model.pkl")

# Thresholds to test
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found. Run create_dataset_from_videos.py first.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found. Run train_model_pose.py first.")
        return

    # ----- load dataset -----
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        print("[ERROR] 'label' column missing in dataset.")
        return

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y_true = df["label"].astype(int).values

    # ----- load model -----
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict) and "model" in data and "feature_cols" in data:
        model = data["model"]
    else:
        model = data

    y_proba = model.predict_proba(X)[:, 1]

    print("\n=== Threshold Tuning (Pose Dataset) ===")
    print(f"Samples: {len(y_true)}, Features: {len(feature_cols)}\n")

    for thr in THRESHOLDS:
        y_pred = (y_proba >= thr).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        print(f"--- Threshold = {thr:.2f} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print()


if __name__ == "__main__":
    main()
