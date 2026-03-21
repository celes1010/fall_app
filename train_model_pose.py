import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import joblib

# =========================
# PATHS & SETTINGS
# =========================
DATA_PATH = os.path.join("data", "dataset_pose.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fall_pose_model.pkl")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found. "
              f"Run create_dataset_from_videos.py first.")
        return

    # ---------- LOAD DATA ----------
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        print("[ERROR] 'label' column missing in dataset.")
        return

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].astype(int).values

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Number of features: {len(feature_cols)}")

    # ---------- TRAIN / TEST SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ---------- MODEL ----------
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    print("[INFO] Training RandomForest...")
    clf.fit(X_train, y_train)

    # ---------- EVALUATION ----------
    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n==== MODEL PERFORMANCE (POSE DATASET) ====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("=========================================\n")

    # ---------- SAVE MODEL ----------
    joblib.dump(
        {"model": clf, "feature_cols": feature_cols},
        MODEL_PATH
    )
    print(f"[INFO] Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
