import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import joblib

DATA_PATH = os.path.join("data", "dataset_pose.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fall_pose_model.pkl")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        print("[ERROR] 'label' column missing.")
        return

    # -------------------------------------------------------
    # FIX 1: Group-based split to prevent data leakage
    # Your CSV must have a 'clip_id' column (video/clip name).
    # If it doesn't, add one when building the dataset:
    #   e.g. df["clip_id"] = video_filename or clip_index
    # -------------------------------------------------------
    if "clip_id" not in df.columns:
        print("[WARN] No 'clip_id' column found — "
              "add clip/video IDs to your dataset to prevent leakage.")
        print("[WARN] Falling back to random split (results will be optimistic).")
        groups = np.arange(len(df))   # each row = its own group (no grouping)
    else:
        groups = df["clip_id"].values

    feature_cols = [c for c in df.columns if c not in ("label", "clip_id")]
    X = df[feature_cols].values
    y = df["label"].astype(int).values

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Features: {len(feature_cols)}")

    # -------------------------------------------------------
    # FIX 2: GroupShuffleSplit keeps clips together
    # -------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # -------------------------------------------------------
    # FIX 3: Scale features (helps generalization)
    # -------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # -------------------------------------------------------
    # FIX 4: Regularize the RandomForest to reduce overfitting
    # -------------------------------------------------------
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,          # was None (fully grown = overfit)
        min_samples_leaf=10,   # prevents tiny leaves memorizing noise
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    print("[INFO] Training RandomForest...")
    clf.fit(X_train, y_train)

    # ---------- EVALUATION ----------
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc          = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n==== MODEL PERFORMANCE (POSE DATASET) ====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("=========================================\n")

    joblib.dump(
        {"model": clf, "scaler": scaler, "feature_cols": feature_cols},
        MODEL_PATH
    )
    print(f"[INFO] Saved model → {MODEL_PATH}")


if __name__ == "__main__":
    main()