import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_classifier.pkl")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.pkl")

# Column indices: x at 0,3,6,..., landmark 9's x/y at columns 27/28
X9_COL = 27
Y9_COL = 28


def renormalize(X):
    """Rescale each row by the wrist -> middle-MCP distance.

    All landmark vectors are wrist-relative, so dividing by this distance
    makes features scale-invariant. The operation is idempotent: rows already
    normalized this way (new extract_landmarks.py) are unchanged, while rows
    from the old max-abs format are converted to the new scale.
    """
    scale = np.hypot(X[:, X9_COL], X[:, Y9_COL])
    valid = scale > 1e-6
    return X[valid] / scale[valid, None], valid


def mirror(X):
    """Flip handedness by negating all x coordinates."""
    Xm = X.copy()
    Xm[:, 0::3] *= -1
    return Xm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Ablation: train without mirror augmentation. Reports accuracy "
             "only; does NOT overwrite the saved model.",
    )
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)

    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values

    X, valid = renormalize(X)
    y = y[valid]

    if args.no_mirror:
        print(f"ABLATION: mirror augmentation disabled. Training samples: {len(X)}")
    else:
        X = np.vstack([X, mirror(X)])
        y = np.concatenate([y, y])
        print(f"Training samples after mirror augmentation: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if args.no_mirror:
        print("\nAblation run complete — model NOT saved (use the accuracy above for Table 2).")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print("\nConfusion Matrix (rows = true label, cols = predicted):")
    print(cm)

    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d", cmap="Blues")
    ax.set_title("Gesture Classifier Confusion Matrix (test set)")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix figure saved to {cm_path}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    labels = sorted(clf.classes_.tolist())
    joblib.dump(labels, LABELS_PATH)
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()
