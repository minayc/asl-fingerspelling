import os
import threading
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
"""
Per-user calibration: the first BASELINE_SAMPLES successful detections are
averaged into a neutral baseline. Emotions are then classified by how far
the current features deviate from that baseline. Keep a neutral face for
the first few seconds after launch.
"""
BASELINE_SAMPLES = 10
HISTORY_LEN = 5

# Deviation thresholds (all features are normalized by inter-ocular distance)
OPEN_DELTA = 0.25    # mouth opens this much beyond neutral -> surprise
WIDTH_DELTA = 0.05   # mouth widens this much -> smile
LIFT_DELTA = 0.015   # mouth corners rise this much -> smile
DROP_DELTA = -0.015  # mouth corners drop this much -> sad

EMOTION_BACKEND = os.environ.get("EMOTION_BACKEND", "landmarks").lower()

_lock = threading.Lock()  # FaceMesh is not thread-safe
_baseline_samples = []
_baseline = None
_history = deque(maxlen=HISTORY_LEN)
_deepface = None


def _detect_deepface(frame_bgr):
    global _deepface
    if _deepface is None:
        from deepface import DeepFace  # imported lazily: slow, optional
        _deepface = DeepFace

    res = _deepface.analyze(
        frame_bgr,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="opencv",
        silent=True,
    )
    if isinstance(res, list):
        res = res[0]

    _history.append(res["dominant_emotion"])
    return Counter(_history).most_common(1)[0][0]


def euclidean(a, b):
    return np.linalg.norm([a.x - b.x, a.y - b.y])


def _features(landmarks):
    io = euclidean(landmarks[33], landmarks[263])
    if io < 1e-6:
        return None

    mouth_open = euclidean(landmarks[13], landmarks[14]) / io
    mouth_width = euclidean(landmarks[61], landmarks[291]) / io

    lip_center_y = (landmarks[13].y + landmarks[14].y) / 2.0
    corner_y = (landmarks[61].y + landmarks[291].y) / 2.0
    corner_lift = (lip_center_y - corner_y) / io

    return np.array([mouth_open, mouth_width, corner_lift])


def _classify(feats, baseline):
    d_open, d_width, d_lift = feats - baseline

    if d_open > OPEN_DELTA:
        return "surprise"
    if d_width > WIDTH_DELTA and d_lift > LIFT_DELTA:
        return "happy"
    if d_lift < DROP_DELTA:
        return "sad"
    return "neutral"


def detect_emotion(frame_bgr):
    """Detect emotion from a BGR frame.

    Returns a label ("happy", "sad", "surprise", "neutral"), or None while no
    face is visible or the neutral baseline is still being calibrated.
    """
    global _baseline, EMOTION_BACKEND

    try:
        if EMOTION_BACKEND == "deepface":
            try:
                return _detect_deepface(frame_bgr)
            except Exception as e:
                print(f"[emotion] DeepFace backend failed ({e}); "
                      "falling back to landmark detector")
                EMOTION_BACKEND = "landmarks"

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with _lock:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        feats = _features(results.multi_face_landmarks[0].landmark)
        if feats is None:
            return None

        if _baseline is None:
            _baseline_samples.append(feats)
            if len(_baseline_samples) >= BASELINE_SAMPLES:
                _baseline = np.mean(_baseline_samples, axis=0)
            return None

        _history.append(_classify(feats, _baseline))
        return Counter(_history).most_common(1)[0][0]

    except Exception:
        return None
