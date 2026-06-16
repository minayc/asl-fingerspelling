import os
import sys
import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tts import speak


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_classifier.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.pkl")

CONSECUTIVE_LETTER = 15
CONSECUTIVE_ACTION = 20
MIN_CONFIDENCE = 0.5  # reject hand poses the classifier is unsure about

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(hand_landmarks):
    """Wrist-relative landmarks normalized by hand size (wrist -> middle MCP)."""
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]

    scale = np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    if scale < 1e-6:
        return None

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return landmarks


def main():
    clf = joblib.load(MODEL_PATH)
    labels = joblib.load(LABELS_PATH)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    current_word = ""
    sentence = []
    last_letter = None
    letter_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for a natural selfie view; training is mirror-augmented
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            flat = extract_landmarks(hand_landmarks)
            if flat is not None:
                X = np.array(flat).reshape(1, -1)
                proba = clf.predict_proba(X)[0]
                best = int(np.argmax(proba))
                top = str(clf.classes_[best])
                if proba[best] >= MIN_CONFIDENCE:
                    predicted = top

                    cv2.putText(
                        frame, f"{top} ({proba[best]:.0%})", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA,
                    )
                else:
                    # Below-threshold guess, shown dimmed for debugging
                    cv2.putText(
                        frame, f"{top}? ({proba[best]:.0%})", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (140, 140, 140), 2, cv2.LINE_AA,
                    )

        if predicted == last_letter:
            letter_count += 1
        else:
            last_letter = predicted
            letter_count = 1

        if predicted is not None:
            if predicted == "space" and letter_count == CONSECUTIVE_ACTION:
                if current_word:
                    sentence.append(current_word)
                    speak(current_word)
                    current_word = ""

            elif predicted == "del" and letter_count == CONSECUTIVE_ACTION:
                current_word = current_word[:-1]

            elif predicted not in ("space", "del") and letter_count == CONSECUTIVE_LETTER:
                current_word += predicted

        sentence_str = " ".join(sentence)
        display_str = sentence_str + (" " if sentence_str else "") + current_word

        cv2.putText(
            frame, display_str, (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow("ASL Gesture Recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if sentence or current_word:
        full = " ".join(sentence + ([current_word] if current_word else []))
        print(f"Final sentence: {full}")


if __name__ == "__main__":
    main()
