import csv
import os

import cv2
import mediapipe as mp
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "asl_alphabet_train")
OUTPUT_CSV = os.path.join(DATA_DIR, "landmarks.csv")

LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3  # x, y, z

mp_hands = mp.solutions.hands


def extract_landmarks_from_image(image_path, hands):
    image = cv2.imread(image_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    wrist = hand.landmark[0]
    middle_mcp = hand.landmark[9]

    scale = np.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    if scale < 1e-6:
        return None

    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])

    return landmarks


def main():
    header = [f"{axis}{i}" for i in range(LANDMARK_COUNT) for axis in ("x", "y", "z")]
    header.append("label")

    letter_folders = sorted(
        f for f in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, f))
    )

    processed = 0
    skipped = 0

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands, \
         open(OUTPUT_CSV, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(header)

        for label in letter_folders:
            folder_path = os.path.join(TRAIN_DIR, label)
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for filename in image_files:
                image_path = os.path.join(folder_path, filename)
                landmarks = extract_landmarks_from_image(image_path, hands)

                if landmarks is None:
                    skipped += 1
                    continue

                writer.writerow(landmarks + [label])
                processed += 1

                if processed % 1000 == 0:
                    print(f"Processed {processed} images (skipped {skipped} — no hand detected)")

    print(f"\nDone. {processed} images saved to {OUTPUT_CSV}")
    print(f"Skipped {skipped} images with no detected hand.")


if __name__ == "__main__":
    main()
