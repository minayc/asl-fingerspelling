import cv2
import os
import numpy as np

MEMES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memes")

WORD_TO_MEME = {
    "HELLO": "Hello.png",
    "HI": "Hello.png",
    "YES": "Yes.png",
    "NO": "No.png",
    "SAD": "Sad.png",
    "HAPPY": "Happy.png",
    "WOW": "Wow.png",
    "GREAT": "Yes.png",
    "BAD": "Sad.png",
}

EMOTION_TO_MEME = {
    "happy": "Happy.png",
    "sad": "Sad.png",
    "angry": "No.png",
    "surprise": "Wow.png",
    "fear": "No.png",
    "disgust": "No.png",
}


def get_meme_path(word, emotion):
    if word and word.upper() in WORD_TO_MEME:
        return os.path.join(MEMES_DIR, WORD_TO_MEME[word.upper()])
    if emotion and emotion in EMOTION_TO_MEME:
        return os.path.join(MEMES_DIR, EMOTION_TO_MEME[emotion])
    return None


def overlay_meme(frame, meme_path):
    meme = cv2.imread(meme_path, cv2.IMREAD_UNCHANGED)
    if meme is None:
        return frame

    h, w = frame.shape[:2]
    meme_w = w // 2
    meme_h = h // 2
    meme = cv2.resize(meme, (meme_w, meme_h))

    x_offset = w - meme_w
    y_offset = 0

    if meme.shape[2] == 4:
        alpha = meme[:, :, 3] / 255.0
        for c in range(3):
            frame[y_offset:y_offset + meme_h, x_offset:x_offset + meme_w, c] = (
                    alpha * meme[:, :, c] + (1 - alpha) * frame[y_offset:y_offset + meme_h, x_offset:x_offset + meme_w,
                                                          c]
            ).astype(np.uint8)
    else:
        frame[y_offset:y_offset + meme_h, x_offset:x_offset + meme_w] = meme

    return frame