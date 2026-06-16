# ASL Fingerspelling to Speech

A real-time system that reads American Sign Language (ASL) fingerspelling from a
webcam, turns the letters into words, speaks them aloud, and shows a meme based
on your facial expression.

Course project for **CMP3011: Introduction to Computer Vision**, Bahçeşehir University.

- **Authors:** Cansu Culu, Mina Ezo Aycı
- **Instructor:** MD Imran Hosen

## What it does

- Recognizes ASL letters A–Z (plus `space` and `delete` gestures) from a webcam.
- Builds words from the letters and speaks them with text-to-speech.
- Detects your emotion (happy / sad / surprise / neutral) and shows a matching meme.

It uses MediaPipe to find 21 hand landmarks, then a Random Forest classifier
(scikit-learn). **Test accuracy: 99.5%.**

## Setup

Needs **Python 3.12** and a webcam.

```
git clone https://github.com/<your-username>/asl-fingerspelling.git
cd asl-fingerspelling
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset and model

The dataset and trained model are not in the repo. Download the
[ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
from Kaggle and extract it to `data/asl_alphabet_train/`, then build the model:

```
python src/extract_landmarks.py    # images -> data/landmarks.csv
python src/train_classifier.py     # -> models/gesture_classifier.pkl
```

## Run

```
python main.py
```

- Sign letters A–Z to spell a word.
- **space** gesture: finish the word and say it out loud.
- **delete** gesture: remove the last letter.
- **q**: quit.

## Notes

- Letters **J** and **Z** use motion, so they are read from the final pose only.
- Emotion detection covers 4 of the 7 basic emotions.
- Works best in good lighting.
