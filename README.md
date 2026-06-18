# 💕 Ayan's Emotion Detector

Real-time facial expression recognition using **CNN Transfer Learning (ResNet18)** + **PyTorch** + **Flask**.

## Architecture
```
Input (48×48 grayscale) → Resize 224×224 → ResNet18 backbone → FC(256) → 7 classes
```

## Emotions + Messages
| Emotion  | Message |
|----------|---------|
| 😤 Angry    | Even your angry face is gorgeous! Breathe babe, Ayan loves you! |
| 😊 Happy    | That SMILE! Ayan is so lucky! |
| 🥺 Sad      | Don't be sad, please smile for Ayan! |
| 😲 Surprise | Ayan has so many more surprises planned for you! |
| 😨 Fear     | Don't be scared! Ayan will always protect you! |
| 🤢 Disgust  | Ayan will get rid of it for you! |
| 😌 Neutral  | Ayan is always here to listen and love you! |

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Get the dataset
Download **FER2013** from Kaggle and extract to:
```
data/
  train/  angry/ disgust/ fear/ happy/ sad/ surprise/ neutral/
  test/   angry/ disgust/ fear/ happy/ sad/ surprise/ neutral/
```

## Step 2 — Train the model
Open `train_emotion_model.ipynb` in Jupyter and run all cells.
This saves `emotion_model.pth` when done (~30 epochs, ~65% val accuracy).

## Step 3 — Run the Flask app
```bash
python app.py
```
Open **http://localhost:5000** — the webcam stream starts automatically.

## Project Structure
```
sentment analysis/
├── app.py                     # Flask server + camera streaming
├── model.py                   # ResNet18 architecture + emotion messages
├── train_emotion_model.ipynb  # Training notebook
├── requirements.txt
├── templates/index.html       # Beautiful dark UI
└── data/                      # FER2013 dataset (you add this)
```
