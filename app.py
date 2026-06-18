import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify
from torchvision import transforms
from PIL import Image
from model import build_model, load_model, EMOTIONS, EMOTION_MESSAGES

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = 'emotion_model.pth'
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ───────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    model, DEVICE = load_model(MODEL_PATH, DEVICE)
    print(f"[✓] Loaded trained model from {MODEL_PATH}")
else:
    # Demo mode: model weights not trained yet — shows random predictions
    model = build_model(pretrained=False).to(DEVICE)
    model.eval()
    print("[!] No trained model found. Run the Jupyter notebook first!")

# ── Image transforms (match training preprocessing) ─────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Shared state ─────────────────────────────────────────────────────────────
current_emotion = {'label': 'Neutral', 'confidence': 0.0}


def predict_emotion(face_img_bgr):
    img = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return EMOTIONS[idx.item()], round(conf.item() * 100, 1)


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(60, 60))

        # Run model every 3 frames to keep UI smooth
        frame_count += 1
        if len(faces) > 0 and frame_count % 3 == 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_crop = frame[y:y+h, x:x+w]
            label, conf = predict_emotion(face_crop)
            current_emotion['label'] = label
            current_emotion['confidence'] = conf

        # Draw on frame
        for (x, y, w, h) in faces:
            label = current_emotion['label']
            _, color_hex = EMOTION_MESSAGES.get(label, ("", "#FFFFFF"))
            # Convert hex to BGR
            color_hex = color_hex.lstrip('#')
            r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (b, g, r)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
            cv2.rectangle(frame, (x, y-35), (x+w, y), color_bgr, -1)
            cv2.putText(frame, f"{label} {current_emotion['confidence']}%",
                        (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

    cap.release()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def emotion_data():
    label = current_emotion['label']
    message, color = EMOTION_MESSAGES.get(label, ("", "#FFFFFF"))
    return jsonify(
        label=label,
        confidence=current_emotion['confidence'],
        message=message,
        color=color,
    )


if __name__ == '__main__':
    print("🚀 Starting Ayan's Emotion Detector...")
    print(f"   Device: {DEVICE}")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=False, threaded=True, port=5000)
