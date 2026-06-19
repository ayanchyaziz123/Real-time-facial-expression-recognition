import os
import base64
import cv2
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
from model import build_model, load_model, EMOTIONS, EMOTION_MESSAGES

app = FastAPI()
templates = Jinja2Templates(directory='templates')

MODEL_PATH = 'emotion_model.pth'
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(MODEL_PATH):
    model, DEVICE = load_model(MODEL_PATH, DEVICE)
    print(f'[+] Loaded model from {MODEL_PATH}')
else:
    model = build_model(pretrained=False).to(DEVICE)
    model.eval()
    print('[!] No trained model found — running in demo mode')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FrameData(BaseModel):
    frame: str = ''


def predict_emotion(face_img_bgr):
    img = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return EMOTIONS[idx.item()], round(conf.item() * 100, 1)


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')


@app.post('/predict')
async def predict(data: FrameData):
    frame_b64 = data.frame

    if not frame_b64:
        return {'label': 'Neutral', 'confidence': 0, 'faces': [], 'message': '', 'color': '#a78bfa'}

    if ',' in frame_b64:
        frame_b64 = frame_b64.split(',')[1]

    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return {'label': 'Neutral', 'confidence': 0, 'faces': [], 'message': '', 'color': '#a78bfa'}

    if frame is None:
        return {'label': 'Neutral', 'confidence': 0, 'faces': [], 'message': '', 'color': '#a78bfa'}

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=4, minSize=(30, 30))

    faces_out = []
    label, conf = 'Neutral', 0.0

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop   = frame[y:y+h, x:x+w]
        label, conf = predict_emotion(face_crop)
        faces_out   = [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}]

    msg, color = EMOTION_MESSAGES.get(label, ('', '#a78bfa'))
    return {'label': label, 'confidence': conf, 'faces': faces_out, 'message': msg, 'color': color}


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host='0.0.0.0', port=port)
