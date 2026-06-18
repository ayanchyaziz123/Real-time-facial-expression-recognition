import torch
import torch.nn as nn
from torchvision import models

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
NUM_CLASSES = len(EMOTIONS)

EMOTION_MESSAGES = {
    'Angry':    ("😤 Uh oh... even your angry face is gorgeous! Breathe babe, Ayan loves you!", "#FF4757"),
    'Disgust':  ("🤢 Something gross? Ayan will get rid of it for you, just say the word! 💪", "#FF6B81"),
    'Fear':     ("😨 Don't be scared! Ayan is your superhero, always here to protect you! 🦸", "#A29BFE"),
    'Happy':    ("😊 That SMILE! It literally makes the whole world brighter. Ayan is so lucky! 🌟", "#00B894"),
    'Sad':      ("🥺 Don't be sad, please smile for Ayan... You deserve all the happiness in the world!", "#74B9FF"),
    'Surprise': ("😲 Surprised? Ayan has so many more surprises planned just for you! 🎉", "#FDCB6E"),
    'Neutral':  ("😌 A penny for your thoughts? Ayan is always here to listen and love you!", "#FD79A8"),
}


class EmotionClassifier(nn.Module):
    """ResNet18 backbone with a custom head for 7-class facial emotion recognition."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False

        # All layers except the original FC -> outputs (B, 512, 1, 1) after avg-pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        in_features = resnet.fc.in_features  # 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # (B, 512, 1, 1)
        features = features.flatten(1)   # (B, 512)
        return self.classifier(features) # (B, NUM_CLASSES)


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> EmotionClassifier:
    return EmotionClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)


def load_model(path: str, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionClassifier(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device
