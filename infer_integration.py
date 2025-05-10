# infer_integration.py

from PIL import Image
import numpy as np
import torch
from torchvision import transforms, models

MODEL_PATH = "models/animal_car_fruit.pth"
CLASSES = ["animal", "car", "fruit"]

# Трансформации (как в train.py → val_tf)
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


def predict_image(model, device, img_input):
    # Приводим всё к PIL.Image
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        # Если одномерный или 2D, считаем как grayscale; если 3D, RGB
        if img_input.ndim == 2:
            img = Image.fromarray(img_input).convert("RGB")
        else:
            img = Image.fromarray(img_input.astype(np.uint8))
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise ValueError(f"Неподдерживаемый тип {type(img_input)} для predict_image")

    # Теперь можно безопасно применять трансформы
    x = _tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        cls_id = logits.argmax(1).item()
    return CLASSES[cls_id]
