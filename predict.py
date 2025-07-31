# predict.py
import torch
import cv2
import argparse
from PIL import Image
import torch.nn.functional as F

from config import config
from model import create_model
from data_loader import get_val_test_transforms, get_class_names

def predict(model_path: str, image_path: str):
    # 1. Загрузка названий классов
    class_names = get_class_names()

    # 2. Создание модели и загрузка весов
    device = torch.device(config.project.device if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Загрузка и предобработка изображения
    transform = get_val_test_transforms(config.data.image_size)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    processed_image = transform(image=image)['image']
    # Добавление batch-измерения
    processed_image = processed_image.unsqueeze(0).to(device)

    # 4. Предсказание
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_class_id = torch.max(probabilities, 0)
    
    predicted_class_name = class_names[predicted_class_id.item()]

    print("--- Prediction Result ---")
    print(f"Image: {image_path}")
    print(f"Predicted Sign: '{predicted_class_name}' (Class ID: {predicted_class_id.item()})")
    print(f"Confidence: {confidence.item():.4f}")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("image", type=str, help="D:\\archive\Meta\\1.png")
    # parser.add_argument("--model", type=str, default=config.project.model_save_path, help="D:\\Python projects\\Road-sign-detector\\models\\best_model.pth")
    # args = parser.parse_args()
    
    predict("D:\\Python projects\\Road-sign-detector\\models\\best_model.pth", "D:\\archive\Meta\\9.png")
