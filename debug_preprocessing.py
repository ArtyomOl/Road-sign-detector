import cv2
import matplotlib.pyplot as plt
from data_loader import get_train_transforms, get_val_test_transforms
from config import config

# 1. Загружаем оригинал
image_path = "D:\\archive\\Train\\14\\00014_00001_00029.png"
orig = cv2.imread(image_path)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

# 2. Применяем аугментации (например train)
transform = get_train_transforms(config.data.image_size)
augmented = transform(image=orig)
proc = augmented["image"]  # это уже torch.Tensor (C,H,W)

# 3. Для отображения нужно вернуть в numpy (H,W,C) и денормализовать
import numpy as np
import torch

def denormalize(tensor, mean, std):
    # tensor: torch.Tensor (C,H,W)
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Денормализация и перевод в numpy
proc_denorm = denormalize(proc, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
proc_denorm = np.transpose(proc_denorm.numpy(), (1, 2, 0))  # (H,W,C)
proc_denorm = np.clip(proc_denorm, 0, 1)

# 4. Визуализация
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(orig)
axes[0].set_title("Оригинал")
axes[0].axis("off")

axes[1].imshow(proc_denorm)
axes[1].set_title("После предобработки")
axes[1].axis("off")

plt.show()
