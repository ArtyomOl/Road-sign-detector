# data_loader.py
import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Any

from config import config

def get_train_transforms(img_size: Tuple[int, int]) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, min_height=1, min_width=1, fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_test_transforms(img_size: Tuple[int, int]) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class GTSRBDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform: A.Compose):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label

def create_dataloaders() -> Dict[str, DataLoader]:
    """
    Загружает данные, разделяет их и создает DataLoader'ы для train, val и test.
    """
    # 1. Загрузка train/val данных из папок
    train_image_paths = []
    train_labels = []
    for class_id in range(config.data.num_classes):
        class_dir = os.path.join(config.data.train_data_path, str(class_id))
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                train_image_paths.append(os.path.join(class_dir, img_name))
                train_labels.append(class_id)
    
    # 2. Разделение на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_image_paths, train_labels,
        test_size=config.train.test_split_size,
        random_state=42,
        stratify=train_labels  # Важно для сохранения баланса классов
    )


    if config.train.train_subset_fraction < 1.0:
        print(f"Using a subset of training data: {config.train.train_subset_fraction * 100:.0f}%")
        # Используем train_test_split для удобного стратифицированного сэмплирования
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=config.train.train_subset_fraction,
            random_state=42,
            stratify=y_train  # КРИТИЧЕСКИ ВАЖНО для сохранения баланса классов в подвыборке
        )

    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # 3. Создание кастомных Dataset'ов
    train_dataset = GTSRBDataset(X_train, y_train, transform=get_train_transforms(config.data.image_size))
    val_dataset = GTSRBDataset(X_val, y_val, transform=get_val_test_transforms(config.data.image_size))

    # 4. Решение проблемы дисбаланса классов с помощью WeightedRandomSampler
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in y_train])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # 5. Создание DataLoader'ов для train и val
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        sampler=sampler, # Используем семплер
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # 6. Загрузка и создание DataLoader для test
    test_df = pd.read_csv(config.data.test_data_csv_path)
    test_image_paths = [os.path.join(config.data.test_data_images_path, p) for p in test_df['Path'].values]
    test_labels = test_df['ClassId'].values
    print(f"Test samples: {len(test_image_paths)}")

    test_dataset = GTSRBDataset(test_image_paths, test_labels, transform=get_val_test_transforms(config.data.image_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}

def get_class_names() -> Dict[int, str]:
    """Загружает маппинг ID класса в имя."""
    # df = pd.read_csv(config.data.class_names_path)
    # return dict(zip(df.ClassId, df.SignName))
    return {i: i for i in range(43)}