# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

from config import config
from data_loader import create_dataloaders, get_class_names
from model import create_model
from trainer import Trainer

def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, class_names: dict):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(config.project.device)
            labels = labels.to(config.project.device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    target_names = [class_names[i] for i in range(config.data.num_classes)]
    
    # Указываем zero_division=0 для случаев, когда класс в test set есть, а в predictions нет
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\n--- Test Set Evaluation ---")
    print(report)

def main():
    # Создание папок для моделей и логов
    os.makedirs(os.path.dirname(config.project.model_save_path), exist_ok=True)
    os.makedirs(config.project.log_dir, exist_ok=True)

    # 1. Загрузка данных
    dataloaders = create_dataloaders()
    class_names = get_class_names()

    # 2. Создание модели
    model = create_model()

    # 3. Настройка обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.train.factor, patience=config.train.patience // 2
    )

    # 4. Настройка TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.project.log_dir, config.project.project_name))

    # 5. Инициализация и запуск тренера
    trainer = Trainer(model, dataloaders, criterion, optimizer, lr_scheduler, writer)
    trainer.train()

    # 6. Тестирование на тестовом наборе
    test_model(trainer.model, dataloaders['test'], class_names)
    
    writer.close()

if __name__ == '__main__':
    main()
