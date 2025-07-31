# model.py
import torch.nn as nn
import timm
from config import config

def create_model() -> nn.Module:
    """
    Создает модель CNN с предобученными весами и заменяет классификатор.
    """
    model = timm.create_model(
        config.train.model_name,
        pretrained=config.train.pretrained,
        num_classes=config.data.num_classes
    )
    print(f"Model '{config.train.model_name}' created with {config.data.num_classes} classes.")
    return model
