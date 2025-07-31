# config.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple

class DataConfig(BaseModel):
    train_data_path: str = "D:\\archive\\Train"
    test_data_csv_path: str = "D:\\archive\\Test.csv"
    test_data_images_path: str = "D:\\archive\\Test" # Root for test images
    class_names_path: str = "data/signnames.csv"
    num_classes: int = 43
    image_size: Tuple[int, int] = (64, 64)

class TrainConfig(BaseModel):
    model_name: str = "resnet50"
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 20
    patience: int = 3
    factor: float = 0.2
    pretrained: bool = True
    test_split_size: float = 0.2
    train_subset_fraction: float = Field(0.1, ge=0.01, le=1.0)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    project_name: str = "Road-sign-detector"
    log_dir: str = "logs"
    model_save_path: str = "models/best_model.pth"
    device: str = "cpu"

class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)

# Создаем единственный экземпляр конфига для импорта в других модулях
config = AppConfig()
