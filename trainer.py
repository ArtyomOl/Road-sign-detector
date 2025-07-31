# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
from typing import Dict

from config import config

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 lr_scheduler,
                 writer: SummaryWriter):
        self.model = model.to(config.project.device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}/{config.train.epochs} [Train]")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(config.project.device)
            labels = labels.to(config.project.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders['train'].dataset)
        
        return epoch_loss, epoch_acc.item()

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar = tqdm(self.dataloaders['val'], desc=f"Epoch {epoch+1}/{config.train.epochs} [Val]")

        for inputs, labels in progress_bar:
            inputs = inputs.to(config.project.device)
            labels = labels.to(config.project.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders['val'].dataset)
        
        return epoch_loss, epoch_acc.item()

    def train(self):
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(config.train.epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate_one_epoch(epoch)
            
            print(f"Epoch {epoch+1}/{config.train.epochs} -> "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Логирование в TensorBoard
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)


            # Обновление LR Scheduler
            self.lr_scheduler.step(val_loss)
            
            # Early Stopping и сохранение лучшей модели
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.project.model_save_path)
                print(f"Model saved to {config.project.model_save_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.train.patience:
                    print(f"Early stopping triggered after {config.train.patience} epochs.")
                    break
        
        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Загрузка лучшей модели для последующего тестирования
        self.model.load_state_dict(torch.load(config.project.model_save_path))
