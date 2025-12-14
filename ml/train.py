"""Training Pipeline for ChestX-AI"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from ml.dataset import ChestXrayDataset
from ml.config import TrainingConfig
from app.models.densenet import ChestXrayDenseNet


def get_transforms(is_training: bool = True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                               factor=0.5, patience=3)
        self.scaler = GradScaler() if config.use_amp else None
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training on: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(config.gpu)
            print(f"GPU: {gpu_name}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        from sklearn.metrics import roc_auc_score
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        try:
            auc = roc_auc_score(all_labels, all_preds, average='macro')
        except ValueError:
            auc = 0.0
        
        return total_loss / len(self.val_loader), auc
    
    def save_checkpoint(self, val_loss, is_best=False):
        torch.save(self.model.state_dict(), self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pth")
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self):
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_auc = self.validate()
            self.scheduler.step(val_loss)
            
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, AUC={val_auc:.4f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(val_loss, is_best)
            
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("\nTraining Complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, gpu=args.gpu
    )
    
    train_dataset = ChestXrayDataset(config.data_dir, 'train', get_transforms(True))
    val_dataset = ChestXrayDataset(config.data_dir, 'val', get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    model = ChestXrayDenseNet(pretrained=True)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
