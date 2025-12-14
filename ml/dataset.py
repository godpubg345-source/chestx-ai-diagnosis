"""Dataset handling for ChestX-ray14"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class ChestXrayDataset(Dataset):
    
    DISEASES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    
    def __init__(self, data_dir: str = "data", split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.images_dir = self.data_dir / "images"
        
        labels_file = self.data_dir / f"{split}_labels.csv"
        
        if labels_file.exists():
            self.labels_df = pd.read_csv(labels_file)
            self.image_names = self.labels_df['image_name'].tolist()
        else:
            self._load_nih_format()
    
    def _load_nih_format(self):
        data_entry = self.data_dir / "Data_Entry_2017.csv"
        
        if data_entry.exists():
            df = pd.read_csv(data_entry)
            for disease in self.DISEASES:
                df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
            
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            train_size = int(0.8 * len(df))
            val_size = int(0.1 * len(df))
            
            if self.split == 'train':
                indices = indices[:train_size]
            elif self.split == 'val':
                indices = indices[train_size:train_size+val_size]
            else:
                indices = indices[train_size+val_size:]
            
            self.labels_df = df.iloc[indices].reset_index(drop=True)
            self.image_names = self.labels_df['Image Index'].tolist()
        else:
            self.labels_df = pd.DataFrame()
            self.image_names = []
            if self.images_dir.exists():
                self.image_names = [f.name for f in self.images_dir.glob("*.png")][:100]
    
    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        
        if not image_path.exists():
            for subdir in self.images_dir.iterdir():
                if subdir.is_dir():
                    alt_path = subdir / image_name
                    if alt_path.exists():
                        image_path = alt_path
                        break
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        if len(self.labels_df) > 0:
            if 'image_name' in self.labels_df.columns:
                row = self.labels_df[self.labels_df['image_name'] == image_name]
            else:
                row = self.labels_df.iloc[idx:idx+1]
            
            if len(row) > 0:
                labels = row[self.DISEASES].values[0].astype(np.float32)
            else:
                labels = np.zeros(len(self.DISEASES), dtype=np.float32)
        else:
            labels = np.zeros(len(self.DISEASES), dtype=np.float32)
        
        return image, torch.tensor(labels)
