"""DenseNet121 Model for Chest X-Ray Classification"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ChestXrayDenseNet(nn.Module):
    
    DISEASES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]
    
    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )
        
        self.features = self.densenet.features
        self.classifier = self.densenet.classifier
        
        # Disable inplace ReLU for compatibility with Grad-CAM backward hooks
        for m in self.densenet.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = nn.functional.relu(features, inplace=False)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
    
    @classmethod
    def load_pretrained(cls, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        model = cls(pretrained=True)
        
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"[OK] Loaded weights from {checkpoint_path}")
            except FileNotFoundError:
                print(f"[WARN] Checkpoint not found: {checkpoint_path}")
                print("   Using ImageNet pretrained weights only")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint: {e}")
                print("   Using ImageNet pretrained weights only")
        
        return model.to(device)


def create_model(num_classes: int = 14, pretrained: bool = True, 
                 checkpoint_path: Optional[str] = None, device: str = "cuda"):
    model = ChestXrayDenseNet(num_classes=num_classes, pretrained=pretrained)
    
    if checkpoint_path:
        model = ChestXrayDenseNet.load_pretrained(checkpoint_path, device)
    else:
        model = model.to(device)
    
    return model
