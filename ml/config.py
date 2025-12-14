"""Training Configuration"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "data"
    num_workers: int = 4
    num_classes: int = 14
    pretrained: bool = True
    dropout: float = 0.5
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    patience: int = 10
    gpu: int = 0
    checkpoint_dir: str = "models"


@dataclass  
class InferenceConfig:
    model_path: str = "models/best.pth"
    device: str = "cuda"
    threshold: float = 0.5
