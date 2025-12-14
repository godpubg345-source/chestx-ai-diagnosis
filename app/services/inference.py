"""Inference Service for ChestX-AI"""

import io
import base64
import time
from typing import Dict, Optional, Any

import torch
from PIL import Image
from torchvision import transforms

from app.models.densenet import ChestXrayDenseNet, create_model
from app.models.gradcam import GradCAM


class InferenceService:
    
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"
        
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()
        self.gradcam = GradCAM(self.model)
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU: {gpu_name}")
    
    def _load_model(self) -> ChestXrayDenseNet:
        print(f"Loading model on {self.device}...")
        model = create_model(num_classes=14, pretrained=True, 
                           checkpoint_path=self.model_path, device=self.device)
        model.eval()
        return model
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.TRANSFORM(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        input_tensor = self.preprocess(image)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        probabilities = output.cpu().numpy()[0]
        
        predictions = []
        for disease, prob in zip(ChestXrayDenseNet.DISEASES, probabilities):
            severity = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"
            predictions.append({"disease": disease, "probability": float(prob), "severity": severity})
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "device": self.device
        }
    
    def generate_gradcam(self, image: Image.Image, target_class: Optional[int] = None) -> str:
        try:
            input_tensor = self.preprocess(image).requires_grad_(True)
            heatmap = self.gradcam.generate_heatmap(
                input_tensor=input_tensor, original_image=image,
                target_class=target_class, alpha=0.4
            )
            heatmap_image = Image.fromarray(heatmap)
        except ValueError as e:
            print(f"Grad-CAM failed: {e}")
            heatmap_image = image.convert("RGB").resize((224, 224))
        
        buffer = io.BytesIO()
        heatmap_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{base64_str}"
