"""Grad-CAM Implementation for Explainable AI"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional


class GradCAM:
    
    def __init__(self, model: nn.Module, target_layer: str = "features"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = self._get_target_layer(target_layer)
        self._register_hooks()
    
    def _get_target_layer(self, layer_name: str) -> nn.Module:
        if hasattr(self.model, 'features'):
            return self.model.features
        elif hasattr(self.model, 'densenet'):
            return self.model.densenet.features
        raise ValueError(f"Could not find layer: {layer_name}")
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target = output[0, target_class]
        target.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class
    
    def generate_heatmap(self, input_tensor: torch.Tensor, original_image: Image.Image,
                         target_class: Optional[int] = None, alpha: float = 0.5,
                         colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        cam, _ = self.generate(input_tensor, target_class)
        
        original_size = original_image.size
        cam_resized = cv2.resize(cam, original_size)
        
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        original_np = np.array(original_image.convert("RGB"))
        overlaid = (1 - alpha) * original_np + alpha * cam_colored
        overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)
        
        return overlaid
