# 🫁 ChestX-AI: Medical Chest X-Ray Diagnosis System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%2012.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

<p align="center">
  <img src="assets/demo.gif" alt="ChestX-AI Demo" width="800">
</p>

> 🏥 **Production-grade AI system** for detecting 14 thoracic diseases from chest X-rays using deep learning with explainable AI (Grad-CAM).

## ✨ Features

- 🔬 **14 Disease Detection** - Trained on NIH ChestX-ray14 dataset
- 🧠 **DenseNet121 Architecture** - State-of-the-art transfer learning
- 🎯 **Grad-CAM Visualization** - See where the AI focuses
- ⚡ **Real-time Inference** - < 500ms response time
- 🎨 **Premium UI** - Modern glassmorphism design
- 🐳 **Docker Ready** - One-command deployment
- 🔄 **CI/CD Pipeline** - Automated testing with GitHub Actions

## 🦠 Detectable Diseases

| Disease | Disease | Disease |
|---------|---------|---------|
| Atelectasis | Cardiomegaly | Effusion |
| Infiltration | Mass | Nodule |
| Pneumonia | Pneumothorax | Consolidation |
| Edema | Emphysema | Fibrosis |
| Pleural Thickening | Hernia | |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (RTX 4060 or better recommended)
- 8GB+ VRAM for training

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chestx-ai-diagnosis.git
cd chestx-ai-diagnosis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser.

## 🏗️ Project Structure

```
chestx-ai-diagnosis/
├── app/                    # FastAPI Application
│   ├── main.py            # Entry point
│   ├── api/               # API routes
│   ├── models/            # ML models
│   ├── services/          # Business logic
│   └── static/            # Frontend UI
├── ml/                     # Training Pipeline
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation metrics
│   └── dataset.py         # Data loading
├── models/                 # Saved model weights
├── data/                   # Dataset directory
├── tests/                  # Unit tests
└── docker/                 # Docker configuration
```

## 🧠 Model Architecture

```
DenseNet121 (ImageNet Pretrained)
         ↓
    Global Avg Pool
         ↓
   Dropout (0.5)
         ↓
    FC Layer (1024 → 14)
         ↓
      Sigmoid
         ↓
   14 Disease Probabilities
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Mean AUC-ROC | 0.84+ |
| Inference Time | ~200ms (GPU) |
| Model Size | ~28MB |

## 🎨 Screenshots

<p align="center">
  <img src="assets/screenshot1.png" alt="Upload Screen" width="400">
  <img src="assets/screenshot2.png" alt="Results Screen" width="400">
</p>

## 🐳 Docker Deployment

```bash
# Build image
docker build -t chestx-ai .

# Run container
docker run -p 8000:8000 --gpus all chestx-ai
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v --cov=app

# Lint check
ruff check app/
```

## 📖 API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/predict` | Upload X-ray for diagnosis |
| GET | `/api/diseases` | List all detectable diseases |

## 🔬 Training Your Own Model

```bash
# Download NIH ChestX-ray14 dataset
python ml/download_dataset.py

# Start training
python ml/train.py --epochs 50 --batch-size 32 --gpu 0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) for the ChestX-ray14 dataset
- [PyTorch](https://pytorch.org) team for the amazing framework
- [FastAPI](https://fastapi.tiangolo.com) for the modern web framework

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/AsfandYar">Asfand Yar</a>
</p>
