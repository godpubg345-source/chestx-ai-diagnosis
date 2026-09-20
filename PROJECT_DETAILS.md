# 🫁 ChestX-AI: Medical Chest X-Ray Diagnosis System

**Course**: Cloud Computing (SE – 3202)  
**Department**: Department of Software Engineering  
**Institution**: The University of Azad Jammu and Kashmir (UAJK)  
**Lab Instructor**: Engr. Fahad Nisar  
**Lab Task**: #05 – Deploying a Web App with a Backend Server  
**GitHub Repository**: [https://github.com/godpubg345-source/chestx-ai-diagnosis](https://github.com/godpubg345-source/chestx-ai-diagnosis)  

---

## 1. Project Overview

**ChestX-AI** is a production-grade, AI-powered Medical Diagnosis Web Application designed to analyze frontal chest X-ray radiographs and detect **14 distinct thoracic conditions** in real time. 

Beyond standard classification, the system incorporates **Explainable AI (Grad-CAM)** to visually highlight the anatomical regions within the lungs and thorax that influenced the model's prediction. This transparency assists clinicians and radiographers in validating model inference.

### Key Highlights
- **14-Disease Multi-Label Classification**: Simultaneously assesses chest radiographs for 14 pulmonary pathologies.
- **Deep Convolutional Architecture**: Built on **DenseNet121** with transfer learning from ImageNet.
- **Explainable AI (XAI)**: Generates gradient-weighted class activation mapping (**Grad-CAM**) heatmaps superimposed directly on the input radiograph.
- **High-Throughput Asynchronous Backend**: Powered by **FastAPI** and **Uvicorn** for low-latency (<500ms) request processing.
- **Hardware Acceleration**: Automatic CUDA GPU acceleration with graceful fallback to CPU.
- **Modern Glassmorphic Frontend**: Standalone, responsive web interface built with vanilla HTML5, CSS3, and JavaScript.

---

## 2. 14 Detectable Thoracic Diseases

The model is structured around the clinical labels established by the NIH ChestX-ray14 dataset:

| # | Disease | Clinical Description |
|---|---|---|
| 1 | **Atelectasis** | Partial or complete collapse of the lung or lobe |
| 2 | **Cardiomegaly** | Abnormal enlargement of the heart silhouette |
| 3 | **Effusion** | Abnormal accumulation of fluid within the pleural cavity |
| 4 | **Infiltration** | Density increase in pulmonary parenchyma (fluid, cells, or pus) |
| 5 | **Mass** | Solid localized lesion with a diameter greater than 3 cm |
| 6 | **Nodule** | Small, well-defined round pulmonary lesion (up to 3 cm) |
| 7 | **Pneumonia** | Inflammatory infection affecting the alveoli of one or both lungs |
| 8 | **Pneumothorax** | Trapped air in the pleural space causing lung collapse |
| 9 | **Consolidation** | Alveolar air spaces replaced by inflammatory exudate or liquid |
| 10 | **Edema** | Accumulation of fluid in lung tissues and air spaces |
| 11 | **Emphysema** | Destruction and enlargement of alveoli causing hyperinflation |
| 12 | **Fibrosis** | Formation of excess fibrous connective tissue (scarring) |
| 13 | **Pleural Thickening** | Calcification or thickening of the pleural lining |
| 14 | **Hernia** | Protrusion of an abdominal organ through a diaphragmatic defect |

---

## 3. Technology Stack & System Architecture

```
                      +-----------------------------+
                      |     Client Web Browser      |
                      |   (Glassmorphic HTML5/JS)   |
                      +--------------+--------------+
                                     |  HTTP (REST)
                                     v
                      +-----------------------------+
                      |      FastAPI + Uvicorn      |
                      |       (ASGI Web Server)     |
                      +--------------+--------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
           v                                                   v
+-----------------------+                           +-----------------------+
|  DenseNet121 Feature  |                           |   Grad-CAM Explainer  |
|  Extraction & Head    |                           |  Heatmap Generator    |
+-----------+-----------+                           +-----------+-----------+
            |                                                   |
            +-------------------------+-------------------------+
                                      |
                                      v
                      +-----------------------------+
                      |  JSON Response + Heatmap    |
                      |  (Predictions & Diagnostics)|
                      +-----------------------------+
```

### Stack Breakdown

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/)
- **Deep Learning Framework**: [PyTorch 2.1](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
- **Model Architecture**: DenseNet121 (`models.densenet121`) with custom classification head:
  - Global Average Pooling
  - Dropout ($p = 0.5$)
  - Fully Connected Layer ($1024 \to 14$)
  - Multi-label Sigmoid Activation
- **Computer Vision & Image Utilities**: OpenCV (`cv2`), Pillow (`PIL`), NumPy
- **Frontend Architecture**: Vanilla HTML5, CSS Variables, Glassmorphism design tokens, Asynchronous Fetch API
- **Deployment Targets**: Render, Railway, Fly.io, Koyeb, Docker

---

## 4. API Endpoints Specification

### 1. Root Webpage
- **Method**: `GET`
- **Path**: `/`
- **Response**: Serves the interactive user interface (`app/static/index.html`).

---

### 2. Service Health Check
- **Method**: `GET`
- **Path**: `/health`
- **Description**: Verifies backend availability, model initialization status, and compute device.
- **Response Sample**:
```json
{
  "status": "healthy",
  "service": "ChestX-AI",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### 3. Diseases Directory
- **Method**: `GET`
- **Path**: `/api/diseases`
- **Description**: Returns all 14 supported diseases along with clinical descriptions.
- **Response Sample**:
```json
{
  "diseases": [
    {
      "name": "Pneumonia",
      "description": "Infection that inflames air sacs in the lungs"
    },
    {
      "name": "Cardiomegaly",
      "description": "Enlargement of the heart"
    }
  ],
  "count": 14
}
```

---

### 4. Single Image Prediction & Explainability
- **Method**: `POST`
- **Path**: `/api/predict`
- **Payload**: `multipart/form-data` with form field `file` containing an image file (`image/jpeg`, `image/png`).
- **Response Sample**:
```json
{
  "success": true,
  "predictions": [
    { "disease": "Pneumonia", "probability": 0.784, "severity": "high" },
    { "disease": "Infiltration", "probability": 0.421, "severity": "medium" },
    { "disease": "Effusion", "probability": 0.125, "severity": "low" }
  ],
  "top_findings": [
    { "disease": "Pneumonia", "probability": 0.784, "severity": "high" }
  ],
  "heatmap": "data:image/png;base64,iVBORw0KGgo...",
  "inference_time_ms": 142.35,
  "device": "cuda"
}
```

---

### 5. Batch Image Prediction
- **Method**: `POST`
- **Path**: `/api/predict/batch`
- **Payload**: `multipart/form-data` with up to 10 files.
- **Response Sample**:
```json
{
  "results": [
    {
      "filename": "patient_001_xray.png",
      "result": { "success": true, "top_findings": [...] }
    }
  ],
  "count": 1
}
```

---

### 6. Interactive OpenAPI / Swagger Documentation
- **Method**: `GET`
- **Path**: `/docs` or `/redoc`
- **Description**: Automatic interactive documentation for inspecting schemas and executing live endpoint tests.

---

## 5. Repository File Structure

```
02-chestx-ai-diagnosis/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app definition, CORS, lifespan & static mounts
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API endpoints (/predict, /diseases, /health)
│   │   └── schemas.py          # Pydantic validation schemas
│   ├── models/
│   │   ├── __init__.py
│   │   ├── densenet.py         # DenseNet121 architecture with Grad-CAM compatibility
│   │   └── gradcam.py          # Grad-CAM forward/backward hook implementation
│   ├── services/
│   │   ├── __init__.py
│   │   └── inference.py        # Pipeline orchestrator: preprocessing, predict, heatmap
│   └── static/
│       └── index.html          # Responsive glassmorphism web UI
├── assets/
│   └── demo.png                # System user interface screenshot
├── ml/
│   ├── dataset.py              # PyTorch Dataset loader for ChestX-ray14
│   ├── evaluate.py             # ROC-AUC, Precision, Recall metric calculations
│   └── train.py                # Model training script
├── models/
│   └── .gitkeep                # Directory for saved model weights
├── .env                        # Local runtime environment configuration
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore specifications
├── Dockerfile                  # Container definition for containerized deployment
├── docker-compose.yml          # Multi-container service definitions
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 6. Local Setup & Execution

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA drivers

### Steps
1. **Activate Virtual Environment**:
   ```bash
   .\.venv\Scripts\activate
   ```
2. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Configure Environment (`.env`)**:
   ```env
   HOST=127.0.0.1
   PORT=8000
   DEBUG=true
   DEVICE=cuda # or cpu
   MODEL_PATH=models/densenet121_chestxray.pth
   ```
4. **Launch Backend Server**:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
5. **Access Application**:
   - Web UI: `http://127.0.0.1:8000`
   - API Docs: `http://127.0.0.1:8000/docs`
   - Health Check: `http://127.0.0.1:8000/health`

---

## 7. Cloud Deployment Guide (Lab Task #05 Compliance)

To satisfy the requirements of **Lab Task #05 (Deploying a Web App with a Backend Server)**:

### Platform: Render (Free Web Service)
1. **Sign in to Render**: Navigate to [render.com](https://render.com) and link your GitHub account.
2. **Create New Web Service**: Click **New +** $\to$ **Web Service**.
3. **Connect Repository**: Select `https://github.com/godpubg345-source/chestx-ai-diagnosis`.
4. **Service Configuration**:
   - **Name**: `chestx-ai-diagnosis`
   - **Environment**: `Python 3`
   - **Region**: Closest region (e.g., Frankfurt or Oregon)
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
5. **Environment Variables**:
   - `DEVICE` = `cpu` (PaaS free tiers use CPU instances)
   - `DEBUG` = `false`
6. **Trigger Deployment**: Click **Create Web Service** and monitor deployment logs.
7. **Verify Live URL**: Test the generated public URL (`https://chestx-ai-diagnosis.onrender.com/health` and `/`).

---

## 8. Lab Task Verification Checklist

- [x] **Step 1: Choose Your Project** - Complete FastAPI backend with functional routes (`/`, `/health`, `/api/diseases`, `/api/predict`).
- [x] **Step 2: Push to GitHub** - Public repository accessible at `https://github.com/godpubg345-source/chestx-ai-diagnosis`.
- [ ] **Step 3: Select Free PaaS** - Selected Render / Koyeb for cloud hosting.
- [ ] **Step 4: Configure & Deploy** - Connect repo, specify build and start commands, deploy live.
- [ ] **Step 5: End-to-End Verification** - Capture side-by-side screenshots (Local vs. Live URL) testing the diagnosis feature.
- [ ] **Document Generation** - Compile screenshots into `2023-SE-YOUR_ROLL_NO.docx` or `.pdf` for submission to Engr. Fahad Nisar.
