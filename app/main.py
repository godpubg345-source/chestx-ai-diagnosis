"""ChestX-AI: Medical Chest X-Ray Diagnosis System"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from app.api.routes import router
from app.services.inference import InferenceService

load_dotenv()

inference_service: InferenceService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service
    
    print("🚀 Starting ChestX-AI Medical Diagnosis System...")
    print("=" * 50)
    
    model_path = os.getenv("MODEL_PATH", "models/densenet121_chestxray.pth")
    device = os.getenv("DEVICE", "cuda")
    
    inference_service = InferenceService(model_path=model_path, device=device)
    
    print(f"✅ Model loaded on {inference_service.device}")
    print(f"✅ Ready to analyze chest X-rays!")
    print("=" * 50)
    
    yield
    
    print("👋 Shutting down ChestX-AI...")


app = FastAPI(
    title="ChestX-AI",
    description="AI-powered Chest X-Ray Diagnosis System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

app.include_router(router, prefix="/api")


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(static_path / "index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ChestX-AI",
        "model_loaded": inference_service is not None,
        "device": inference_service.device if inference_service else "not loaded"
    }


def get_inference_service() -> InferenceService:
    return inference_service
