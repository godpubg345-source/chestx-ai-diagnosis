"""ChestX-AI: Medical Chest X-Ray Diagnosis System"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv

from app.api.routes import router
from app.services.inference import InferenceService

load_dotenv()

inference_service: InferenceService = None


def get_inference_service() -> InferenceService:
    global inference_service
    if inference_service is None:
        model_path = os.getenv("MODEL_PATH", "models/densenet121_chestxray.pth")
        device = os.getenv("DEVICE", "cuda")
        inference_service = InferenceService(model_path=model_path, device=device)
    return inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_service
    
    print("Starting ChestX-AI Medical Diagnosis System...")
    print("=" * 50)
    
    service = get_inference_service()
    print(f"Model loaded on {service.device}")
    print("Ready to analyze chest X-rays!")
    print("=" * 50)
    
    yield
    
    print("Shutting down ChestX-AI...")


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

static_path = Path(__file__).resolve().parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>ChestX-AI is running!</h1><p>Visit <a href='/docs'>/docs</a> for API.</p>")


@app.get("/health")
async def health_check():
    service = get_inference_service()
    return {
        "status": "healthy",
        "service": "ChestX-AI",
        "model_loaded": service is not None,
        "device": service.device if service else "not loaded"
    }
