"""Pydantic Schemas for ChestX-AI API"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DiseaseInfo(BaseModel):
    name: str
    description: str


class DiseasesListResponse(BaseModel):
    diseases: List[DiseaseInfo]
    count: int


class PredictionResult(BaseModel):
    disease: str
    probability: float = Field(..., ge=0, le=1)
    severity: str


class DiagnosisResponse(BaseModel):
    success: bool = True
    predictions: List[PredictionResult]
    top_findings: List[PredictionResult]
    heatmap: Optional[str] = None
    inference_time_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    device: str
