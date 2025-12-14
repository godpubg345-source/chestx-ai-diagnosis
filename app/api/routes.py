"""API Routes for ChestX-AI"""

import io
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.api.schemas import DiagnosisResponse, DiseaseInfo, DiseasesListResponse

router = APIRouter()

DISEASES = {
    "Atelectasis": "Collapse or incomplete expansion of the lung",
    "Cardiomegaly": "Enlargement of the heart",
    "Effusion": "Fluid accumulation in the pleural space",
    "Infiltration": "Substance denser than air in the lung tissue",
    "Mass": "Lesion larger than 3cm in diameter",
    "Nodule": "Small growth or lump in the lung",
    "Pneumonia": "Infection that inflames air sacs in the lungs",
    "Pneumothorax": "Collapsed lung due to air in pleural space",
    "Consolidation": "Lung tissue filled with liquid instead of air",
    "Edema": "Excess fluid in the lungs",
    "Emphysema": "Damage to the air sacs in the lungs",
    "Fibrosis": "Scarring of lung tissue",
    "Pleural_Thickening": "Thickening of the pleural membrane",
    "Hernia": "Organ pushing through chest wall opening"
}


@router.get("/diseases", response_model=DiseasesListResponse)
async def list_diseases():
    diseases = [DiseaseInfo(name=name, description=desc) for name, desc in DISEASES.items()]
    return DiseasesListResponse(diseases=diseases, count=len(diseases))


@router.post("/predict", response_model=DiagnosisResponse)
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_types}")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        from app.main import get_inference_service
        service = get_inference_service()
        if service is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
        
        result = service.predict(image)
        heatmap_base64 = service.generate_gradcam(image)
        
        predictions = sorted(result["predictions"], key=lambda x: x["probability"], reverse=True)
        top_findings = [p for p in predictions if p["probability"] > 0.3]
        
        return DiagnosisResponse(
            success=True,
            predictions=predictions,
            top_findings=top_findings,
            heatmap=heatmap_base64,
            inference_time_ms=result["inference_time_ms"],
            device=result["device"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append({"filename": file.filename, "result": result})
        except HTTPException as e:
            results.append({"filename": file.filename, "error": e.detail})
    
    return {"results": results, "count": len(results)}
