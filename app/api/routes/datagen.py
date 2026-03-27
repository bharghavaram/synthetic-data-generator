"""Synthetic Data Generator – API routes."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.services.datagen_service import DataGenService, get_datagen_service

router = APIRouter(prefix="/datagen", tags=["Synthetic Data"])

class SchemaInferRequest(BaseModel):
    sample_data: str

class GenerateRequest(BaseModel):
    schema: dict
    num_samples: Optional[int] = 100
    use_claude: Optional[bool] = False

class DescriptionRequest(BaseModel):
    description: str
    num_samples: Optional[int] = 50

class AugmentRequest(BaseModel):
    existing_data: List[dict]
    schema: dict
    additional_samples: int = 50

@router.post("/infer-schema")
async def infer_schema(req: SchemaInferRequest, svc: DataGenService = Depends(get_datagen_service)):
    if not req.sample_data.strip():
        raise HTTPException(400, "sample_data cannot be empty")
    return svc.infer_schema(req.sample_data)

@router.post("/generate")
async def generate(req: GenerateRequest, svc: DataGenService = Depends(get_datagen_service)):
    if not req.schema:
        raise HTTPException(400, "schema is required")
    if req.num_samples and req.num_samples > 500:
        raise HTTPException(400, "Max 500 samples per request")
    return svc.generate(req.schema, req.num_samples, req.use_claude)

@router.post("/generate-from-description")
async def generate_from_description(req: DescriptionRequest, svc: DataGenService = Depends(get_datagen_service)):
    if len(req.description.strip()) < 10:
        raise HTTPException(400, "Description too short")
    return svc.generate_from_description(req.description, req.num_samples)

@router.post("/augment")
async def augment(req: AugmentRequest, svc: DataGenService = Depends(get_datagen_service)):
    if not req.existing_data:
        raise HTTPException(400, "existing_data cannot be empty")
    if req.additional_samples > 200:
        raise HTTPException(400, "Max 200 additional samples")
    return svc.augment_dataset(req.existing_data, req.schema, req.additional_samples)

@router.get("/jobs")
async def list_jobs(svc: DataGenService = Depends(get_datagen_service)):
    return {"jobs": svc.list_jobs()}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, svc: DataGenService = Depends(get_datagen_service)):
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job

@router.get("/health")
async def health():
    return {"status": "ok", "service": "AI Synthetic Data Generator – LLM-powered training data generation"}
