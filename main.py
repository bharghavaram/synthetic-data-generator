"""Synthetic Data Generator – FastAPI Entry Point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.datagen import router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

app = FastAPI(
    title="AI Synthetic Data Generator",
    description="Generate high-quality synthetic training datasets using GPT-4o and Claude. Supports schema inference from samples, description-based generation, dataset augmentation, and automated quality checks.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "AI Synthetic Data Generator",
        "version": "1.0.0",
        "generation_modes": ["schema_based", "description_based", "augmentation"],
        "supported_domains": ["e-commerce", "finance", "healthcare", "education", "hr", "social", "general"],
        "features": ["Automatic schema inference", "Multi-model generation (GPT-4o + Claude for diversity)", "Automated quality scoring", "CSV export", "Dataset augmentation"],
        "max_samples_per_request": 500,
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
