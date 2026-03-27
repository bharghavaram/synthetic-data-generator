"""Tests for Synthetic Data Generator."""
import pytest
from unittest.mock import patch, MagicMock
from app.core.config import settings

def test_settings():
    assert settings.MAX_BATCH_SIZE == 50
    assert settings.DEFAULT_SAMPLES == 100
    assert settings.DIVERSITY_THRESHOLD == 0.7

def test_output_dir_creation():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with patch("app.services.datagen_service.OpenAI"), \
             patch("app.services.datagen_service.Anthropic"), \
             patch.object(__import__("app.core.config", fromlist=["settings"]).settings, "OUTPUT_DIR", tmp):
            from app.services.datagen_service import DataGenService
            svc = DataGenService()
            assert Path(tmp).exists()

def test_schema_field_types():
    valid_types = {"string", "integer", "float", "boolean", "email", "phone", "date", "category", "address", "name"}
    schema = {
        "fields": [
            {"name": "user_id", "type": "integer"},
            {"name": "email", "type": "email"},
            {"name": "age", "type": "integer"},
        ]
    }
    for field in schema["fields"]:
        assert field["type"] in valid_types

def test_list_jobs_empty():
    with patch("app.services.datagen_service.OpenAI"), \
         patch("app.services.datagen_service.Anthropic"):
        from app.services.datagen_service import DataGenService
        svc = DataGenService()
        assert svc.list_jobs() == []

def test_get_nonexistent_job():
    with patch("app.services.datagen_service.OpenAI"), \
         patch("app.services.datagen_service.Anthropic"):
        from app.services.datagen_service import DataGenService
        svc = DataGenService()
        assert svc.get_job("nonexistent") is None

@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/datagen/health")
    assert resp.status_code == 200

@pytest.mark.asyncio
async def test_api_generate_too_many_samples():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/api/v1/datagen/generate", json={"schema": {"fields": []}, "num_samples": 600})
    assert resp.status_code == 400
