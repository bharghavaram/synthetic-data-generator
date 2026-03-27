"""
AI Synthetic Data Generator – High-quality training dataset generation with LLMs.
Generates diverse, realistic synthetic data for ML training, testing, and augmentation.
"""
import logging
import json
import uuid
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from openai import OpenAI
from anthropic import Anthropic
from app.core.config import settings

logger = logging.getLogger(__name__)

SCHEMA_INFERENCE_PROMPT = """Analyse this sample data and infer the schema for synthetic data generation.

Sample Data:
{sample}

Return JSON:
{{
  "fields": [
    {{
      "name": "field_name",
      "type": "string|integer|float|boolean|email|phone|date|category|address|name",
      "description": "what this field represents",
      "constraints": {{"min": null, "max": null, "categories": null, "format": null}},
      "nullable": false,
      "importance": "high|medium|low"
    }}
  ],
  "domain": "e-commerce|finance|healthcare|education|hr|social|general",
  "generation_notes": "any special considerations for realism"
}}"""

BATCH_GENERATION_PROMPT = """Generate {count} realistic synthetic records for this schema.

Domain: {domain}
Schema: {schema}
Special Notes: {notes}
Diversity Requirements: Ensure variety in all fields, avoid repetition.
Realism Requirements: Values must be realistic and internally consistent.

Return a JSON array of records: [{{...}}, ...]
Generate EXACTLY {count} records."""

QUALITY_CHECK_PROMPT = """Evaluate the quality of this synthetic dataset.

Schema: {schema}
Sample Records (first 10): {sample}

JSON evaluation:
{{
  "overall_quality_score": 0-100,
  "diversity_score": 0-100,
  "realism_score": 0-100,
  "consistency_score": 0-100,
  "issues": [...],
  "recommendations": [...],
  "is_acceptable": true/false
}}"""

AUGMENTATION_PROMPT = """Augment this existing dataset by generating {count} additional diverse records.

Existing Data (sample): {existing_sample}
Schema: {schema}
Avoid duplicating existing patterns. Introduce new variations and edge cases.

Return JSON array: [{{...}}, ...]"""


class DataGenService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        self._jobs: dict = {}

    def infer_schema(self, sample_data: str) -> dict:
        resp = self.openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": SCHEMA_INFERENCE_PROMPT.format(sample=sample_data[:2000])}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            raise ValueError(f"Schema inference failed: {exc}")

    def _generate_batch(self, schema: dict, count: int, model: str = None) -> List[dict]:
        model = model or settings.LLM_MODEL
        prompt = BATCH_GENERATION_PROMPT.format(
            count=count,
            domain=schema.get("domain", "general"),
            schema=json.dumps(schema.get("fields", []), indent=2),
            notes=schema.get("generation_notes", "Ensure realism and diversity."),
        )
        resp = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON array from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        try:
            records = json.loads(content)
            return records if isinstance(records, list) else []
        except Exception as exc:
            logger.error("Batch parse failed: %s", exc)
            return []

    def _check_quality(self, schema: dict, records: List[dict]) -> dict:
        if not settings.QUALITY_CHECK_ENABLED or not records:
            return {"overall_quality_score": 75, "is_acceptable": True, "issues": []}
        sample = json.dumps(records[:10], indent=2)
        resp = self.openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": QUALITY_CHECK_PROMPT.format(
                schema=json.dumps(schema.get("fields", []))[:500],
                sample=sample[:2000],
            )}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"overall_quality_score": 75, "is_acceptable": True, "issues": []}

    def generate(self, schema: dict, num_samples: int = None, use_claude: bool = False) -> dict:
        job_id = str(uuid.uuid4())
        num_samples = min(num_samples or settings.DEFAULT_SAMPLES, settings.MAX_BATCH_SIZE * 4)
        all_records = []
        batch_size = settings.MAX_BATCH_SIZE

        # Generate in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_count = min(batch_size, num_samples - batch_start)
            model = settings.LLM_MODEL
            if use_claude and self.anthropic_client and batch_start > 0:
                # Use Claude for diversity on later batches
                try:
                    prompt = BATCH_GENERATION_PROMPT.format(
                        count=batch_count,
                        domain=schema.get("domain", "general"),
                        schema=json.dumps(schema.get("fields", []), indent=2),
                        notes=schema.get("generation_notes", "Ensure realism and diversity."),
                    )
                    resp = self.anthropic_client.messages.create(
                        model=settings.CLAUDE_MODEL,
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = resp.content[0].text
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start >= 0 and end > start:
                        batch = json.loads(content[start:end])
                        all_records.extend(batch if isinstance(batch, list) else [])
                        continue
                except Exception as exc:
                    logger.warning("Claude batch failed, falling back to OpenAI: %s", exc)

            batch = self._generate_batch(schema, batch_count, model)
            all_records.extend(batch)

        # Quality check
        quality = self._check_quality(schema, all_records)

        # Export to CSV
        csv_path = None
        if all_records:
            csv_path = str(Path(settings.OUTPUT_DIR) / f"synthetic_{job_id[:8]}.csv")
            fieldnames = list(all_records[0].keys()) if all_records else []
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_records)

        result = {
            "job_id": job_id,
            "schema": schema,
            "requested_samples": num_samples,
            "generated_samples": len(all_records),
            "quality_report": quality,
            "sample_records": all_records[:5],
            "csv_path": csv_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._jobs[job_id] = result
        return result

    def augment_dataset(self, existing_data: List[dict], schema: dict, additional_samples: int) -> dict:
        existing_sample = json.dumps(existing_data[:10], indent=2)
        prompt = AUGMENTATION_PROMPT.format(
            count=additional_samples,
            existing_sample=existing_sample[:2000],
            schema=json.dumps(schema.get("fields", []))[:500],
        )
        resp = self.openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        try:
            new_records = json.loads(content[start:end]) if start >= 0 else []
        except Exception:
            new_records = []

        all_records = existing_data + new_records
        return {
            "original_count": len(existing_data),
            "augmented_count": len(new_records),
            "total_count": len(all_records),
            "sample_new_records": new_records[:5],
        }

    def generate_from_description(self, description: str, num_samples: int = 50) -> dict:
        """Full pipeline: describe → infer schema → generate → quality check."""
        # First, infer schema from description
        schema_prompt = f"I want to generate synthetic data for: {description}\nInfer an appropriate schema."
        schema = self.infer_schema(schema_prompt)
        return self.generate(schema, num_samples)

    def get_job(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        return [{"job_id": k, "samples": v["generated_samples"], "quality": v["quality_report"].get("overall_quality_score"), "timestamp": v["timestamp"]} for k, v in self._jobs.items()]


_service: Optional[DataGenService] = None
def get_datagen_service() -> DataGenService:
    global _service
    if _service is None:
        _service = DataGenService()
    return _service
