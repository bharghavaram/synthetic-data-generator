> **📅 Project Period:** Jan 2026 – Feb 2026 &nbsp;|&nbsp; **Status:** Completed &nbsp;|&nbsp; **Author:** [Bharghava Ram Vemuri](https://github.com/bharghavaram)

# AI Synthetic Data Generator

> Generate high-quality training datasets using GPT-4o and Claude for ML and testing

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o-Generator-purple)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Claude-Diversity-orange)](https://anthropic.com)

## Overview

A production-ready synthetic data generation platform that uses **GPT-4o and Claude** to generate realistic, diverse, domain-specific datasets for ML training, software testing, and data augmentation — with automatic schema inference, quality scoring, and CSV export.

## Generation Modes

| Mode | Description |
|------|-------------|
| **Schema-based** | Provide a schema, get realistic records |
| **Description-based** | Describe what you need in plain English |
| **Augmentation** | Add diverse new samples to existing data |

## Supported Domains

`e-commerce` · `finance` · `healthcare` · `education` · `hr` · `social` · `general`

## Multi-Model Strategy

- **GPT-4o** – primary generation for consistency and realism
- **Claude** – used for later batches to maximise diversity
- **Quality checker** – LLM-based quality scoring (diversity, realism, consistency)

## Quick Start

```bash
git clone https://github.com/bharghavaram/synthetic-data-generator
cd synthetic-data-generator
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/datagen/infer-schema` | Infer schema from sample data |
| POST | `/api/v1/datagen/generate` | Generate from schema |
| POST | `/api/v1/datagen/generate-from-description` | Generate from plain text description |
| POST | `/api/v1/datagen/augment` | Augment existing dataset |
| GET | `/api/v1/datagen/jobs` | List generation jobs |

### Example: Generate from Description

```bash
curl -X POST "http://localhost:8000/api/v1/datagen/generate-from-description" \
  -H "Content-Type: application/json" \
  -d '{"description": "E-commerce orders with customer info, product details, prices, and order status", "num_samples": 100}'
```

### Example: Quality Report

```json
{
  "overall_quality_score": 87,
  "diversity_score": 85,
  "realism_score": 92,
  "consistency_score": 88,
  "is_acceptable": true,
  "issues": [],
  "recommendations": ["Add more edge case values for order_status"]
}
```
