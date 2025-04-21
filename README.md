# ARSII

![Project Diagram](static/diag.png)

A FastAPI application for extracting data from CNAM reimbursement forms using OpenAI’s OCR and JSON-schema-driven pipeline, storing results in a database, and generating summary or analytical reports via retrieval-augmented generation (RAG).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Database Setup](#database-setup)
  - [Running Locally](#running-locally)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ARSII is an AI-powered service for automating the processing of CNAM (Tunisian national health insurance) reimbursement forms. It uses:

1. **OCR Extraction**: Compares a blank form template to a user-uploaded filled form and extracts structured data via OpenAI's image-to-text responses API.
2. **Validation & Logging**: Validates extracted data before persisting to a PostgreSQL-compatible database.
3. **Claim Evaluation**: Applies business rules (via JSON-schema-driven prompts) to approve, deny, or partially accept reimbursement claims.
4. **Reporting**: Generates summary, analysis, or trend reports across past submissions using RAG with ChromaDB and OpenAI embeddings/chat completions.


## Features

- **High-Accuracy OCR** with JSON-schema enforcement
- **Automatic Claim Evaluation** against CNAM tariff rules
- **Persistent Logging** of all form submissions and extraction results
- **Flexible Reporting** (summary, analysis, trends)
- **Dockerized & 12‑factor** friendly for cloud deployments


## Architecture

![Project Diagram](static/diag.png)

1. **Client** uploads a filled CNAM form via `/extract` endpoint.
2. **OCR Processor** compares images, calls OpenAI for structured extraction, applies missing-field validation.
3. **Database** stores valid submissions in `ocr_logs` table.
4. **Claim Evaluator** uses OpenAI to compute reimbursement logic.
5. **Reporting Module** retrieves past logs, constructs embeddings, and produces RAG-based reports.


## Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- PostgreSQL or Aiven Cloud SQL instance
- OpenAI API credentials


### Installation

```bash
git clone https://github.com/RamiIbrahim2002/ARSII_PROD.git
cd ARSII_PROD
python -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```


### Configuration

Create a `.env` file or set environment variables:

```bash
# OpenAI
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Database (PostgreSQL URL)
DATABASE_URL=postgres://user:pass@host:port/dbname?sslmode=require
```


### Database Setup

Execute your SQL migration to create the `ocr_logs` table:

```sql
CREATE TABLE ocr_logs (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMP DEFAULT NOW(),
  file_name VARCHAR NOT NULL,
  extracted_data JSONB NOT NULL
);
```


### Running Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.


## Usage

### API Endpoints

#### 1. Extract OCR Data & Evaluate Claim

```
POST /extract
Content-Type: multipart/form-data

Form Data:
- filled_form: (file) PNG/JPG of completed CNAM form
```

- **Response**: JSON payload of claim evaluation results conforming to the defined schema.


#### 2. Retrieve Logs

```
GET /logs?limit=20&offset=0
```

- **Response**: Array of logged `OcrLogRead` entries.


#### 3. Generate Report

```
GET /report?report_type=summary&time_period=last_month
```

- `report_type`: `summary` | `analysis` | `trends`
- `time_period`: `all` | `last_week` | `last_month`

- **Response**: `{ report: string, document_count: int, period: string, generated_at: timestamp }


## Deployment

### Docker

```bash
docker build -t arsi-prod .
docker run -e OPENAI_API_KEY=... -e DATABASE_URL=... -p 8000:8000 arsi-prod
```

### Heroku / Railway

- Uses `Procfile`:
  ```
  web: uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- Ensure `runtime.txt` lists `python-3.10.x`


## Project Structure

```
├── api
│   ├── routes
│   │   ├── ocr.py       # OCR + claim evaluation routes
│   │   └── logs.py      # retrieval of past submissions
│   └── dependencies.py  # DI for OCR processor & DB repo
├── core
│   ├── ocr.py           # OCR & RAG processor logic
│   ├── constants.py     # CNAM tariff definitions
│   ├── prompts.py       # system/user prompts for LLM
│   └── schema.py        # Pydantic schemas for JSON outputs
├── db
│   ├── database.py      # DB connect/disconnect
│   ├── models.py        # Pydantic models for logs
│   └── repositories.py  # CRUD operations on `ocr_logs`
├── static              # images (blank form template, diagrams)
├── utils               # helper for image encoding
├── main.py             # FastAPI app initialization
├── DockerFile
├── Procfile
├── requirements.txt
├── runtime.txt
└── README.md
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

