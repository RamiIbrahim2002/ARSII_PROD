# api/routes/ocr.py
from datetime import datetime, timedelta
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from db.models import OcrLogCreate, OcrLogRead
from api.dependencies import get_ocr_repository
from core.ocr import OcrProcessor
from config import logger
from core.constants import TARIFFS
import random

def get_ocr_processor():
    return OcrProcessor()

router = APIRouter()


def count_missing_fields(obj: dict) -> int:
    """
    Recursively count how many values are either None or the literal string "NULL" (case‑insensitive).
    """
    missing = 0

    def _recurse(v):
        nonlocal missing
        if v is None or (isinstance(v, str) and v.strip().upper() == "NULL") or v is None or (isinstance(v, str) and v.strip().upper() == ""):
            missing += 1
        elif isinstance(v, dict):
            for sub in v.values():
                _recurse(sub)
        elif isinstance(v, list):
            for item in v:
                _recurse(item)

    _recurse(obj)
    return missing

@router.post("/extract", response_model=OcrLogRead)
async def extract(
    filled_form: UploadFile = File(...),
    ocr_processor = Depends(get_ocr_processor),
    ocr_repository = Depends(get_ocr_repository)
):
    # 1) OCR
    filename, extracted = await ocr_processor.process_form(filled_form)

    # 2) Validate: if more than 3 missing fields, reject early
    missing_count = count_missing_fields(extracted)
    if missing_count > 3:
        raise HTTPException(
            status_code=400,
            detail="Submit a valid doc please"
        )

    # 3) Save only if valid
    ocr_record = await ocr_repository.create(OcrLogCreate(
        file_name=filename,
        extracted_data=extracted
    ))

    # 4) Define 3 variants of the purchase and pick one at random
    #    POC assumes different quantities and corresponding totals
    purchase_variants = [
        {"code_acte": "MED046", "quantite": 1, "montant_total": extracted.get("montant_paye", 50.0)},
        {"code_acte": "MED045", "quantite": 2, "montant_total": extracted.get("montant_paye", 50.0) * 2},
        {"code_acte": "MED046", "quantite": 3, "montant_total": extracted.get("montant_paye", 50.0) * 3},
        {"code_acte": "MED045", "quantite": 2, "montant_total": extracted.get("montant_paye", 40.0) * 2},

    ]
    purchase = random.choice(purchase_variants)

    # 5) Prepare evaluation payload with the selected purchase variant
    eval_payload = {
        "form_data": extracted,
        "purchase":  purchase,
        "tariff":    TARIFFS["MED045"]
    }

    # 6) Call the evaluator
    evaluation = await ocr_processor.evaluate_claim(eval_payload)

    # 7) Return result
    return {
        **ocr_record.dict(),
        "evaluation": evaluation
    }

@router.get("/report", response_model=dict)
async def generate_report(
    report_type: str = "summary",
    time_period: str = "all",
    ocr_processor=Depends(get_ocr_processor),
    ocr_repository=Depends(get_ocr_repository)
):
    """
    Fetches OCR logs based on filters, runs a RAG-based report generation, and returns the report.
    """
    
    # 1) Retrieve filtered records from the database
    if time_period == "last_week":
        from_date = datetime.now() - timedelta(days=7)
        records = await ocr_repository.list_by_date_range(from_date=from_date)
    elif time_period == "last_month":
        from_date = datetime.now() - timedelta(days=30)
        records = await ocr_repository.list_by_date_range(from_date=from_date)
    else:
        records = await ocr_repository.list_all()
    
    # 2) Prepare documents with metadata for RAG
    docs = []
    for record in records:
        # serialize the extracted_data dict to a JSON string
        content_str = json.dumps(record.extracted_data, ensure_ascii=False)
        docs.append({
            "content": content_str,
            "metadata": {
                "document_id": record.id,
                "timestamp": record.created_at.isoformat(),
                "confidence_score": getattr(record, "confidence_score", 0.0),
                "document_type": getattr(record, "document_type", "unknown")
            }
        })

    # 3) Ask the OCR processor to generate a report using RAG
    report = await ocr_processor.generate_rag_report(
        documents=docs,
        report_type=report_type
    )

    # 4) Return the report payload
    return {
        "report": report,
        "document_count": len(records),
        "period": time_period,
        "generated_at": datetime.now().isoformat()
    }