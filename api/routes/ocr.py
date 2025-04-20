# api/routes/ocr.py
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
        if v is None or (isinstance(v, str) and v.strip().upper() == "NULL"):
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
