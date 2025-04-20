from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List
from db.models import OcrLogRead
from api.dependencies import get_ocr_repository
from config import logger

router = APIRouter()

@router.get("/logs", response_model=List[OcrLogRead])
async def get_logs(
    limit: int = Query(20, ge=1, le=100), 
    offset: int = Query(0, ge=0),
    ocr_repository = Depends(get_ocr_repository)
):
    try:
        logs = await ocr_repository.get_logs(limit, offset)
        return logs
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving logs")