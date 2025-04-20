
# db/models.py
from pydantic import BaseModel
from typing import Any, Dict
from datetime import datetime

class OcrLogCreate(BaseModel):
    file_name: str
    extracted_data: Dict[str, Any]

class OcrLogRead(BaseModel):
    id: int
    created_at: datetime
    file_name: str
    extracted_data: Dict[str, Any]
    confidence_score: float = 0.0  # Default value if not present
    document_type: str = "unknown"  # Default value if not present