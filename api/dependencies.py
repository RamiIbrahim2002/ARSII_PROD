print("Loading api/dependencies.py")
from core.ocr import OcrProcessor
from db.repositories import OcrLogRepository
from db.database import database

def get_ocr_processor():
    print("Called get_ocr_processor")
    return OcrProcessor()

def get_ocr_repository():
    print("Called get_ocr_repository")
    return OcrLogRepository(database)