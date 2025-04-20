# db/repositories.py
from datetime import datetime
import json
from typing import List, Dict, Any
from databases import Database
from config import logger
from db.models import OcrLogCreate, OcrLogRead

class OcrLogRepository:
    def __init__(self, database: Database):
        self.database = database

    async def create(self, log_entry: OcrLogCreate) -> OcrLogRead:
        sql = (
            "INSERT INTO ocr_logs (file_name, extracted_data) "
            "VALUES (:file_name, :extracted_data) RETURNING id, created_at"
        )
        values = {"file_name": log_entry.file_name, "extracted_data": json.dumps(log_entry.extracted_data)}
        
        try:
            row = await self.database.fetch_one(sql, values)
            logger.info(f"Logged to database with ID {row['id']}")
            return OcrLogRead(
                id=row['id'], 
                created_at=row['created_at'], 
                file_name=log_entry.file_name, 
                extracted_data=log_entry.extracted_data
            )
        except Exception as e:
            logger.error(f"DB insert error: {e}")
            raise

    async def get_logs(self, limit: int, offset: int) -> List[OcrLogRead]:
        sql = (
            "SELECT id, created_at, file_name, extracted_data FROM ocr_logs "
            "ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        )
        try:
            rows = await self.database.fetch_all(sql, {"limit": limit, "offset": offset})
            return [OcrLogRead(**row) for row in rows]
        except Exception as e:
            logger.error(f"DB query error: {e}")
            raise
    async def list_all(self) -> List[OcrLogRead]:
        """Fetch all OCR log entries from the database."""
        sql = "SELECT id, created_at, file_name, extracted_data FROM ocr_logs ORDER BY created_at DESC"
        try:
            rows = await self.database.fetch_all(sql)
            result = []
            for row in rows:
                # Convert row to a dict we can modify
                row_dict = dict(row)
                # Parse JSON string to dict
                if isinstance(row_dict['extracted_data'], str):
                    row_dict['extracted_data'] = json.loads(row_dict['extracted_data'])
                result.append(OcrLogRead(**row_dict))
            return result
        except Exception as e:
            logger.error(f"DB query error: {e}")
            raise
        
    async def list_by_date_range(self, from_date: datetime, to_date: datetime = None) -> List[OcrLogRead]:
        """Fetch OCR log entries within a specific date range."""
        if to_date is None:
            sql = "SELECT id, created_at, file_name, extracted_data FROM ocr_logs WHERE created_at >= :from_date ORDER BY created_at DESC"
            params = {"from_date": from_date}
        else:
            sql = "SELECT id, created_at, file_name, extracted_data FROM ocr_logs WHERE created_at >= :from_date AND created_at <= :to_date ORDER BY created_at DESC"
            params = {"from_date": from_date, "to_date": to_date}
        
        try:
            rows = await self.database.fetch_all(sql, params)
            result = []
            for row in rows:
                # Convert row to a dict we can modify
                row_dict = dict(row)
                # Parse JSON string to dict
                if isinstance(row_dict['extracted_data'], str):
                    row_dict['extracted_data'] = json.loads(row_dict['extracted_data'])
                result.append(OcrLogRead(**row_dict))
            return result
        except Exception as e:
            logger.error(f"DB query error: {e}")
            raise