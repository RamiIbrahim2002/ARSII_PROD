# core/ocr.py
import os
import json
from fastapi import HTTPException, UploadFile
from openai import OpenAI
from config import get_openai_client, logger
from core.schema import CNAM_FORM_SCHEMA
from core.prompts import OCR_SYSTEM_PROMPT
from utils.image import get_blank_form_b64, encode_file_to_b64
from core.schema import CLAIM_EVAL_SCHEMA
from core.prompts import EVAL_SYSTEM_PROMPT

        
class OcrProcessor:
    def __init__(self):
        self.client = get_openai_client()
        self.blank_form_b64 = get_blank_form_b64()
        if self.blank_form_b64 is None:
            logger.error("Failed to load blank form template")

    async def process_form(self, filled_form: UploadFile):
        """Process a filled form and extract information using OCR."""
        request_id = os.urandom(4).hex()
        logger.info(f"[{request_id}] Extracting {filled_form.filename}")

        if self.blank_form_b64 is None:
            raise HTTPException(status_code=500, detail="Blank form template missing.")

        # Prepare the input images
        inputs = [{"type": "input_image", "image_url": f"data:image/png;base64,{self.blank_form_b64}"}]
        filled_data = await filled_form.read()
        filled_b64 = await encode_file_to_b64(filled_data)
        inputs.append({"type": "input_image", "image_url": f"data:image/png;base64,{filled_b64}"})

        try:
            # Send to OpenAI for OCR processing
            resp = self.client.responses.create(
                model="gpt-4o-2024-08-06",
                input=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT}, 
                    {"role": "user", "content": inputs}
                ],
                text={"format": {"type": "json_schema", "name": "cnam_form", "schema": CNAM_FORM_SCHEMA, "strict": True}}
            )
        except Exception as e:
            logger.error(f"[{request_id}] OpenAI error: {e}")
            raise HTTPException(status_code=500, detail="OCR service error.")

        try:
            result = json.loads(resp.output_text)
            logger.info(f"[{request_id}] Successfully extracted data from form")
            return filled_form.filename, result
        except Exception as e:
            logger.error(f"[{request_id}] JSON parse error: {e}")
            raise HTTPException(status_code=500, detail="Invalid OCR output.")
        
    async def evaluate_claim(self, extracted: dict) -> dict:
        """Ask OpenAI to run through reimbursement questions with structured JSON output."""
        # 1) JSON‑encode for the LLM
        body = json.dumps(extracted, ensure_ascii=False)
    
        try:
            resp = self.client.responses.create(
                model="gpt-4o-2024-08-06",
                input=[
                    {"role": "system",  "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",    "content": body}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "claim_evaluation",
                        "schema": CLAIM_EVAL_SCHEMA,
                        "strict": True
                    }
                }
            )
        except Exception as e:
            logger.error(f"Evaluation call failed: {e}")
            raise HTTPException(status_code=500, detail="Evaluation service error.")
    
        try:
            evaluation = json.loads(resp.output_text)
        except Exception as e:
            logger.error(f"Invalid evaluation JSON: {e}")
            raise HTTPException(status_code=500, detail="Invalid evaluation output.")
    
        logger.info(f"Claim evaluation result: {evaluation}")
        return evaluation
