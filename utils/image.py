# utils/image.py
import os
import base64
from config import logger

def get_blank_form_b64():
    """Load and encode blank form template."""
    blank_form_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'blank_cnam_form.png')
    try:
        with open(blank_form_path, "rb") as f:
            blank_form_b64 = base64.b64encode(f.read()).decode()
        logger.info(f"Loaded blank template from {blank_form_path}")
        return blank_form_b64
    except FileNotFoundError:
        logger.error(f"Blank form template not found at {blank_form_path}")
        return None

async def encode_file_to_b64(file_content):
    """Encode file content to base64."""
    return base64.b64encode(file_content).decode()