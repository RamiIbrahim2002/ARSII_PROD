# config.py
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from databases import Database

# --- Logging Setup ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("ocr_app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ocr_app")

logger = setup_logging()

# --- Load Environment ---
def load_environment():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(dotenv_path=env_path)
    
    # Check for required environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set in environment")
        raise ValueError("DATABASE_URL not set in environment")
    
    return {
        "api_key": api_key,
        "database_url": database_url
    }

# --- Create clients ---
def get_openai_client():
    env = load_environment()
    return OpenAI(api_key=env["api_key"])

def get_database():
    env = load_environment()
    return Database(env["database_url"])