# db/database.py
from databases import Database
from config import get_database, logger

database = get_database()

async def connect_db():
    await database.connect()
    logger.info("Connected to database")

async def disconnect_db():
    await database.disconnect()
    logger.info("Disconnected from database")