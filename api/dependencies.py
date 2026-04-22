from sqlalchemy.orm import Session
from database.crud import AttendanceDB
from config.settings import DATABASE_URL
import os

# Initialize the global DB manager
# We use the DATABASE_URL from settings
db_manager = AttendanceDB(DATABASE_URL)

def get_db():
    """
    Dependency to provide a database session for each request.
    Handles automatic closing of the session.
    """
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

def get_db_manager():
    """Returns the global database manager instance."""
    return db_manager
