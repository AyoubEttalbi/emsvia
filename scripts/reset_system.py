#!/usr/bin/env python3
"""
Reset Script - Delete all faces, data, and model embeddings for GPU project
"""

import shutil
import sys
import os
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    STUDENT_IMAGES_DIR,
    EMBEDDINGS_DIR,
    UNKNOWN_FACES_DIR,
    DATABASE_URL
)
from database.models import Base
from database.crud import AttendanceDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def confirm_deletion():
    """Ask user to confirm deletion"""
    print("=" * 60)
    print("⚠️  WARNING: DATA RESET (GPU PROJECT)")
    print("=" * 60)
    print("\nThis will DELETE all of the following in /home/ayoub/projects/Ai-Ml/emsvia:")
    print("  ✓ All student face images")
    print("  ✓ All face embeddings (cache.pkl and database vectors)")
    print("  ✓ All unknown face captures")
    print("  ✓ All database records (attendance, students, history)")
    print("\n" + "=" * 60)
    print("⚠️  THIS ACTION CANNOT BE UNDONE!")
    print("=" * 60)
    
    confirm = input("\nType 'DELETE' to confirm: ")
    return confirm == "DELETE"

def delete_directory_contents(directory: Path):
    """Delete all contents of a directory while keeping the directory itself"""
    if not directory.exists():
        print(f"  Directory does not exist: {directory}")
        return 0
    
    count = 0
    for item in directory.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
        except Exception as e:
            print(f"  Error deleting {item}: {e}")
    
    return count

def reset_database():
    """Drop and recreate all database tables"""
    try:
        print("\n🗄️  Resetting database...")
        db_manager = AttendanceDB(DATABASE_URL)
        Base.metadata.drop_all(bind=db_manager.engine)
        Base.metadata.create_all(bind=db_manager.engine)
        print("  ✓ Database reset successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error resetting database: {e}")
        return False

def main():
    print("\n🔄 Face Recognition System - Data Reset\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Force mode enabled (skipping confirmation)")
    elif not confirm_deletion():
        print("\n❌ Reset cancelled.")
        return
    
    print("\n" + "=" * 60)
    print("🗑️  Deleting data...")
    print("=" * 60)
    
    # Delete student images
    print(f"\n📁 Cleaning: {STUDENT_IMAGES_DIR}")
    count = delete_directory_contents(STUDENT_IMAGES_DIR)
    print(f"  ✓ Deleted {count} items")
    
    # Delete embeddings
    print(f"\n📁 Cleaning: {EMBEDDINGS_DIR}")
    count = delete_directory_contents(EMBEDDINGS_DIR)
    print(f"  ✓ Deleted {count} items")
    
    # Delete unknown faces
    print(f"\n📁 Cleaning: {UNKNOWN_FACES_DIR}")
    count = delete_directory_contents(UNKNOWN_FACES_DIR)
    print(f"  ✓ Deleted {count} items")
    
    # Reset database
    reset_database()
    
    print("\n" + "=" * 60)
    print("✅ RESET COMPLETE")
    print("=" * 60)
    print("\nAll data has been cleared!")
    print("You can now start fresh with: python scripts/collect_student_data.py")
    print()

if __name__ == "__main__":
    main()
