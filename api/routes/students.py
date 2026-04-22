from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from pathlib import Path
from api.dependencies import get_db, get_db_manager
from api import schemas
from database.models import Student

router = APIRouter()

@router.get("/", response_model=List[schemas.StudentResponse])
def get_students(active_only: bool = True, db: Session = Depends(get_db)):
    """Retrieve all students."""
    db_manager = get_db_manager()
    return db_manager.get_all_students(db, active_only=active_only)

@router.get("/{student_id_str}", response_model=schemas.StudentResponse)
def get_student(student_id_str: str, db: Session = Depends(get_db)):
    """Retrieve a specific student by their string ID."""
    db_manager = get_db_manager()
    student = db_manager.get_student_by_id(db, student_id_str)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@router.post("/", response_model=schemas.StudentResponse, status_code=status.HTTP_201_CREATED)
async def create_student(
    student_id: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Enroll a new student with photos."""
    db_manager = get_db_manager()
    
    # 1. Check if exists
    existing = db_manager.get_student_by_id(db, student_id)
    if existing:
        raise HTTPException(status_code=400, detail="Student ID already registered")
    
    # 2. Create student record
    student = db_manager.add_student(
        db, 
        student_id=student_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone
    )
    
    if not student:
        raise HTTPException(status_code=500, detail="Could not create student")
    
    # 3. Save photos to data/student_images/{student_id} as expected by generate_embeddings.py
    student_dir = Path(f"data/student_images/{student_id}")
    student_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files):
        file_ext = os.path.splitext(file.filename)[1]
        if not file_ext: file_ext = ".jpg"
        file_path = student_dir / f"face_{i}{file_ext}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    # 4. Trigger embedding generation in background
    import subprocess
    import sys
    subprocess.Popen([sys.executable, "scripts/generate_embeddings.py"], cwd=os.getcwd())
            
    return student

