from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import List, Optional

# --- Student Schemas ---

class StudentBase(BaseModel):
    student_id: str = Field(..., json_schema_extra={"example": "STU001"})
    first_name: str = Field(..., json_schema_extra={"example": "Ayoub"})
    last_name: str = Field(..., json_schema_extra={"example": "Ettalbi"})
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None

class StudentResponse(StudentBase):
    id: int
    enrollment_date: datetime
    is_active: bool
    model_config = ConfigDict(from_attributes=True)

# --- Attendance Schemas ---

class AttendanceBase(BaseModel):
    student_id: int
    status: str = "present"
    camera_id: str = "main_camera"
    confidence_score: float = 1.0

class AttendanceCreate(AttendanceBase):
    image_path: Optional[str] = None

class AttendanceUpdate(BaseModel):
    status: Optional[str] = None
    timestamp: Optional[datetime] = None

class AttendanceResponse(AttendanceBase):
    id: int
    timestamp: datetime
    image_path: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

# --- Unknown Face Schemas ---

class UnknownFaceResponse(BaseModel):
    id: int
    timestamp: datetime
    image_path: str
    reviewed: bool
    confidence: Optional[float] = None
    model_config = ConfigDict(from_attributes=True)

# --- Global Stats Schema ---

class SystemStats(BaseModel):
    total_students: int
    attendance_today: int
    unknown_faces_pending: int
    gpu_active: bool
