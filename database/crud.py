import logging
import json
import numpy as np
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

from database.models import Base, Student, FaceEmbedding, AttendanceRecord, UnknownFace

logger = logging.getLogger(__name__)

class AttendanceDB:
    """
    CRUD operations for the Face Recognition Attendance System.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database engine and session factory.
        
        Args:
            database_url: Database connection URL
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # --- Student Operations ---

    def add_student(self, session: Session, student_id: str, first_name: str, last_name: str, 
                    email: Optional[str] = None, phone: Optional[str] = None) -> Optional[Student]:
        """Add a new student to the database."""
        try:
            student = Student(
                student_id=student_id,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone
            )
            session.add(student)
            session.commit()
            session.refresh(student)
            logger.info(f"Added student: {first_name} {last_name} ({student_id})")
            return student
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding student {student_id}: {e}")
            return None

    def get_student_by_id(self, session: Session, student_id: str) -> Optional[Student]:
        """Retrieve a student by their unique student_id string (e.g., 'STU001')."""
        return session.query(Student).filter(Student.student_id == student_id).first()

    def get_student_by_pk(self, session: Session, pk: int) -> Optional[Student]:
        """Retrieve a student by their integer database primary key."""
        return session.query(Student).filter(Student.id == pk).first()

    def get_all_students(self, session: Session, active_only: bool = True) -> List[Student]:
        """Retrieve all students."""
        query = session.query(Student)
        if active_only:
            query = query.filter(Student.is_active == True)
        return query.all()

    # --- Face Embedding Operations ---

    def add_face_embedding(self, session: Session, student_id: int, embedding: np.ndarray, 
                           image_path: Optional[str] = None, model_name: str = "Facenet512") -> bool:
        """Add a face embedding for a student."""
        try:
            # Serialize numpy array to JSON string
            embedding_json = json.dumps(embedding.tolist())
            
            face_embedding = FaceEmbedding(
                student_id=student_id,
                embedding_vector=embedding_json,
                image_path=image_path,
                model_name=model_name
            )
            session.add(face_embedding)
            session.commit()
            logger.info(f"Added face embedding for student ID {student_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding embedding for student {student_id}: {e}")
            return False

    def get_student_embeddings(self, session: Session, student_id: int) -> List[Dict[str, Any]]:
        """Retrieve all embeddings for a student as a list of dicts with numpy arrays."""
        embeddings = session.query(FaceEmbedding).filter(FaceEmbedding.student_id == student_id).all()
        result = []
        for emb in embeddings:
            result.append({
                "id": emb.id,
                "vector": np.array(json.loads(emb.embedding_vector)),
                "model": emb.model_name,
                "image_path": emb.image_path
            })
        return result

    def get_all_embeddings_for_recognition(self, session: Session) -> Dict[int, Dict[str, List[np.ndarray]]]:
        """Load all embeddings into memory for the recognizer (student_id: {model_name: [vectors]})."""
        all_embeddings = session.query(FaceEmbedding).all()
        db_embeddings = {}
        for emb in all_embeddings:
            sid = emb.student_id
            mname = emb.model_name or "Facenet512" # Fallback for legacy data
            
            if sid not in db_embeddings:
                db_embeddings[sid] = {}
            if mname not in db_embeddings[sid]:
                db_embeddings[sid][mname] = []
                
            db_embeddings[sid][mname].append(np.array(json.loads(emb.embedding_vector)))
        return db_embeddings

    # --- Attendance Operations ---

    def mark_attendance(self, session: Session, student_id: int, confidence: float, 
                        status: str = "present", camera_id: str = "main_camera", 
                        image_path: Optional[str] = None) -> Optional[AttendanceRecord]:
        """Log a new attendance record."""
        try:
            record = AttendanceRecord(
                student_id=student_id,
                confidence_score=confidence,
                status=status,
                camera_id=camera_id,
                image_path=image_path
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info(f"Attendance marked for student ID {student_id} (Status: {status})")
            return record
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error marking attendance for student {student_id}: {e}")
            return None

    def get_attendance_history(self, session: Session, student_id: Optional[int] = None, 
                               start_date: Optional[datetime] = None, 
                               end_date: Optional[datetime] = None) -> List[AttendanceRecord]:
        """Query attendance history with filters."""
        query = session.query(AttendanceRecord)
        if student_id:
            query = query.filter(AttendanceRecord.student_id == student_id)
        if start_date:
            query = query.filter(AttendanceRecord.timestamp >= start_date)
        if end_date:
            query = query.filter(AttendanceRecord.timestamp <= end_date)
        return query.order_by(AttendanceRecord.timestamp.desc()).all()

    # --- Unknown Face Operations ---

    def log_unknown_face(self, session: Session, image_path: str, confidence: Optional[float] = None) -> Optional[UnknownFace]:
        """Log an unrecognized face for review."""
        try:
            unknown = UnknownFace(
                image_path=image_path,
                confidence=confidence
            )
            session.add(unknown)
            session.commit()
            session.refresh(unknown)
            logger.info(f"Log unknown face: {image_path}")
            return unknown
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error logging unknown face: {e}")
            return None

    def get_unreviewed_faces(self, session: Session) -> List[UnknownFace]:
        """Retrieve all unreviewed unknown faces."""
        return session.query(UnknownFace).filter(UnknownFace.reviewed == False).all()

    def mark_unknown_face_reviewed(self, session: Session, unknown_id: int) -> bool:
        """Mark an unknown face as reviewed."""
        try:
            unknown = session.query(UnknownFace).filter(UnknownFace.id == unknown_id).first()
            if unknown:
                unknown.reviewed = True
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error marking unknown face {unknown_id} as reviewed: {e}")
            return False
