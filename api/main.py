from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import uvicorn
import logging
import cv2
import time



# Import core modules
from api import schemas
from api.dependencies import get_db

# Import routers
from api.routes import students, attendance


app = FastAPI(
    title="EMSVIA Attendance API",
    description="Backend API for the Industrial-Grade Face Recognition Attendance System",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Streamlit's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to EMSVIA Attendance API",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/api/stats", response_model=schemas.SystemStats)
def get_system_stats(db: Session = Depends(get_db)):
    """Retrieve high-level system statistics for the dashboard."""
    from database.models import Student, AttendanceRecord, UnknownFace
    from datetime import datetime, time
    from api.dependencies import get_db_manager
    from models.gpu_manager import GPUModelManager
    
    db_manager = get_db_manager()
    gpu_mgr = GPUModelManager()
    
    # Calculate today's start
    today_start = datetime.combine(datetime.now().date(), time.min)
    
    total_students = db.query(Student).count()
    attendance_today = db.query(AttendanceRecord).filter(AttendanceRecord.timestamp >= today_start).count()
    unknown_pending = db.query(UnknownFace).filter(UnknownFace.reviewed == False).count()
    
    return {
        "total_students": total_students,
        "attendance_today": attendance_today,
        "unknown_faces_pending": unknown_pending,
        "gpu_active": gpu_mgr.is_gpu_ready()
    }


@app.get("/api/unknown/pending", response_model=List[schemas.UnknownFaceResponse])
def get_pending_unknowns(db: Session = Depends(get_db)):
    """Retrieve unreviewed unknown faces."""
    from database.models import UnknownFace
    return db.query(UnknownFace).filter(UnknownFace.reviewed == False).order_by(UnknownFace.timestamp.desc()).limit(10).all()

@app.patch("/api/unknown/{face_id}/review", response_model=bool)
def review_unknown_face(face_id: int, db: Session = Depends(get_db)):
    """Mark an unknown face as reviewed."""
    from database.models import UnknownFace
    face = db.query(UnknownFace).filter(UnknownFace.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    face.reviewed = True
    db.commit()
    return True

@app.get("/api/cameras/stream")
async def video_stream(cam_id: int = 0):
    """MJPEG Video Stream with specified camera index."""
    def gen_frames(cid):
        cap = cv2.VideoCapture(cid)
        if not cap.isOpened():
            return
        
        # Low resolution for streaming speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04) # ~25 FPS
        
        cap.release()
            
    return StreamingResponse(gen_frames(cam_id), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/api/cameras/detect")
def detect_cameras():
    """Scan and return working camera indices."""
    working = []
    # Scan up to 10 indices
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    working.append({
                        "id": i,
                        "name": f"Camera {i}"
                    })
                cap.release()
        except:
            continue
    return working

@app.post("/api/embeddings/rebuild")
async def rebuild_embeddings():
    """Trigger the embedding generation script."""
    import subprocess
    import sys
    try:
        # Run scripts/generate_embeddings.py as a separate process to avoid blocking
        # and to ensure a clean model context
        process = subprocess.Popen(
            [sys.executable, "scripts/generate_embeddings.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        return {"status": "processing", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def get_health():
    """Retrieve detailed hardware health stats."""
    from models.gpu_manager import GPUModelManager
    gpu_mgr = GPUModelManager()
    
    return {
        "gpu": gpu_mgr.monitor_memory(),
        "system": {
            "status": "online",
            "version": "2.1.0-gpu",
            "engine": "Async-TensorFlow-XLA"
        }
    }

@app.get("/api/system/logs")
def get_system_logs():
    """Read the last 50 lines of the system logs."""
    from config.settings import LOG_FILE
    import os
    if not os.path.exists(LOG_FILE):
        return {"logs": ["Log file not found."]}
    
    try:
        with open(LOG_FILE, "r") as f:
            # Efficiently read last 50 lines
            lines = f.readlines()
            return {"logs": [line.strip() for line in lines[-50:]]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}

@app.delete("/api/unknown/{face_id}", status_code=204)
def delete_unknown_face(face_id: int, db: Session = Depends(get_db)):
    """Delete an unknown face record."""
    from database.models import UnknownFace
    face = db.query(UnknownFace).filter(UnknownFace.id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face not found")
    db.delete(face)
    db.commit()
    return None






# Include routers
app.include_router(students.router, prefix="/api/students", tags=["Students"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["Attendance"])


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
